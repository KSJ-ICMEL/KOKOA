import os, sys, json
import numpy as np
from pymatgen.core import Structure

# === 1. Structure Loading ===
script_dir = os.path.dirname(os.path.abspath(__file__))
cif_path = os.path.join(script_dir, "LLZO.cif")
structure = Structure.from_file(cif_path)
N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

# Initialize Li sites with occupancy probability
initial_sites = []
for site in structure:
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

print(f"Li sites initialized: {len(initial_sites)}")

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for i, site in enumerate(structure):
    if "Li" not in [s.symbol for s in site.species.elements]:
        continue
    neighbors = []
    for nb in neighbors_data[i]:
        if "Li" in [s.symbol for s in structure[nb.index].species.elements]:
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((nb.index, cart_disp))
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with Correlation Correction ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start),
                    'current': np.array(start),
                    'charge_disp': np.zeros(3)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        kb = 8.617e-5  # eV/K
        # Base rate for single-particle hops (classical isolated barrier)
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

        # Precompute Li-Li coordinated neighbor lists (shared neighbors)
        # For each Li site, find adjacent Li sites that share a common empty site
        self.coordinated_pairs = self._build_coordinated_pairs()

    def _build_coordinated_pairs(self):
        """
        Build a map of possible coordinated two-ion exchanges:
        For a pair of occupied Li sites (i, j) that have a common neighboring site k,
        we allow a concerted event where one ion moves into k and the other ion moves
        along its network. We do not explicitly simulate synchronized hops, but this
        map is used to estimate local correlation when hops occur.
        """
        coord_pairs = {}
        for i in self.adj_list:
            neighbors_i = self.adj_list[i]
            nb_sites_i = set([nb_idx for nb_idx, _ in neighbors_i])
            for j in self.adj_list:
                if j <= i:
                    continue
                neighbors_j = self.adj_list[j]
                nb_sites_j = set([nb_idx for nb_idx, _ in neighbors_j])
                common = nb_sites_i.intersection(nb_sites_j)
                if common:
                    coord_pairs.setdefault(i, set()).add(j)
                    coord_pairs.setdefault(j, set()).add(i)
        return coord_pairs

    def _local_correlation_factor(self, site_index):
        """
        Estimate a local effective contribution of a hop at 'site_index' to net charge
        transport, based on nearby occupied Li that can participate in concerted motion.

        We use a simple correction inspired by the Haven ratio concept:
        - If the local region has more Li involved in possible concerted moves,
          a larger fraction of microscopic hops are correlated and do not
          contribute independently to macroscopic charge transport.
        - We approximate this via a local Haven-like factor H_loc in (0,1].

        Construction:
        H_loc = 1 / (1 + alpha * n_corr)

        where:
        - n_corr is the number of occupied coordinated neighbors around the site.
        - alpha is a tunable parameter that encodes strength of correlation.
          We relate it to the ratio of classical E_a to concerted E_concerted:

          alpha ~ (E_a / E_concerted - 1)

        In Tufail et al. and He et al., concerted barriers for LGPS/LLZO/LATP
        are ~0.20–0.27 eV, consistent with AIMD and experiments, while
        classical isolated barriers are higher. We assume user-supplied
        E_a (classical) and E_concerted for LLZO.
        """
        alpha = max(self.params['E_a'] / self.params['E_concerted'] - 1.0, 0.0)

        neighbors = self.coordinated_pairs.get(site_index, [])
        if not neighbors or alpha == 0.0:
            return 1.0

        n_corr = 0
        for j in neighbors:
            if self.occupancy[j] == 1:
                n_corr += 1

        if n_corr == 0:
            return 1.0

        H_loc = 1.0 / (1.0 + alpha * n_corr)
        # Bound between a minimum global Haven factor and 1
        H_min = self.params.get('H_min', 0.1)
        H_loc = max(H_loc, H_min)
        H_loc = min(H_loc, 1.0)
        return H_loc

    def run_step(self):
        events, rates, total = [], [], 0.0
        # Build event list; base microscopic rate still comes from single-ion barrier.
        for src in self.li_indices:
            # Local correlation factor for this source site
            H_loc = self._local_correlation_factor(src)
            # Effective rate maintains microscopic attempt frequency but will be
            # interpreted with reduced charge contribution.
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self.base_rate  # microscopic hop frequency
                    total += rate
                    # Store local Haven factor for this event to weight charge transport
                    events.append((src, tgt, vec, H_loc))
                    rates.append(total)

        if total == 0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1

        # Select and execute event
        idx = np.searchsorted(rates, np.random.uniform(0, total))
        src, tgt, vec, H_loc = events[idx]

        p_id = self.site_to_particle.pop(src)
        # Update tracer position (full microscopic displacement)
        self.particle_positions[p_id]['current'] += vec
        # Update correlated (charge-carrying) displacement scaled by local Haven factor
        self.particle_positions[p_id]['charge_disp'] += H_loc * vec

        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Tracer MSD from microscopic trajectories
        msd_tracer = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2

        # Charge MSD from correlation-corrected displacements
        msd_charge = np.mean([
            np.sum((p['charge_disp']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2

        # Diffusivities (cm^2/s), MSD(t) = 6 D t
        D_tracer = msd_tracer / (6.0 * self.current_time) * 1e-16
        D_charge = msd_charge / (6.0 * self.current_time) * 1e-16

        # Haven ratio H = D_charge / D_tracer
        H = D_charge / D_tracer if D_tracer > 0 else 0.0

        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)

        # Nernst-Einstein conductivity using charge diffusivity
        e = 1.602e-19
        kB_SI = 1.38e-23
        sigma = (n * e ** 2 * D_charge) / (kB_SI * self.params['T'])  # S/cm

        return msd_tracer, msd_charge, D_tracer, sigma, H

# === 4. Run Simulation ===
# E_a: classical isolated single-ion barrier (eV)
# E_concerted: concerted migration barrier from NEB/experiments (eV), ~0.26 eV for LLZO
sim_params = {
    'T': 300,
    'E_a': 0.30,
    'E_concerted': 0.26,
    'nu': 1e13,
    'volume': structure.volume,
    # Minimum local Haven factor to avoid unphysical suppression
    'H_min': 0.1
}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

target_time = 1000e-9  # 1000 ns timeout
log_interval = 100
sigma_history = []
H_history = []

while sim.current_time < target_time:
    if not sim.run_step():
        print("Deadlock - stopping")
        break
    if sim.step_count % log_interval == 0:
        msd_tracer, msd_charge, D_tracer, sigma, H = sim.calculate_properties()
        sigma_history.append(sigma)
        H_history.append(H)

        # Check convergence
        if len(sigma_history) > 1000:
            sigma_history.pop(0)
            H_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0

            avg_H = np.mean(H_history)
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD_tracer={msd_tracer:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, "
                f"Haven={avg_H:.3f}, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD_tracer={msd_tracer:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
            )

# Final result
msd_tracer, msd_charge, D_tracer, sigma, H = sim.calculate_properties()
D_charge = msd_charge / (6.0 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D_tracer={D_tracer:.4e} cm^2/s")
print(f"D_charge={D_charge:.4e} cm^2/s")
print(f"Haven ratio H={H:.3f}")
print(f"Conductivity: {sigma:.4e} S/cm")

# Save result to JSON
result = {
    "is_success": True,
    "conductivity": sigma,
    "diffusivity_tracer": D_tracer,
    "diffusivity_charge": D_charge,
    "msd_tracer": msd_tracer,
    "msd_charge": msd_charge,
    "haven_ratio": H,
    "simulation_time_ns": sim.current_time * 1e9,
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns; "
        f"Haven ratio={H:.3f}"
    )
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")