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
li_site_indices = []
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(i)

print(f"Li sites initialized: {len(initial_sites)}")

# Map global structure indices (Li only) to compact Li-site indices
global_to_li = {g_idx: li_idx for li_idx, g_idx in enumerate(li_site_indices)}
li_to_global = {li_idx: g_idx for g_idx, li_idx in global_to_li.items()}

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for li_idx, g_i in enumerate(li_site_indices):
    site = structure[g_i]
    neighbors = []
    for nb in neighbors_data[g_i]:
        g_j = nb.index
        # Only consider Li-Li connectivity
        if g_j in global_to_li:
            li_j = global_to_li[g_j]
            frac_diff = structure[g_j].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((li_j, cart_disp))
    adj_list[li_idx] = neighbors

print(f"Graph built for Li sublattice (cutoff={cutoff}A)")

# === 2b. Build Li–Li neighbor lists for Coulombic repulsion ===
# We use a short cutoff (~3 Å) to represent nearest-neighbor Li–Li repulsion, as in LLZO
ll_cutoff = 3.0  # Angstrom, Li-Li repulsive interaction range
ll_neighbors = {li_idx: [] for li_idx in range(len(li_site_indices))}

for li_idx, g_i in enumerate(li_site_indices):
    site = structure[g_i]
    for nb in neighbors_data[g_i]:
        g_j = nb.index
        if g_j in global_to_li:
            li_j = global_to_li[g_j]
            if li_j == li_idx:
                continue
            # Only include within Li-Li cutoff
            if nb.distance <= ll_cutoff + 1e-8:
                ll_neighbors[li_idx].append(li_j)

print(f"Li-Li environment neighbor list built (cutoff={ll_cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with environment-dependent barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params, ll_neighbors):
        self.params = params
        self.adj_list = adj_list
        self.ll_neighbors = ll_neighbors
        self.num_sites = len(initial_sites)

        # Occupancy on Li-only index basis
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Particle bookkeeping
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start, dtype=float),
                    'current': np.array(start, dtype=float)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # Constants
        self.kb = 8.617e-5  # eV/K
        self.E_a0 = params['E_a']   # base barrier (eV) for an isolated hop
        self.nu = params['nu']      # attempt frequency (1/s)
        # Repulsion strength per neighboring Li within ll_cutoff (eV)
        # Simple linear penalty consistent with "Coulomb repulsion raises energy"
        self.E_rep = params.get('E_rep', 0.03)

    def local_li_count(self, site_idx):
        """Count occupied Li neighbors around a given Li site."""
        neighbors = self.ll_neighbors.get(site_idx, [])
        if not neighbors:
            return 0
        return int(np.sum(self.occupancy[neighbors]))

    def hop_barrier(self, src, tgt):
        """
        Environment-dependent activation barrier.

        Simple linear model:
        E_a(src -> tgt) = E_a0 + E_rep * (n_tgt_env - n_src_env)

        where n_src_env is the number of occupied Li neighbors around src
        (excluding the hopping ion itself), and n_tgt_env is the number
        of occupied Li neighbors around tgt before the hop.

        This implements:
          - higher barrier into crowded regions (n_tgt_env large),
          - lower barrier when leaving crowded regions (n_src_env large),
        following the qualitative Coulomb-repulsion arguments.
        """
        # Count occupied Li neighbors excluding the hopping ion
        n_src_env = self.local_li_count(src)
        # For src, we must not include the particle itself, but ll_neighbors
        # never includes src, so this is already satisfied.

        # For tgt, count occupied neighbors before hop (src will still be occupied)
        n_tgt_env = self.local_li_count(tgt)

        dN = n_tgt_env - n_src_env
        E_a_env = self.E_a0 + self.E_rep * dN
        # Prevent negative barriers
        if E_a_env < 0.0:
            E_a_env = 0.0
        return E_a_env

    def hop_rate(self, src, tgt):
        """Compute rate for a hop using environment-dependent barrier."""
        E_a = self.hop_barrier(src, tgt)
        return self.nu * np.exp(-E_a / (self.kb * self.params['T']))

    def run_step(self):
        """
        One BKL step with environment-dependent barriers.

        We still enforce site exclusion (no double occupancy), but
        now each allowed hop has its own rate depending on local Li configuration.
        """
        events = []
        cum_rates = []
        total_rate = 0.0

        # Enumerate all possible hops for occupied sites
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                # Hard exclusion: target must be empty
                if self.occupancy[tgt] == 0:
                    rate = self.hop_rate(src, tgt)
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    cum_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # No possible moves (deadlock)

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cum_rates, r)
        src, tgt, vec = events[idx]

        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)

        return True

    def calculate_properties(self):
        if self.current_time == 0.0 or self.num_particles == 0:
            return 0.0, 0.0
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2
        # MSD(t) = 6 D t  ->  D in Å^2/s
        D_A2_per_s = msd / (6.0 * self.current_time)
        D = D_A2_per_s * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
# Slightly larger base barrier; environment will further modulate it.
sim_params = {
    'T': 300,
    'E_a': 0.30,       # eV, base single-ion migration barrier
    'nu': 1e13,        # 1/s
    'volume': structure.volume,
    'E_rep': 0.03      # eV per occupied Li neighbor difference
}

sim = KMCSimulator(structure, adj_list, initial_sites, sim_params, ll_neighbors)

target_time = 1000e-9  # 1000 ns timeout
log_interval = 100
sigma_history = []

while sim.current_time < target_time:
    if not sim.run_step():
        print("Deadlock - stopping")
        break
    if sim.step_count % log_interval == 0:
        msd, sigma = sim.calculate_properties()
        sigma_history.append(sigma)

        # Keep only last 1000 sigma values
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0.0

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criterion
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
            )

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6.0 * sim.current_time) * 1e-16 if sim.current_time > 0.0 else 0.0

print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D={D:.4e} cm^2/s")
print(f"Conductivity: {sigma:.4e} S/cm")

# Save result to JSON
result = {
    "is_success": True,
    "conductivity": sigma,
    "diffusivity": D,
    "msd": msd,
    "simulation_time_ns": sim.current_time * 1e9 if sim.current_time > 0.0 else 0.0,
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns"
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")