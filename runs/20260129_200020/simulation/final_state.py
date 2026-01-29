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
li_site_indices = []  # map Li sites to indices in occupancy array
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state, "struct_index": i})
        li_site_indices.append(i)

print(f"Li sites initialized: {len(initial_sites)}")

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

# Build a mapping from structure index to Li-site index
struct_to_li = {site_info["struct_index"]: idx for idx, site_info in enumerate(initial_sites)}

for i, site in enumerate(structure):
    if i not in struct_to_li:
        continue
    li_src_idx = struct_to_li[i]
    neighbors = []
    for nb in neighbors_data[i]:
        nb_idx = nb.index
        if nb_idx in struct_to_li:
            li_tgt_idx = struct_to_li[nb_idx]
            frac_diff = structure[nb_idx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((li_tgt_idx, cart_disp))
    adj_list[li_src_idx] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator with configuration-dependent rates ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.structure = structure

        # Occupancy is defined only on Li sites
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Particle tracking
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[idx] = p_id
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
        self.nu = params['nu']
        self.E_a0 = params['E_a']  # base activation energy in eV
        self.T = params['T']

        # Parameters for configuration-dependent barrier
        # These are phenomenological scaling factors; they do not introduce new physics,
        # but modulate the base Arrhenius barrier according to local occupancy
        self.delta_E_nn = params.get('delta_E_nn', 0.01)  # eV per occupied neighbor
        self.nn_cutoff = params.get('nn_cutoff', 4.0)     # Å, same as graph cutoff

        # Precompute neighbor list between Li sites for local environment
        self._build_local_env_neighbors()

    def _build_local_env_neighbors(self):
        """
        Build neighbor list between Li sites for environmental dependence.
        This uses the same cutoff as the adjacency graph but is undirected and
        only used to count occupied neighbors.
        """
        self.env_neighbors = {i: [] for i in range(len(self.occupancy))}
        # Reuse structure and li_site_indices to compute neighbors between Li sites
        li_positions = [self.structure[site_info["struct_index"]].frac_coords for site_info in initial_sites]
        li_positions = np.array(li_positions)
        lattice = self.structure.lattice

        # Brute-force neighbor search among Li sites
        n_li = len(li_positions)
        for i in range(n_li):
            for j in range(i + 1, n_li):
                frac_diff = li_positions[j] - li_positions[i]
                # Wrap to nearest image
                frac_diff -= np.round(frac_diff)
                cart_disp = lattice.get_cartesian_coords(frac_diff)
                dist = np.linalg.norm(cart_disp)
                if dist <= self.nn_cutoff and dist > 1e-6:
                    self.env_neighbors[i].append(j)
                    self.env_neighbors[j].append(i)

    def _local_barrier(self, src, tgt):
        """
        Compute configuration-dependent activation energy for hop src -> tgt.

        We start from a base barrier E_a0 and add a contribution proportional to
        the number of occupied Li neighbors around the initial and final sites.
        This preserves the Arrhenius form of the rate while allowing rates to
        vary with local configuration, as discussed in kMC literature for ionic
        conductors (see e.g. Gavilán-Arriazu et al. 2021, Sec. on lattice models).
        """
        # Count occupied neighbors around src (excluding tgt) and tgt (excluding src)
        n_occ_src = 0
        for nb in self.env_neighbors.get(src, []):
            if nb == tgt:
                continue
            if self.occupancy[nb] == 1:
                n_occ_src += 1

        n_occ_tgt = 0
        for nb in self.env_neighbors.get(tgt, []):
            if nb == src:
                continue
            if self.occupancy[nb] == 1:
                n_occ_tgt += 1

        # Symmetric environment contribution
        n_eff = 0.5 * (n_occ_src + n_occ_tgt)

        # Linear dependence of barrier on local crowding
        E_a = self.E_a0 + self.delta_E_nn * n_eff
        return E_a

    def _event_rate(self, src, tgt):
        """
        Compute Arrhenius rate for hop src -> tgt with configuration-dependent barrier.
        """
        E_a = self._local_barrier(src, tgt)
        rate = self.nu * np.exp(-E_a / (self.kb * self.T))
        return rate

    def run_step(self):
        """
        Single kMC step with configuration-dependent event rates.

        The global time step Δt is sampled from an exponential distribution with
        rate equal to the sum of all *current* event rates, which depend on the
        local environment (occupancies) through E_a.
        """
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Enumerate all possible hops and compute their current rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    r = self._event_rate(src, tgt)
                    if r <= 0.0:
                        continue
                    total_rate += r
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # No possible events (deadlock)

        # BKL time advance with configuration-dependent total rate
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        u = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cumulative_rates, u)
        src, tgt, vec = events[idx]

        # Move particle
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id

        self.li_indices.discard(src)
        self.li_indices.add(tgt)

        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        # Mean Square Displacement (Å^2)
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )
        # Diffusivity (cm^2/s), MSD(t) = 6 D t for 3D
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein: σ = (n e^2 D) / (k_B T)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    'E_a': 0.30,
    'nu': 1e13,
    'volume': structure.volume,
    # Parameters controlling configuration dependence of activation barriers
    'delta_E_nn': 0.01,  # eV per occupied Li neighbor in local shell
    'nn_cutoff': cutoff
}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

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

        # Keep only the last 1000 values
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

            if rsd < 0.05:  # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
            )

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6.0 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D={D:.4e} cm^2/s")
print(f"Conductivity: {sigma:.4e} S/cm")

# Save result to JSON
result = {
    "is_success": True,
    "conductivity": float(sigma),
    "diffusivity": float(D),
    "msd": float(msd),
    "simulation_time_ns": float(sim.current_time * 1e9),
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns"
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")