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
for idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state, "index": idx})
        li_site_indices.append(idx)

print(f"Li sites initialized: {len(initial_sites)}")

# Map from structure index to Li sublattice index
struct_to_li_index = {s["index"]: i for i, s in enumerate(initial_sites)}

# === 2. Build Adjacency Graph (Li-Li) ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for li_sub_idx, site_info in enumerate(initial_sites):
    struct_idx = site_info["index"]
    neighbors = []
    for nb in neighbors_data[struct_idx]:
        nb_struct_idx = nb.index
        nb_site = structure[nb_struct_idx]
        if "Li" in nb_site.species.elements[0].symbol:
            tgt_li_idx = struct_to_li_index[nb_struct_idx]
            frac_diff = nb_site.frac_coords - structure[struct_idx].frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((tgt_li_idx, cart_disp))
    adj_list[li_sub_idx] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 2b. Precompute Local Environment Metrics for Li Sites ===
# We use average Li-Li distance within 3.5 Å as a proxy for local volume/strain
env_cutoff = 3.5  # Angstrom, smaller than hop cutoff to characterize local crowding
all_neighbors_env = structure.get_all_neighbors(r=env_cutoff)

local_env_metric = np.zeros(len(initial_sites), dtype=float)

for li_sub_idx, site_info in enumerate(initial_sites):
    struct_idx = site_info["index"]
    nbs = all_neighbors_env[struct_idx]
    li_dists = []
    for nb in nbs:
        nb_struct_idx = nb.index
        if nb_struct_idx == struct_idx:
            continue
        nb_site = structure[nb_struct_idx]
        if "Li" in nb_site.species.elements[0].symbol:
            li_dists.append(nb.nn_distance)
    if li_dists:
        local_env_metric[li_sub_idx] = float(np.mean(li_dists))
    else:
        # Isolated Li: assign a large distance to indicate low crowding
        local_env_metric[li_sub_idx] = env_cutoff

print("Local environment metrics (average Li-Li distance) precomputed.")

# === 3. kMC Simulator (BKL Algorithm) with Environment-Dependent Barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, env_metric, params):
        self.params = params
        self.structure = structure
        self.adj_list = adj_list
        self.env_metric = env_metric

        # Occupancy on Li sublattice
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Particle tracking
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start_cart = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start_cart, dtype=float),
                    'current': np.array(start_cart, dtype=float)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # Constants
        self.kb = 8.617e-5  # eV/K

        # Global fallback parameters (used only if something goes wrong in Ea calc)
        self.global_Ea = params.get('E_a_global', 0.60)  # eV, calibrated lower than old heterogeneous upper bound
        self.nu = params['nu']

        # Range of geometry-dependent barriers, from DFT/NEB-informed LLZO-like values
        self.Ea_min = params.get('E_a_min', 0.45)  # eV
        self.Ea_max = params.get('E_a_max', 0.85)  # eV

        # Scaling factors for geometric contributions
        self.dist_scale = params.get('dist_scale', 0.4)
        self.asym_scale = params.get('asym_scale', 0.3)

        # Precompute typical Li-Li hop distance range from adjacency list
        hop_dists = []
        for src, nbs in adj_list.items():
            src_struct_idx = initial_sites[src]["index"]
            src_cart = structure[struct_idx_from_li := src_struct_idx].coords  # noqa: F841
            for tgt, vec in nbs:
                hop_dists.append(np.linalg.norm(vec))
        if hop_dists:
            self.d_min = float(np.percentile(hop_dists, 5))
            self.d_max = float(np.percentile(hop_dists, 95))
            if self.d_max <= self.d_min:
                self.d_max = self.d_min + 1e-3
        else:
            self.d_min, self.d_max = 1.0, 3.5

        # Normalize environment metric range
        if len(env_metric) > 0:
            self.env_min = float(np.min(env_metric))
            self.env_max = float(np.max(env_metric))
            if self.env_max <= self.env_min:
                self.env_max = self.env_min + 1e-3
        else:
            self.env_min, self.env_max = 1.5, 3.5

    def _compute_Ea(self, src, tgt, disp_vec):
        """
        Geometry- and environment-dependent activation barrier E_a (eV).

        - Depends on hop distance (proxy for local bottleneck width / strain).
        - Depends on asymmetry of local Li environments between src and tgt
          (proxy for local distortion / volume imbalance).
        """
        try:
            # Hop distance in Å
            d = float(np.linalg.norm(disp_vec))

            # Normalize distance to [0, 1] based on observed distribution
            x_d = (d - self.d_min) / (self.d_max - self.d_min)
            x_d = min(max(x_d, 0.0), 1.0)

            # Local environment metrics: larger distance -> lower crowding
            env_src = self.env_metric[src]
            env_tgt = self.env_metric[tgt]

            # Normalize env metrics
            e_src = (env_src - self.env_min) / (self.env_max - self.env_min)
            e_tgt = (env_tgt - self.env_min) / (self.env_max - self.env_min)
            e_src = min(max(e_src, 0.0), 1.0)
            e_tgt = min(max(e_tgt, 0.0), 1.0)

            # Asymmetry metric in [0, 1]
            s_raw = abs(e_src - e_tgt)
            s = min(max(s_raw, 0.0), 1.0)

            # Base barrier between Ea_min and Ea_max
            Ea = self.Ea_min + (self.Ea_max - self.Ea_min) * (
                self.dist_scale * x_d + self.asym_scale * s
            )

            # Ensure within [Ea_min, Ea_max]
            Ea = min(max(Ea, self.Ea_min), self.Ea_max)

            return Ea
        except Exception:
            # Fallback to a reasonable global Ea if something goes wrong
            return self.global_Ea

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        T = self.params['T']

        # Enumerate all possible hops and compute hop-specific rates
        for src in list(self.li_indices):
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    Ea_ij = self._compute_Ea(src, tgt, vec)
                    rate_ij = self.nu * np.exp(-Ea_ij / (self.kb * T))
                    if rate_ij <= 0.0:
                        continue
                    total_rate += rate_ij
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        rand = np.random.rand()
        self.current_time += -np.log(rand) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, vec = events[idx]

        # Execute event
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
        # Diffusivity (cm^2/s), MSD(t) = 6 D t
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein Equation: σ = (n * e^2 * D) / (k_B * T)
        e_charge = 1.602e-19  # C
        k_B_SI = 1.38e-23     # J/K
        sigma = (n * e_charge ** 2 * D) / (k_B_SI * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    # Global fallback Ea; heterogeneous barriers use Ea_min/max with geometry dependence
    'E_a_global': 0.60,
    'E_a_min': 0.45,
    'E_a_max': 0.85,
    'nu': 1e13,
    'volume': structure.volume,
    # Tuned scaling to avoid over-penalizing hops
    'dist_scale': 0.4,
    'asym_scale': 0.3,
}

sim = KMCSimulator(structure, adj_list, initial_sites, local_env_metric, sim_params)

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

        # Keep last 1000 entries
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = float(np.mean(sigma_history))
            std_sigma = float(np.std(sigma_history))
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0.0

            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")

            if rsd < 0.05:
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

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
    "simulation_time_ns": sim.current_time * 1e9,
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns"
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")