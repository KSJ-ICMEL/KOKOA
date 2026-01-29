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
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(idx)

print(f"Li sites initialized: {len(initial_sites)}")

# Map from structure index to Li-sublattice index and back
struct_to_li = {s_idx: li_idx for li_idx, s_idx in enumerate(li_site_indices)}
li_to_struct = {li_idx: s_idx for li_idx, s_idx in enumerate(li_site_indices)}

# === 2. Build Adjacency Graph with Geometry-Dependent Barriers ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Parameters for geometry-dependent activation energy
E_min = 0.60  # eV, lower bound for relatively unstrained hops (from NEB ~0.67 eV range)
E_max = 1.00  # eV, upper bound for strongly strained / narrow paths
d_ref = 3.0   # Å, reference Li-Li distance for low-strain hops
s_ref = 0.5   # Å, reference asymmetry scale

def compute_activation_barrier(distance, asymmetry):
    """
    Geometry-dependent activation barrier E_a(d, s).

    distance: cartesian Li-Li separation (Å)
    asymmetry: |d1 - d2|, where d1 and d2 are distances from both Li sites to a shared local environment proxy.
               Here we approximate asymmetry using local Li-Li neighbor imbalance.
    The barrier increases with both distance (narrower, more strained connections) and asymmetry
    to mimic lattice-relaxation effects making many paths less favorable.
    """
    # Distance term: penalize longer hops (proxy for stronger local distortion when relaxed)
    delta_d = max(distance - d_ref, 0.0)
    # Asymmetry term: penalize more asymmetric environments
    delta_s = max(asymmetry - s_ref, 0.0)

    # Scale contributions so that E_a spans [E_min, E_max] over typical geometry variations
    # Functional form is linear in geometry descriptors, consistent with using NEB-derived ranges.
    geom_factor = 1.0 + 0.7 * delta_d + 0.7 * delta_s
    E_a = E_min * geom_factor
    # Cap within [E_min, E_max]
    if E_a < E_min:
        E_a = E_min
    if E_a > E_max:
        E_a = E_max
    return E_a

# Precompute a simple local environment descriptor for each Li site:
# average distance to Li neighbors within a smaller cutoff, as a proxy for local volume/strain.
local_env_cutoff = 3.5  # Å
li_cart_coords = [structure[li_to_struct[i]].coords for i in range(len(li_site_indices))]
li_cart_coords = np.array(li_cart_coords)

local_env_metric = np.zeros(len(li_site_indices), dtype=float)
for i, coord in enumerate(li_cart_coords):
    # Distances to all other Li sites (periodic via Pymatgen can be more exact, but we keep current neighbors_data style)
    dists = np.linalg.norm(li_cart_coords - coord, axis=1)
    mask = (dists > 1e-3) & (dists < local_env_cutoff)
    if np.any(mask):
        local_env_metric[i] = np.mean(dists[mask])
    else:
        # Isolated or sparse environment; assign a larger effective distance (more open volume)
        local_env_metric[i] = local_env_cutoff

adj_list = {}
barrier_dict = {}

for li_idx, s_idx in enumerate(li_site_indices):
    site = structure[s_idx]
    neighbors = []
    for nb in neighbors_data[s_idx]:
        nb_s_idx = nb.index
        # Only Li-Li hops are considered
        if nb_s_idx not in struct_to_li:
            continue
        tgt_li_idx = struct_to_li[nb_s_idx]

        # Compute cartesian displacement using given image
        frac_diff = structure[nb_s_idx].frac_coords - site.frac_coords + nb.image
        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
        distance = np.linalg.norm(cart_disp)

        # Compute simple asymmetry metric using difference in local environment descriptor
        asymmetry = abs(local_env_metric[li_idx] - local_env_metric[tgt_li_idx])

        # Geometry-dependent activation energy for this hop
        E_a_ij = compute_activation_barrier(distance, asymmetry)

        neighbors.append((tgt_li_idx, cart_disp))
        barrier_dict[(li_idx, tgt_li_idx)] = E_a_ij

    adj_list[li_idx] = neighbors

print(f"Graph built (cutoff={cutoff}A) with geometry-dependent barriers")

# === 3. kMC Simulator (BKL Algorithm) with Heterogeneous Barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params, barrier_dict):
        self.params = params
        self.adj_list = adj_list
        self.barrier_dict = barrier_dict

        # Occupancy only on Li sublattice
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start, dtype=float),
                    'current': np.array(start, dtype=float),
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.T = params['T']

        # Precompute maximum possible rate factor to avoid repeated exponentials where possible
        self.rate_cache = {}

    def get_rate(self, src, tgt):
        key = (src, tgt)
        if key in self.rate_cache:
            return self.rate_cache[key]
        E_a = self.barrier_dict.get(key, self.params['E_a'])
        rate = self.nu * np.exp(-E_a / (self.kb * self.T))
        self.rate_cache[key] = rate
        return rate

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Build event list with heterogeneous rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    r = self.get_rate(src, tgt)
                    if r <= 0.0:
                        continue
                    total_rate += r
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        r_select = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r_select)
        src, tgt, vec = events[idx]

        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )  # Å^2
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
# Use a reference E_a (only as backup if geometry data missing, main barriers from barrier_dict)
sim_params = {'T': 300, 'E_a': 0.80, 'nu': 1e13, 'volume': structure.volume}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params, barrier_dict)

target_time = 1000e-9  # 1000ns timeout
log_interval = 100
sigma_history = []

while sim.current_time < target_time:
    if not sim.run_step():
        print("Deadlock - stopping")
        break
    if sim.step_count % log_interval == 0:
        msd, sigma = sim.calculate_properties()
        sigma_history.append(sigma)

        # Check convergence
        if len(sigma_history) > 1000:
            sigma_history.pop(0)  # Keep last 1000

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0

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
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")