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

# === 1a. Identify Li sites and build per-species site lists ===
li_site_indices = []
li_site_occupancies = []
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        li_site_indices.append(i)
        li_site_occupancies.append(site.species.get("Li", 0.0))

num_li_sites = len(li_site_indices)
total_nominal_li = np.sum(li_site_occupancies)
print(f"Identified {num_li_sites} Li sites with total nominal Li {total_nominal_li:.3f}")

# === 1b. Construct a correlated, partially ordered Li configuration ===
# We enforce the nominal Li count from the CIF, but avoid purely random Bernoulli filling.
# Strategy (evidence-based, but heuristic within allowed context):
# - Determine target Li count by rounding total nominal occupancy.
# - Rank Li sites by their occupancy in the CIF, which encodes experimental site preference.
# - Preferentially occupy higher-occupancy sites.
# - For sites with similar occupancy, we reduce local clustering by penalizing placing Li on
#   nearest-neighbor Li sites that are already occupied.

# Build neighbor list restricted to Li sites for correlation-aware initialization
cutoff_init = 4.0  # Angstrom, same as used for hopping graph
neighbors_data_all = structure.get_all_neighbors(r=cutoff_init)
li_adj_init = {idx: [] for idx in li_site_indices}
index_map = {struct_idx: k for k, struct_idx in enumerate(li_site_indices)}

for struct_idx in li_site_indices:
    li_neighbors = []
    for nb in neighbors_data_all[struct_idx]:
        if "Li" in structure[nb.index].species.elements[0].symbol:
            li_neighbors.append(nb.index)
    li_adj_init[struct_idx] = li_neighbors

# Determine target number of Li ions
target_li = int(round(total_nominal_li))
target_li = max(1, min(target_li, num_li_sites))
print(f"Target Li count for initialization: {target_li}")

# Prepare list of Li sites with their occupancies
li_sites_info = [
    (struct_idx, occ, structure[struct_idx].frac_coords)
    for struct_idx, occ in zip(li_site_indices, li_site_occupancies)
]

# Sort primarily by descending occupancy (site preference), secondarily by a random key
rng = np.random.default_rng()
random_keys = rng.random(len(li_sites_info))
li_sites_info_sorted = sorted(
    zip(li_sites_info, random_keys),
    key=lambda x: (-x[0][1], x[1])
)
li_sites_info_sorted = [x[0] for x in li_sites_info_sorted]

# Greedy placement with simple local "repulsion" to avoid over-clustering:
# For each candidate site in sorted order, we compute a score that penalizes
# having too many occupied Li neighbors. We only accept sites that do not
# exceed a maximum allowed occupied neighbors threshold, which we vary
# to reach the target Li count while maintaining local dilution.
occupied_set = set()
max_occupied_neighbors = 1  # initial strict threshold
attempts = 0
max_attempts = 5

while len(occupied_set) < target_li and attempts < max_attempts:
    occupied_set.clear()
    for struct_idx, occ, _coords in li_sites_info_sorted:
        if len(occupied_set) >= target_li:
            break
        # Count how many Li neighbors are already occupied
        n_occupied_neighbors = sum(
            (nb in occupied_set) for nb in li_adj_init[struct_idx]
        )
        if n_occupied_neighbors <= max_occupied_neighbors:
            occupied_set.add(struct_idx)
    if len(occupied_set) < target_li:
        # Relax the local dilution constraint and try again
        max_occupied_neighbors += 1
        attempts += 1
    else:
        break

# If still not enough Li after relaxing constraints, fill remaining spots by highest occupancy
if len(occupied_set) < target_li:
    print("Warning: Correlation constraint limited Li placement; filling remaining with highest-occupancy sites.")
    for struct_idx, occ, _coords in li_sites_info_sorted:
        if len(occupied_set) >= target_li:
            break
        occupied_set.add(struct_idx)

print(f"Initialized correlated Li configuration with {len(occupied_set)} occupied sites.")

# Build initial_sites list in structure order (as expected by the rest of the code)
initial_sites = []
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        state = 1 if i in occupied_set else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

print(f"Li sites initialized (correlated): {len(initial_sites)}")

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

# We only care about Li sites that exist in initial_sites. Build a mapping from
# "Li site index in initial_sites" to "structure index", and vice versa.
li_structure_indices = [idx for idx in li_site_indices]
li_index_map = {struct_idx: k for k, struct_idx in enumerate(li_structure_indices)}

for struct_idx in li_structure_indices:
    site = structure[struct_idx]
    neighbors = []
    for nb in neighbors_data[struct_idx]:
        if "Li" in structure[nb.index].species.elements[0].symbol:
            if nb.index not in li_index_map:
                continue
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((li_index_map[nb.index], cart_disp))
    adj_list[li_index_map[struct_idx]] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list

        # occupancy, site_to_particle, etc. are defined on the Li-site index space
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
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        kb = 8.617e-5  # eV/K
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

    def run_step(self):
        events, rates, total = [], [], 0.0
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    total += self.base_rate
                    events.append((src, tgt, vec))
                    rates.append(total)

        if total == 0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1

        # Select and execute event
        idx = np.searchsorted(rates, np.random.uniform(0, total))
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
            return 0, 0
        msd = np.mean(
            [
                np.sum((p['current'] - p['start']) ** 2)
                for p in self.particle_positions.values()
            ]
        )  # Mean Square Displacement (Å^2)
        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s), MSD(t)=6Dt
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # σ = (n*e^2*D)/(k*T) (S/cm)
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {'T': 300, 'E_a': 0.30, 'nu': 1e13, 'volume': structure.volume}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

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
                print(
                    f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns"
                )
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
            )

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0

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
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")