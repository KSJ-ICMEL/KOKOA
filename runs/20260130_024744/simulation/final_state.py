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

# === 1a. Identify Li Sublattices (24d vs 48g/96h) ===
# We approximate tetrahedral vs octahedral Li sites by local Li-O coordination.
# Tetrahedral sites (24d) have 4 nearest O neighbors; octahedral (48g/96h) have 6.
# This uses only structural information from the CIF and does not invent new physics.

# Build neighbor list once for coordination analysis
o_indices = [i for i, s in enumerate(structure) if s.species.elements[0].symbol == "O"]
li_indices_all = [i for i, s in enumerate(structure) if s.species.elements[0].symbol == "Li"]

# Cutoff large enough to include first-shell O neighbors
neighbor_cutoff = 3.0  # Å, typical Li-O distances in garnets ~2 Å

# Precompute O neighbors for each Li
li_site_types = {}  # index -> "tet" or "oct"
for i in li_indices_all:
    site = structure[i]
    # Skip non-Li (safety, though we filtered li_indices_all)
    if site.species.elements[0].symbol != "Li":
        continue
    # Count nearby oxygens
    o_count = 0
    for j in o_indices:
        dist = structure.get_distance(i, j)
        if dist <= neighbor_cutoff:
            o_count += 1
    # Classify: 4-fold = tetrahedral (24d), 6-fold = octahedral-like (48g/96h)
    if o_count <= 4:
        li_site_types[i] = "tet"
    else:
        li_site_types[i] = "oct"

num_tet_sites = sum(1 for i in li_indices_all if li_site_types.get(i) == "tet")
num_oct_sites = sum(1 for i in li_indices_all if li_site_types.get(i) == "oct")
print(f"Identified Li sublattices: {num_tet_sites} tetrahedral-like, {num_oct_sites} octahedral-like Li sites.")

# === 1b. Equilibrium-like Li Occupancy Initialization ===
# Use experimentally informed average occupancies at room temperature for cubic garnets:
#   ~67.4% of Li on tetrahedral 24d sites
#   ~22% on octahedral 48g
#   ~21.8% on displaced octahedral 96h
# Since 48g/96h are both octahedral-like for transport topology, we target:
#   67.4% of all occupied Li on "tet" sites,
#   32.6% of all occupied Li on "oct" sites.
#
# The total Li content is taken from the CIF site occupancies (preserving chemistry of the input).
# This replaces the previous fully random initialization with a site-selective equilibrium-like distribution.

# Compute total Li count implied by CIF occupancies
total_Li_from_cif = 0.0
for i in li_indices_all:
    site = structure[i]
    occ = site.species.get("Li", 0.0)
    total_Li_from_cif += occ

# Round to integer number of Li ions
total_Li_target = int(round(total_Li_from_cif))
print(f"Total Li implied by CIF occupancies (rounded): {total_Li_target}")

# If there are no Li or invalid CIF, fall back safely
if total_Li_target <= 0 or len(li_indices_all) == 0:
    raise RuntimeError("No Li sites or invalid Li content in structure; cannot initialize kMC.")

# Target Li distribution across sublattices
# Fractions from internal paper for cubic Li garnets
f_tet = 0.674  # 67.4% of Li on 24d (tetrahedral)
f_oct = 1.0 - f_tet  # 32.6% on 48g/96h (octahedral/displaced)

num_Li_tet_target = int(round(f_tet * total_Li_target))
num_Li_oct_target = total_Li_target - num_Li_tet_target

# Clip targets so they do not exceed available sites
num_Li_tet_target = min(num_Li_tet_target, num_tet_sites)
num_Li_oct_target = min(num_Li_oct_target, num_oct_sites)

# If clipping reduces total below target, reassign remaining Li to any available sites
remaining_Li = total_Li_target - (num_Li_tet_target + num_Li_oct_target)
print(f"Target Li: tet={num_Li_tet_target}, oct={num_Li_oct_target}, remaining={remaining_Li}")

tet_indices = [i for i in li_indices_all if li_site_types.get(i) == "tet"]
oct_indices = [i for i in li_indices_all if li_site_types.get(i) == "oct"]

# Randomly choose which tetrahedral and octahedral sites are occupied
rng = np.random.default_rng()
occupied_tet = set(rng.choice(tet_indices, size=num_Li_tet_target, replace=False)) if num_Li_tet_target > 0 else set()
occupied_oct = set(rng.choice(oct_indices, size=num_Li_oct_target, replace=False)) if num_Li_oct_target > 0 else set()
occupied_sites = occupied_tet | occupied_oct

# Place any remaining Li at random on still-empty Li sites
if remaining_Li > 0:
    available_extra = list(set(li_indices_all) - occupied_sites)
    if remaining_Li > len(available_extra):
        remaining_Li = len(available_extra)
    extra_occupied = set(rng.choice(available_extra, size=remaining_Li, replace=False)) if remaining_Li > 0 else set()
    occupied_sites |= extra_occupied

print(f"Final Li occupation counts: {len(occupied_tet)} on tetrahedral, {len(occupied_oct)} on octahedral, "
      f"{len(occupied_sites)} total occupied Li sites.")

# Build initial_sites list consistent with new equilibrium-like distribution
initial_sites = []
for idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        state = 1 if idx in occupied_sites else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

print(f"Li sites initialized (equilibrium-like): {len(initial_sites)}")

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for i, site in enumerate(structure):
    if "Li" not in site.species.elements[0].symbol:
        continue
    neighbors = []
    for nb in neighbors_data[i]:
        if "Li" in structure[nb.index].species.elements[0].symbol:
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((nb.index, cart_disp))
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        # Map from "Li-site index in structure" to index in initial_sites/occupancy
        self.li_site_indices = [i for i, site in enumerate(structure) if "Li" in [s.symbol for s in site.species.elements]]
        self.structure_index_to_li_array_index = {struct_i: li_i for li_i, struct_i in enumerate(self.li_site_indices)}

        # Initialize particles at occupied Li sites
        for li_idx, struct_idx in enumerate(self.li_site_indices):
            if self.occupancy[li_idx] == 1:
                start = structure.lattice.get_cartesian_coords(initial_sites[li_idx]['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {'start': np.array(start), 'current': np.array(start)}
                p_id += 1
        
        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0
        
        kb = 8.617e-5  # eV/K
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

        # Build adjacency list in Li-site index space (not full-structure index)
        self.li_adj_list = {}
        for struct_i in self.li_site_indices:
            li_i = self.structure_index_to_li_array_index[struct_i]
            neighbors = []
            for nb_struct, disp in adj_list.get(struct_i, []):
                if nb_struct in self.structure_index_to_li_array_index:
                    li_j = self.structure_index_to_li_array_index[nb_struct]
                    neighbors.append((li_j, disp))
            self.li_adj_list[li_i] = neighbors

    def run_step(self):
        events, rates, total = [], [], 0.0
        for src in self.li_indices:
            for tgt, vec in self.li_adj_list.get(src, []):
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
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()]) # Mean Square Displacement (Å^2)
        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s), MSD(t)=6Dt
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # Nernst-Einstein Equation: σ = (n*e^2*D)/(k*T) (S/cm)
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
            sigma_history.pop(0) # Keep last 1000
            
        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0
            
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")
            
            if rsd < 0.05: # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

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