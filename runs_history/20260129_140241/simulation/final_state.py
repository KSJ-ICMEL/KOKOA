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
        initial_sites.append({"coords": site.frac_coords, "state": state, "site_index": idx})
        li_site_indices.append(idx)

print(f"Li sites initialized: {len(initial_sites)}")

# Map from global structure index to Li-site index in initial_sites
global_to_li_index = {entry["site_index"]: i for i, entry in enumerate(initial_sites)}

# === 2. Build Adjacency Graph with Local, Direction-Dependent Barriers ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

# Base migration parameters
kb = 8.617e-5  # eV/K
base_Ea = 0.30  # eV (reference NEB barrier)

# Helper: compute local Li coordination (within Li sublattice) as simple strain proxy
def local_li_coord(index, neighbors_data, structure, r_cut=3.0):
    count = 0
    for nb in neighbors_data[index]:
        if "Li" in structure[nb.index].species.elements[0].symbol and nb.nn_distance <= r_cut:
            count += 1
    return count

# Precompute local Li coordination for all Li sites (using global indices)
li_local_coord = {}
for idx in li_site_indices:
    li_local_coord[idx] = local_li_coord(idx, neighbors_data, structure, r_cut=3.0)

# Compute mean coordination to center the strain-like term
mean_coord = np.mean(list(li_local_coord.values())) if li_local_coord else 0.0

# Directional scaling factors to mimic anisotropic migration landscape (xy easier than z)
direction_factors = {
    "xy": 1.0,   # reference
    "z": 1.2     # slightly higher barrier along z, consistent with less likely z-plane diffusion
}

for i_global in li_site_indices:
    neighbors = []
    site = structure[i_global]
    for nb in neighbors_data[i_global]:
        j_global = nb.index
        if "Li" in structure[j_global].species.elements[0].symbol:
            frac_diff = structure[j_global].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)

            # Direction classification
            dx, dy, dz = cart_disp
            if abs(dz) > abs(dx) and abs(dz) > abs(dy):
                dir_key = "z"
            else:
                dir_key = "xy"

            # Local strain/coordination modifier:
            # compressed (higher-than-average coordination) -> higher barrier
            # dilated (lower-than-average coordination) -> lower barrier
            coord_i = li_local_coord.get(i_global, mean_coord)
            coord_j = li_local_coord.get(j_global, mean_coord)
            avg_coord = 0.5 * (coord_i + coord_j)
            delta_coord = avg_coord - mean_coord

            # Choose a modest sensitivity so we do not invent unphysical extremes.
            # Simple linear modifier: Ea = base_Ea * dir_factor * (1 + alpha * delta_coord)
            alpha = 0.05  # 5% barrier change per coordination excess/deficit
            dir_factor = direction_factors[dir_key]
            Ea_ij = base_Ea * dir_factor * (1.0 + alpha * delta_coord)
            # Ensure barriers remain positive
            Ea_ij = max(Ea_ij, 0.01)

            # Store edge in Li-site index space
            src_li = global_to_li_index[i_global]
            tgt_li = global_to_li_index[j_global]
            neighbors.append((tgt_li, cart_disp, Ea_ij))
    adj_list[global_to_li_index[i_global]] = neighbors

print(f"Graph built with local, direction-dependent barriers (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with Site-/Direction-Dependent Barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list

        # Occupancy only over Li sublattice
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

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

        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.T = params['T']

    def hop_rate(self, Ea):
        # Harmonic TST: k = nu * exp(-Ea / (kB T))
        return self.nu * np.exp(-Ea / (self.kb * self.T))

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Build event list with local barriers
        for src in self.li_indices:
            for tgt, vec, Ea in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self.hop_rate(Ea)
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock: no allowed hops

        # BKL time advance
        rnd = np.random.rand()
        self.current_time += -np.log(rnd) / total_rate
        self.step_count += 1

        # Select and execute event
        r_select = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r_select)
        src, tgt, vec = events[idx]

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
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s (MSD(t)=6Dt)
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {'T': 300, 'E_a': base_Ea, 'nu': 1e13, 'volume': structure.volume}
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

        # Keep last 1000 entries
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