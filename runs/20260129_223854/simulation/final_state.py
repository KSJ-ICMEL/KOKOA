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
        initial_sites.append({"coords": site.frac_coords, "state": state, "struct_index": i})
        li_site_indices.append(i)

print(f"Li sites initialized: {len(initial_sites)}")

# Map from structure index to Li-site index in initial_sites
struct_index_to_li_index = {entry["struct_index"]: li_i for li_i, entry in enumerate(initial_sites)}

# === 2. Build Adjacency Graph with Path-Dependent Barriers ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Use Li–Li separation-based activation energies from Table 2
# A: 2.45 Å → 0.44 eV
# B: 2.52 Å → 0.35 eV
# C: 2.58 Å → 0.26 eV
# D: 2.59 Å → 0.45 eV
li_paths = [
    (2.45, 0.44),
    (2.52, 0.35),
    (2.58, 0.26),
    (2.59, 0.45),
]
li_paths = sorted(li_paths, key=lambda x: x[0])  # sort by separation

def assign_li_barrier(distance):
    """Assign activation energy based on nearest Li–Li separation from Table 2."""
    closest_sep, closest_Ea = min(li_paths, key=lambda x: abs(x[0] - distance))
    return closest_Ea

# Build adjacency list over Li sites only, keyed by Li-site index
# Each edge: (neighbor_li_index, displacement_vector_cart, Ea, nu)
adj_list = {li_i: [] for li_i in range(len(initial_sites))}

# Distinct attempt frequencies for different path types (still uniform here, but
# separated for future refinement). We keep them equal to avoid inventing data.
# All nu set to 1e13 s^-1 per the original code and typical phonon frequencies.
PATH_NU = 1e13  # Hz

for li_i, site_entry in enumerate(initial_sites):
    struct_i = site_entry["struct_index"]
    site = structure[struct_i]
    neighbors = []
    for nb in neighbors_data[struct_i]:
        nb_struct_index = nb.index
        nb_site = structure[nb_struct_index]
        if "Li" not in nb_site.species.elements[0].symbol:
            continue
        # Only consider Li neighbors that are part of our Li-site list
        if nb_struct_index not in struct_index_to_li_index:
            continue
        li_j = struct_index_to_li_index[nb_struct_index]

        # Compute Li–Li separation in Å including periodic image
        frac_diff = nb_site.frac_coords - site.frac_coords + nb.image
        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
        dist = np.linalg.norm(cart_disp)

        # Assign environment-dependent activation energy using Table 2
        Ea_ij = assign_li_barrier(dist)
        neighbors.append((li_j, cart_disp, Ea_ij, PATH_NU))
    adj_list[li_i] = neighbors

print(f"Graph with environment-dependent barriers built (cutoff={cutoff}Å)")

# === 3. kMC Simulator (BKL Algorithm) with Path-Specific Rates ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list

        # Occupancy defined on Li sites only
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map Li site index -> particle id and track trajectories
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_i, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_i] = p_id
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

    def _rate(self, nu, Ea):
        """Arrhenius rate for a hop with given attempt frequency and barrier."""
        T = self.params['T']
        return nu * np.exp(-Ea / (self.kb * T))

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Build event list with path-dependent rates
        for src in self.li_indices:
            if self.occupancy[src] == 0:
                continue
            for tgt, vec, Ea, nu in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._rate(nu, Ea)
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock: no allowed hops

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, vec = events[idx]

        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0.0 or self.num_particles == 0:
            return 0.0, 0.0
        # Mean square displacement in Å^2
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )
        # Diffusivity D via MSD(t) = 6 D t in 3D
        D = msd / (6.0 * self.current_time) * 1e-16  # Å^2 → cm^2
        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Å^3 → cm^3
        # Nernst-Einstein conductivity
        e = 1.602e-19  # C
        kB_SI = 1.38e-23  # J/K
        sigma = (n * e * e * D) / (kB_SI * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
# We no longer use a single global barrier; Ea is path-dependent.
sim_params = {'T': 300, 'volume': structure.volume}
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

        # Keep only last 1000 entries
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0.0

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}Å^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}Å^2, sigma={sigma*1e3:.4f}mS/cm"
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