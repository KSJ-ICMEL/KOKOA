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

# === 2. Build Adjacency Graph with Bottleneck-Dependent Barriers ===
#
# The previous implementation used a single geometric cutoff (4 Å) and
# a uniform activation energy (0.30 eV) for all Li–Li hops. This ignored
# the variation of migration barriers associated with different
# bottleneck geometries. Here we keep the geometric cutoff to define
# potential connectivity, but we assign a hop-specific activation
# energy based on a distance–barrier relationship extracted from the
# provided LLZO data (Table 2 in the cited paper).
#
# We only use *Li–Li separation vs. activation energy* from Table 2:
#
#   Path   d_Li–Li (Å)   E_a (eV)
#   A      2.45          0.44
#   B      2.52          0.35
#   C      2.58          0.26
#   D      2.59          0.45
#
# To stay evidence-based, we construct an empirical mapping from
# Li–Li distance to E_a by piecewise-linear interpolation between these
# data points. Hops outside the calibrated distance window are treated
# as high-barrier (effectively blocked) to avoid artificially easy
# conduction through geometrically unfavorable bottlenecks.
#
# IMPORTANT: We do not invent any new functional forms beyond linear
# interpolation between the tabulated points, and we do not use the
# O–O data (Table 3) in this Li-ion kMC.

from pymatgen.analysis.local_env import MinimumDistanceNN

# Tabulated Li–Li distances (Å) and activation energies (eV) from Table 2
li_li_distances = np.array([2.45, 2.52, 2.58, 2.59], dtype=float)
li_li_barriers = np.array([0.44, 0.35, 0.26, 0.45], dtype=float)

# Sort them to ensure monotonic x for interpolation
sort_idx = np.argsort(li_li_distances)
li_li_distances = li_li_distances[sort_idx]
li_li_barriers = li_li_barriers[sort_idx]

# Define a "blocked" barrier energy (eV) for hops whose Li–Li distance
# lies outside the calibrated range. Using a value >> 1 eV suppresses
# their rate strongly relative to the 0.26–0.45 eV paths.
BLOCKED_EA = 2.0  # eV, chosen to effectively remove such paths

def distance_to_barrier(d):
    """
    Map Li–Li separation (Å) to activation energy (eV) using the
    tabulated data and linear interpolation. Distances outside the
    [min, max] window are treated as high-barrier (blocked).
    """
    d_min, d_max = li_li_distances[0], li_li_distances[-1]
    if d < d_min or d > d_max:
        return BLOCKED_EA
    return float(np.interp(d, li_li_distances, li_li_barriers))

# Use a neighbor finder that gives minimum-image distances
cutoff = 4.0  # Angstrom (geometric upper bound; energetic filtering is via E_a)
mdnn = MinimumDistanceNN(cutoff=cutoff, get_all_sites=True)
neighbors_data = mdnn.get_all_nn_info(structure)

# Build adjacency list with hop-specific activation energies
# adj_list[src] = list of (tgt_index, displacement_vector_cart, Ea_eV)
adj_list = {}

for i, site in enumerate(structure):
    if "Li" not in [s.symbol for s in site.species.elements]:
        continue

    neighbors = []
    for nb in neighbors_data[i]:
        j = nb['site_index']
        if "Li" not in [s.symbol for s in structure[j].species.elements]:
            continue

        # Cartesian displacement vector from site i to site j, accounting for periodicity
        frac_diff = structure[j].frac_coords - site.frac_coords + nb['image']
        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
        d_ij = np.linalg.norm(cart_disp)

        # Assign activation energy based on Li–Li distance
        Ea_ij = distance_to_barrier(d_ij)

        # Skip hops that are effectively blocked
        if Ea_ij >= BLOCKED_EA:
            continue

        neighbors.append((j, cart_disp, Ea_ij))
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}Å) with bottleneck-dependent barriers")

# === 3. kMC Simulator (BKL Algorithm) with Hop-Specific Barriers ===
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

    def hop_rate(self, Ea):
        """
        Arrhenius rate for a hop with activation energy Ea (eV):
            k = nu * exp(-Ea / (k_B T))
        """
        return self.nu * np.exp(-Ea / (self.kb * self.T))

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Enumerate all possible hops from occupied Li sites to vacant Li sites
        for src in self.li_indices:
            if self.occupancy[src] != 1:
                continue
            for tgt, vec, Ea in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self.hop_rate(Ea)
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # No available events (deadlock)

        # BKL time advance
        rnd = np.random.rand()
        self.current_time += -np.log(rnd) / total_rate
        self.step_count += 1

        # Select event
        rnd_event = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cumulative_rates, rnd_event)
        src, tgt, vec = events[idx]

        # Execute event
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
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
# The global E_a parameter is no longer used as a uniform barrier; it is
# retained here only for record-keeping. The actual hop rates are based
# on the hop-specific barriers determined above.
sim_params = {'T': 300, 'E_a': 0.30, 'nu': 1e13, 'volume': structure.volume}
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

        # Keep last 1000 samples
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
                print(
                    f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns"
                )
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}Å^2, sigma={sigma*1e3:.4f}mS/cm"
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
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")