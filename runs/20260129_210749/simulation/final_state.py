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
li_indices_global = []
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_indices_global.append(i)

print(f"Li sites initialized: {len(initial_sites)}")

# Map from global structure index -> compact Li index
global_to_li = {g_idx: li_idx for li_idx, g_idx in enumerate(li_indices_global)}
li_to_global = {li_idx: g_idx for li_idx, g_idx in enumerate(li_indices_global)}

# === 2. Build Adjacency Graph with Environment-Dependent Barrier ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Adjacency list in terms of Li-site indices (compact indexing)
adj_list = {}

for li_idx, g_idx in enumerate(li_indices_global):
    site = structure[g_idx]
    neighbors = []
    for nb in neighbors_data[g_idx]:
        nb_elem = structure[nb.index].species.elements[0].symbol
        if nb_elem == "Li":
            # Neighbor is Li site
            if nb.index in global_to_li:
                tgt_li_idx = global_to_li[nb.index]
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                neighbors.append((tgt_li_idx, cart_disp))
    adj_list[li_idx] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 2b. Precompute local framework environment for Li sites ===
# We characterize local La/Zr/O coordination for each Li site, used to modulate E_a.
framework_cutoff = 3.0  # Angstrom, local environment radius around Li

# Build neighbor list restricted to framework species around each Li site
framework_env = [{} for _ in range(len(initial_sites))]  # per Li site: dict of species -> count

for li_idx, g_idx in enumerate(li_indices_global):
    site = structure[g_idx]
    env_counts = {}
    for nb in neighbors_data[g_idx]:
        elem = structure[nb.index].species.elements[0].symbol
        if elem == "Li":
            continue  # exclude Li-Li from framework environment
        # within framework_cutoff?
        # neighbors_data is for cutoff=4.0; we must re-check distance
        if nb.nn_distance <= framework_cutoff:
            env_counts[elem] = env_counts.get(elem, 0) + 1
    framework_env[li_idx] = env_counts

# Compute reference (average) framework environment across all Li sites
avg_env = {}
if len(framework_env) > 0:
    all_species = set().union(*[env.keys() for env in framework_env])
    for sp in all_species:
        avg_env[sp] = np.mean([env.get(sp, 0) for env in framework_env])

# Function to compute environment-dependent activation energy for a hop
def compute_Ea_for_hop(src_li_idx, tgt_li_idx, base_Ea, alpha=0.02):
    """
    Environment-dependent activation energy.
    base_Ea: base activation energy (eV) from DFT-NEB assuming average environment.
    alpha: penalty per unit deviation in local framework coordination (eV per neighbour difference).
    The idea is to mimic increased barrier when local framework coordination deviates
    from the average (representing elastic/vibrational penalty).
    """
    src_env = framework_env[src_li_idx]
    tgt_env = framework_env[tgt_li_idx]

    # Measure deviation from average for source and target local frameworks
    def env_deviation(env):
        dev = 0.0
        for sp, avg_val in avg_env.items():
            dev += abs(env.get(sp, 0) - avg_val)
        return dev

    dev_src = env_deviation(src_env)
    dev_tgt = env_deviation(tgt_env)

    # Effective deviation for the hop (average of src and tgt)
    dev_eff = 0.5 * (dev_src + dev_tgt)

    # Environment-adjusted barrier
    Ea_eff = base_Ea + alpha * dev_eff
    return Ea_eff

# === 3. kMC Simulator (BKL Algorithm) ===
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
                self.particle_positions[p_id] = {'start': np.array(start), 'current': np.array(start)}
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        kb = 8.617e-5  # eV/K
        self.kbT = kb * params['T']
        self.nu = params['nu']
        self.base_Ea = params['E_a']  # base activation energy (eV)

    def run_step(self):
        events, rates, total = [], [], 0.0

        # Build list of possible hops with environment-dependent rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    # Compute environment-dependent activation energy
                    Ea_eff = compute_Ea_for_hop(src, tgt, self.base_Ea)
                    rate = self.nu * np.exp(-Ea_eff / self.kbT)
                    if rate <= 0:
                        continue
                    total += rate
                    events.append((src, tgt, vec))
                    rates.append(total)

        if total == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0, total)
        idx = np.searchsorted(rates, r)
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
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])  # Å^2
        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s)
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # S/cm
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

            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")

            if rsd < 0.05:  # 5% convergence criteria
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