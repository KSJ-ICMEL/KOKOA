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
        initial_sites.append({"coords": site.frac_coords, "state": state, "struct_index": idx})
        li_site_indices.append(idx)

print(f"Li sites initialized: {len(initial_sites)}")

# === 2. Build Adjacency Graph with Direction-Dependent Barriers ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Map from structure index to Li-site index for adjacency restricted to Li sublattice
struct_to_li = {s['struct_index']: i for i, s in enumerate(initial_sites)}
li_to_struct = {i: s['struct_index'] for i, s in enumerate(initial_sites)}

# Direction-dependent activation energies (eV) to mimic flexible lattice NEB results.
# Following context that diffusion along z is less favorable than in x-y plane [57,58]
# and that barriers depend on crystallographic direction [59,60].
E_a_dir = {
    'xy': 0.26,  # concerted migration barriers ~0.26 eV in LLZO-like systems
    'z_pos': 0.32,  # slightly higher along +z
    'z_neg': 0.32   # slightly higher along -z (symmetric)
}

# Geometry-dependent prefactors (Hz) can reflect coupling to soft modes; keep them
# modestly varying with direction, consistent with transition-state theory form
# k = nu * exp(-E_a / (k_B T)).
nu_dir = {
    'xy': 1.0e13,
    'z_pos': 0.7e13,
    'z_neg': 0.7e13
}

kb = 8.617e-5  # eV/K

def classify_direction(cart_disp):
    """
    Classify hop direction into xy vs +/- z based on displacement vector.
    """
    dx, dy, dz = cart_disp
    # If z-component is small relative to in-plane, treat as xy
    if abs(dz) < 0.3 * np.sqrt(dx*dx + dy*dy + 1e-12):
        return 'xy'
    # Otherwise classify by sign of dz
    return 'z_pos' if dz >= 0.0 else 'z_neg'

# Build adjacency: for each Li site (index in initial_sites), store neighbors as:
# li_adj_list[li_idx] = list of (li_neighbor_idx, cartesian_displacement, direction_key)
li_adj_list = {}

for li_idx, s in enumerate(initial_sites):
    struct_idx = s['struct_index']
    neighbors = []
    for nb in neighbors_data[struct_idx]:
        nb_struct_idx = nb.index
        if nb_struct_idx not in struct_to_li:
            continue
        tgt_li_idx = struct_to_li[nb_struct_idx]
        # fractional displacement including image
        frac_diff = structure[nb_struct_idx].frac_coords - structure[struct_idx].frac_coords + nb.image
        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
        dir_key = classify_direction(cart_disp)
        neighbors.append((tgt_li_idx, cart_disp, dir_key))
    li_adj_list[li_idx] = neighbors

print(f"Graph with direction-dependent barriers built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with configuration-/direction-dependent rates ===
class KMCSimulator:
    def __init__(self, structure, li_adj_list, initial_sites, params, E_a_dir, nu_dir):
        self.params = params
        self.adj_list = li_adj_list
        # occupancy is defined over Li sites only (indexing initial_sites)
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Particle tracking over Li sites
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
        self.E_a_dir = E_a_dir
        self.nu_dir = nu_dir

        # Precompute base attempt frequencies for current temperature
        self.precomputed_rates = {}
        for dkey in self.E_a_dir:
            self.precomputed_rates[dkey] = self.nu_dir[dkey] * np.exp(
                -self.E_a_dir[dkey] / (self.kb * self.params['T'])
            )

    def local_barrier_modifier(self, src, tgt):
        """
        Simple configuration-dependent modifier to mimic lattice relaxation effects:
        if the target site is in a more crowded local environment (more occupied
        Li neighbors) relative to the source, increase the barrier slightly; if
        it is less crowded, decrease slightly. This encodes that lattice breathing/
        tilting and local strain around Li hops modifies activation energies.

        We keep the modifier small (±0.02 eV) to remain consistent with
        NEB-based distributions around the average barrier.
        """
        # Count occupied Li neighbors around src and tgt
        def count_occ_neighbors(site_idx):
            count = 0
            for nb_idx, _, _ in self.adj_list.get(site_idx, []):
                if self.occupancy[nb_idx] == 1:
                    count += 1
            return count

        n_src = count_occ_neighbors(src)
        n_tgt = count_occ_neighbors(tgt)

        # If target more crowded, add small positive delta; if less, subtract
        delta_n = n_tgt - n_src
        # Scale factor: 0.01 eV per excess neighbor, capped at ±0.02 eV
        delta_E = 0.01 * delta_n
        delta_E = max(min(delta_E, 0.02), -0.02)
        return delta_E

    def compute_rate(self, dir_key, src, tgt):
        """
        Compute hop rate using direction-dependent NEB-like barrier and an
        environment modifier: k = nu * exp(-(E_a_dir + ΔE_env)/(k_B T)).
        """
        base_rate = self.precomputed_rates[dir_key]
        delta_E = self.local_barrier_modifier(src, tgt)
        # Apply modifier multiplicatively: exp(-ΔE / kT)
        rate = base_rate * np.exp(-delta_E / (self.kb * self.params['T']))
        return rate

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Build event list with configuration- and direction-dependent rates
        for src in list(self.li_indices):
            for tgt, vec, dir_key in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    k_ij = self.compute_rate(dir_key, src, tgt)
                    if k_ij <= 0.0:
                        continue
                    total_rate += k_ij
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock: no allowed events

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
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
        if self.current_time == 0.0 or self.num_particles == 0:
            return 0.0, 0.0
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
# Use representative barriers consistent with NEB/AIMD and experiment for LLZO-like garnets:
# concerted migration ~0.26 eV with slightly higher barriers along z [146,67,106,173].
sim_params = {
    'T': 300,
    'volume': structure.volume
}

sim = KMCSimulator(structure, li_adj_list, initial_sites, sim_params, E_a_dir, nu_dir)

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

            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")

            if rsd < 0.05:  # 5% convergence
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

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