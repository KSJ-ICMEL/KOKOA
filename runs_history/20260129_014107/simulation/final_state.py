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

# === 1a. Identify Li sites in the supercell ===
li_site_indices = []
li_frac_coords = []
li_occ_probs = []

for i, site in enumerate(structure):
    # Identify Li by element symbol
    if any(sp.symbol == "Li" for sp in site.species.elements):
        li_site_indices.append(i)
        li_frac_coords.append(site.frac_coords)
        li_occ_probs.append(site.species.get("Li", 0.0))

num_li_sites = len(li_site_indices)
li_frac_coords = np.array(li_frac_coords)
li_occ_probs = np.array(li_occ_probs)

print(f"Total Li sites in supercell: {num_li_sites}")

# === 1b. Build Li–Li adjacency on Li sublattice ===
cutoff = 4.0  # Angstrom for Li–Li connectivity
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Map: li_local_index -> list of (neighbor_li_local_index, displacement_cart)
li_adj_list = {i: [] for i in range(num_li_sites)}

# Create a map from global structure index -> local Li index
global_to_li = {g: li for li, g in enumerate(li_site_indices)}

for li_local, g_index in enumerate(li_site_indices):
    site = structure[g_index]
    for nb in neighbors_data[g_index]:
        nb_g = nb.index
        if nb_g in global_to_li:
            nb_li_local = global_to_li[nb_g]
            # displacement including periodic image
            frac_diff = structure[nb_g].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            li_adj_list[li_local].append((nb_li_local, cart_disp))

print(f"Li–Li graph built on Li sublattice (cutoff={cutoff} Å)")

# === 1c. Initialize Li/vacancy configuration with short-range order (SRO) ===
#
# Goal: avoid completely random, uncorrelated Li distribution.
# Strategy: simple Metropolis Monte Carlo at 300 K on a pair-interaction model
# that penalizes Li–Li nearest-neighbor occupancy. This generates a
# Li–vacancy configuration with reduced local crowding (short-range order)
# without inventing new physics beyond Li–Li repulsion.

kb_eV = 8.617e-5  # eV/K
T_init = 300.0
beta_init = 1.0 / (kb_eV * T_init)

# Effective pair penalty for each occupied Li–Li neighbor pair (heuristic scale)
J_pair = 0.05  # eV; small repulsion to reduce nearest-neighbor Li clustering

# Initial random occupation according to nominal probabilities
occupancy = np.zeros(num_li_sites, dtype=int)
for i in range(num_li_sites):
    if np.random.rand() < li_occ_probs[i]:
        occupancy[i] = 1

def local_pair_energy(site_idx, occ_array):
    """
    Compute pair-interaction contribution for a given site index:
    E_i = J_pair * n_occupied_neighbors(i)
    Only pairs with both sites occupied contribute.
    """
    if occ_array[site_idx] == 0:
        return 0.0
    e = 0.0
    for nb_idx, _ in li_adj_list[site_idx]:
        if occ_array[nb_idx] == 1:
            e += J_pair
    return e

def total_pair_energy(occ_array):
    """
    Total pair energy: sum over each pair once.
    """
    e_total = 0.0
    counted = set()
    for i in range(num_li_sites):
        if occ_array[i] == 1:
            for j, _ in li_adj_list[i]:
                if occ_array[j] == 1:
                    pair = tuple(sorted((i, j)))
                    if pair not in counted:
                        counted.add(pair)
                        e_total += J_pair
    return e_total

# Metropolis Monte Carlo to equilibrate at 300 K with fixed number of Li
num_li = int(np.sum(occupancy))

print(f"Initial Li count (from CIF probabilities): {num_li}")

# Construct a list of site indices and perform canonical swaps Li <-> vacancy
mc_steps = 200000  # number of attempted swaps
site_indices_all = np.arange(num_li_sites, dtype=int)

# Precompute neighbor lists as pure indices for faster MC
li_neighbors_indices = {i: [nb for nb, _ in li_adj_list[i]] for i in range(num_li_sites)}

def site_local_energy(i, occ):
    """Energy contribution of site i with its neighbors (each pair counted once)."""
    if occ[i] == 0:
        return 0.0
    e = 0.0
    for j in li_neighbors_indices[i]:
        if occ[j] == 1:
            e += J_pair
    return e

def delta_energy_swap(i, j, occ):
    """
    Compute energy change ΔE when swapping occupations of i and j.
    i and j have opposite occupation (1 and 0).
    We consider only local neighborhoods around i and j.
    """
    # Sites that can change pair contributions are i, j and their neighbors
    affected_sites = set([i, j])
    affected_sites.update(li_neighbors_indices[i])
    affected_sites.update(li_neighbors_indices[j])

    # Compute current local energy for affected sites
    e_before = 0.0
    for s in affected_sites:
        if occ[s] == 1:
            for nb in li_neighbors_indices[s]:
                if nb in affected_sites and occ[nb] == 1 and s < nb:
                    e_before += J_pair

    # Propose swap: flip occ[i], occ[j]
    occ_i, occ_j = occ[i], occ[j]
    occ[i], occ[j] = occ_j, occ_i

    e_after = 0.0
    for s in affected_sites:
        if occ[s] == 1:
            for nb in li_neighbors_indices[s]:
                if nb in affected_sites and occ[nb] == 1 and s < nb:
                    e_after += J_pair

    # Revert swap
    occ[i], occ[j] = occ_i, occ_j

    return e_after - e_before

# Perform MC sweeps of random Li-vacancy swaps
if num_li > 0 and num_li < num_li_sites:
    for step in range(mc_steps):
        # pick a random occupied site and a random vacant site
        occ_sites = np.where(occupancy == 1)[0]
        vac_sites = np.where(occupancy == 0)[0]
        if len(occ_sites) == 0 or len(vac_sites) == 0:
            break
        i = np.random.choice(occ_sites)
        j = np.random.choice(vac_sites)
        dE = delta_energy_swap(i, j, occupancy)
        if dE <= 0.0 or np.random.rand() < np.exp(-beta_init * dE):
            # accept swap
            occupancy[i], occupancy[j] = occupancy[j], occupancy[i]
        if (step + 1) % (mc_steps // 10) == 0:
            print(f"MC equilibration step {step+1}/{mc_steps}")
else:
    print("Trivial occupation (all full or all empty); skipping MC equilibration.")

final_energy = total_pair_energy(occupancy)
print(f"Final pair energy after MC SRO equilibration: {final_energy:.4f} eV")

# Build initial_sites consistent with original code structure
initial_sites = []
for li_local, g_index in enumerate(li_site_indices):
    state = int(occupancy[li_local])
    initial_sites.append({
        "coords": li_frac_coords[li_local],
        "state": state
    })

print(f"Li sites initialized with SRO: {len(initial_sites)} (occupied: {np.sum(occupancy)})")

# === 2. (Legacy) Build Adjacency Graph on full structure (kept for compatibility, unused in kMC) ===
cutoff_full = 4.0  # Angstrom
neighbors_data_full = structure.get_all_neighbors(r=cutoff_full)
adj_list = {}

for i, site in enumerate(structure):
    if "Li" not in site.species.elements[0].symbol:
        continue
    neighbors = []
    for nb in neighbors_data_full[i]:
        if "Li" in structure[nb.index].species.elements[0].symbol:
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((nb.index, cart_disp))
    adj_list[i] = neighbors

print(f"Graph built on full structure (cutoff={cutoff_full} Å)")

# === 3. kMC Simulator (BKL Algorithm) on Li sublattice ===
class KMCSimulator:
    def __init__(self, structure, li_adj_list, initial_sites, params):
        self.params = params
        self.li_adj_list = li_adj_list

        # Occupancy array only over Li sublattice
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map li_site_index -> particle_id
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start),
                    'current': np.array(start)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        kb = 8.617e-5  # eV/K
        self.kb = kb

        # Use the same base barrier parameter as before; per-step rate is still uniform.
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

    def run_step(self):
        events = []
        rates_cum = []
        total_rate = 0.0

        # Enumerate all possible Li hops (occupied -> vacant) on Li sublattice
        for src in list(self.li_indices):
            for tgt, vec in self.li_adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    total_rate += self.base_rate
                    events.append((src, tgt, vec))
                    rates_cum.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock: no allowed hops

        # Advance time (BKL)
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(rates_cum, r)
        src, tgt, vec = events[idx]

        # Execute hop
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
        # Mean square displacement in Å^2
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])
        # Diffusivity in cm^2/s (1 Å^2 = 1e-16 cm^2)
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein conductivity (S/cm)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    'E_a': 0.30,
    'nu': 1e13,
    'volume': structure.volume
}
sim = KMCSimulator(structure, li_adj_list, initial_sites, sim_params)

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

        # Keep last 1000 sigma values
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0.0

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f} ns, "
                f"MSD={msd:.2f} Å^2, sigma={sigma*1e3:.4f} mS/cm, "
                f"RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criterion
                print(
                    f"Convergence reached (RSD < 5%) at "
                    f"{sim.current_time*1e9:.2f} ns"
                )
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f} ns, "
                f"MSD={msd:.2f} Å^2, sigma={sigma*1e3:.4f} mS/cm"
            )

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6.0 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

print(f"\n=== Simulation Complete ===")
print(f"T = {sim_params['T']} K, Time = {sim.current_time*1e9:.2f} ns")
print(f"D = {D:.4e} cm^2/s")
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
    "execution_log": (
        f"Completed {sim.step_count} steps in "
        f"{sim.current_time*1e9:.2f} ns with SRO-initialized Li configuration"
    ),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")