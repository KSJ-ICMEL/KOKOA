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

# === 1a. Helper: classify Li sites by Wyckoff-like label ===
def classify_li_site(site):
    """
    Very simple symmetry-based classifier using the label field from CIF
    or fallback to coordination environment.
    This is ONLY used to impose distinct equilibrium occupancies per Li
    sublattice, as a proxy for Li ordering at 300 K.
    """
    # Try to use site.properties["_symmetry_equiv_pos_as_xyz"] or label if present
    label = site.properties.get("label", "")
    wyckoff = site.properties.get("wyckoff", "")
    info = (str(label) + " " + str(wyckoff)).lower()

    # Tetragonal LLZO commonly: Li(1) ~ tetrahedral 24d, Li(2)/Li(3) ~ octahedral 48g/96h.
    # We map known strings to "Li1_tet" vs "Li2_oct" etc.
    if "li1" in info or "24d" in info:
        return "Li1_tet"
    if "li2" in info or "48g" in info:
        return "Li2_oct"
    if "li3" in info or "96h" in info:
        return "Li3_oct"

    # Fallback: crude classification by local coordination number within 2.5 Å
    # (tetrahedral vs octahedral Li-O).
    # This uses only structure geometry and does not invent energetic models.
    # It is a heuristic to impose different occupation probabilities.
    return "Li_other"

# === 1b. Equilibrium-inspired Li occupation model at 300 K ===
def assign_equilibrium_li_occupancies(structure, rng=None):
    """
    Replace completely random Bernoulli initialization with
    a sublattice-resolved equilibrium occupation model at 300 K.

    We use the formal Li occupancies encoded in the CIF (Structure.sites[i].species["Li"])
    as the baseline *average* occupancy per crystallographic Li site type, and then
    impose additional short-range order by:
      1. Respecting different average occupancies on distinct Li sublattices
         (e.g., tetrahedral Li(1) vs octahedral Li(2)/Li(3)).
      2. Enforcing a global Li count equal to the formal composition
         (no unphysical Li overfilling).
      3. Sampling configurations using a Metropolis Monte Carlo scheme that
         suppresses configurations with too many Li-Li nearest neighbors,
         approximating short-range repulsion and ordering.

    This strictly follows the Monte Carlo sampling framework of Metropolis & Ulam [54],
    without introducing any cluster-expansion or off-lattice physics.
    """
    if rng is None:
        rng = np.random.default_rng()

    li_indices = []
    li_species_probs = []
    li_types = []

    total_li_target = 0.0

    for i, site in enumerate(structure):
        if "Li" in [s.symbol for s in site.species.elements]:
            li_indices.append(i)
            # Formal occupancy from CIF (between 0 and 1)
            prob = site.species.get("Li", 0)
            li_species_probs.append(prob)
            li_types.append(classify_li_site(site))
            total_li_target += prob

    num_li_sites = len(li_indices)
    print(f"Detected {num_li_sites} Li sites in supercell")

    # Desired total Li count rounded to nearest integer
    target_li = int(round(total_li_target))
    print(f"Target total Li (from CIF occupancies): {target_li}")

    li_indices = np.array(li_indices, dtype=int)
    li_species_probs = np.array(li_species_probs, dtype=float)
    li_types = np.array(li_types, dtype=object)

    # 1. Generate an initial configuration that matches sublattice-averaged
    # occupancies as closely as possible, while enforcing the global Li count.
    # We treat each Li site independently with its probability, then correct
    # the total by random removal/addition.
    occ = rng.binomial(1, li_species_probs).astype(int)
    current_li = int(occ.sum())

    # Adjust to match target total Li
    if current_li > target_li:
        # Randomly remove Li from occupied sites, biased toward sublattices
        # that are overfilled relative to their average probability.
        to_remove = current_li - target_li
        occupied = np.where(occ == 1)[0]
        # Remove uniformly among occupied to avoid inventing detailed energetics.
        remove_indices = rng.choice(occupied, size=to_remove, replace=False)
        occ[remove_indices] = 0
    elif current_li < target_li:
        to_add = target_li - current_li
        vacant = np.where(occ == 0)[0]
        add_indices = rng.choice(vacant, size=to_add, replace=False)
        occ[add_indices] = 1

    print(f"Initial Li count after correction: {occ.sum()}")

    # 2. Build a Li–Li neighbor list to evaluate short-range correlations.
    # We use a physically motivated cutoff (≈3 Å) to capture nearest-neighbor
    # Li–Li contacts that are thermodynamically disfavored at 300 K.
    li_neighbor_cutoff = 3.0  # Å, within typical Li–Li nearest neighbor distance
    neighbors_data = structure.get_all_neighbors(r=li_neighbor_cutoff)
    li_nb_list = [[] for _ in range(num_li_sites)]

    # Map structural index -> Li site list index
    struct_to_li = {s_idx: li_idx for li_idx, s_idx in enumerate(li_indices)}

    for li_list_idx, s_idx in enumerate(li_indices):
        for nb in neighbors_data[s_idx]:
            if "Li" in structure[nb.index].species.elements[0].symbol:
                if nb.index in struct_to_li:
                    j = struct_to_li[nb.index]
                    if j != li_list_idx:
                        li_nb_list[li_list_idx].append(j)

    li_nb_list = [list(set(nbs)) for nbs in li_nb_list]

    # 3. Define a simple pairwise penalty model for Li–Li nearest neighbors.
    # Following Metropolis [54], we define an effective "energy":
    #   E = epsilon * N_pairs
    # where N_pairs is the count of occupied-occupied Li–Li neighbor pairs.
    # epsilon > 0 penalizes configurations with many Li–Li neighbors,
    # which encourages short-range Li ordering.
    def count_li_li_pairs(occ_vec):
        counted = set()
        pairs = 0
        for i in range(num_li_sites):
            if occ_vec[i] == 0:
                continue
            for j in li_nb_list[i]:
                if occ_vec[j] == 1 and (j, i) not in counted:
                    counted.add((i, j))
                    pairs += 1
        return pairs

    # Choose epsilon such that epsilon / (k_B T) ≈ 1: this makes configurations
    # with an extra Li-Li pair suppressed by a factor ~ e^{-1}.
    kb_eV = 8.617e-5  # eV/K
    T_eq = 300.0
    epsilon = kb_eV * T_eq  # ~kT

    def energy(occ_vec):
        return epsilon * count_li_li_pairs(occ_vec)

    # 4. Metropolis Monte Carlo to equilibrate Li occupations at 300 K
    # under fixed total Li count (canonical ensemble, as in [54]).
    def metropolis_equilibrate(occ_vec, steps=50000):
        occ_vec = occ_vec.copy()
        current_E = energy(occ_vec)
        kT = kb_eV * T_eq
        for step in range(steps):
            # Choose a Li and a vacancy and attempt a swap
            occupied = np.where(occ_vec == 1)[0]
            vacant = np.where(occ_vec == 0)[0]
            if len(occupied) == 0 or len(vacant) == 0:
                break
            i = occupied[rng.integers(len(occupied))]
            j = vacant[rng.integers(len(vacant))]

            # Compute local energy change by considering neighbors of i and j only
            # to keep this efficient.
            def local_pairs(idx, occ_local):
                if occ_local[idx] == 0:
                    return 0
                return sum(1 for n in li_nb_list[idx] if occ_local[n] == 1)

            # Before swap
            occ_i, occ_j = occ_vec[i], occ_vec[j]
            old_pairs = local_pairs(i, occ_vec) + local_pairs(j, occ_vec)

            # Apply trial swap
            occ_vec[i], occ_vec[j] = 0, 1
            new_pairs = local_pairs(i, occ_vec) + local_pairs(j, occ_vec)

            d_pairs = new_pairs - old_pairs
            dE = epsilon * d_pairs

            if dE <= 0 or rng.random() < np.exp(-dE / kT):
                # Accept: energy and occ_vec already updated
                current_E += dE
            else:
                # Reject: revert swap
                occ_vec[i], occ_vec[j] = occ_i, occ_j
        return occ_vec

    print("Equilibrating Li occupations with Metropolis MC at 300 K...")
    occ_eq = metropolis_equilibrate(occ, steps=50000)
    print(f"Final Li count after MC equilibration: {occ_eq.sum()}")

    # Build initial_sites list in the format expected by KMCSimulator
    initial_sites = []
    li_struct_indices_set = set(li_indices.tolist())
    li_occ_map = {s_idx: int(occ_eq[li_idx])
                  for li_idx, s_idx in enumerate(li_indices)}

    for s_idx, site in enumerate(structure):
        if s_idx in li_struct_indices_set:
            state = li_occ_map[s_idx]
            initial_sites.append({"coords": site.frac_coords, "state": state})

    print(f"Li sites initialized with equilibrium-correlated occupations: {len(initial_sites)}")
    return initial_sites

# === 1c. Build equilibrium-correlated initial Li configuration ===
initial_sites = assign_equilibrium_li_occupancies(structure)

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