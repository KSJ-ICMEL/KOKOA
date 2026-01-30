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

# === 1a. Precompute Neighbor Lists for All Species ===
# We will need Li neighbors (for hops) and O neighbors (for bottleneck geometry).
# The original cutoff for Li-Li connectivity is kept (4.0 Å).
li_li_cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=li_li_cutoff)

# Also precompute O neighbors for each Li site with a reasonable cutoff
# to capture coordinating oxygens forming the bottleneck.
o_neighbor_cutoff = 3.0  # Angstrom (typical Li-O distances in LLZO are ~1.9–2.3 Å)
o_neighbors_data = structure.get_all_neighbors(r=o_neighbor_cutoff)

# === 1b. Identify Li and O Indices ===
li_indices_all = []
o_indices_all = []
for i, site in enumerate(structure):
    elem = site.species.elements[0].symbol
    if elem == "Li":
        li_indices_all.append(i)
    elif elem == "O":
        o_indices_all.append(i)

li_indices_all = np.array(li_indices_all, dtype=int)
o_indices_all = np.array(o_indices_all, dtype=int)

# === 1c. Initialize Li sites with occupancy probability ===
initial_sites = []
for site in structure:
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

print(f"Li sites initialized: {len(initial_sites)}")

# === 2. Build Adjacency Graph with Bottleneck-Modulated Barriers ===
#
# Diagnosis: A uniform geometric cutoff with a single global E_a
# makes every Li-Li hop have the same barrier. To fix this while
# staying evidence-based, we use the tabulated relationship between
# O–O bottleneck separation and activation energy (Table 3 from
# Scientific Reports 11:451 (2021)).
#
# We will:
#   1. Keep the 4.0 Å Li–Li geometric cutoff to define candidate hops.
#   2. For each candidate Li–Li hop, estimate the local O–O bottleneck
#      distance as the minimum distance between oxygens in the first
#      coordination shells of the two Li sites.
#   3. Map this O–O distance to an activation energy using a
#      piecewise linear fit to the O-vacancy data (P–Y) in Table 3:
#         P: 2.75 Å -> 0.84 eV
#         Q: 2.83 Å -> 0.83 eV
#         R: 2.86 Å -> 0.85 eV
#         S: 2.88 Å -> 1.73 eV
#         T: 2.99 Å -> 1.70 eV
#         U: 3.02 Å -> 0.93 eV
#         V: 3.04 Å -> 1.04 eV
#         X: 3.07 Å -> 1.65 eV
#         Y: 3.11 Å -> 1.35 eV
#
#     These data explicitly connect O–O separation and Ea; we use them
#     as a proxy for Li bottleneck sensitivity: wider O–O generally
#     correlates with higher barrier in these paths.
#
#   4. Use hop-specific activation energies and corresponding rates
#      instead of a single global E_a.

# Table 3 data: O–O separation (Å) vs activation energy Ea (eV)
OO_SEP_DATA = np.array([2.75, 2.83, 2.86, 2.88, 2.99, 3.02, 3.04, 3.07, 3.11])
OO_EA_DATA = np.array([0.84, 0.83, 0.85, 1.73, 1.70, 0.93, 1.04, 1.65, 1.35])

def bottleneck_Ea_from_OO_distance(d_oo: float) -> float:
    """
    Map an O–O bottleneck distance to an activation energy (eV)
    by piecewise linear interpolation of Table 3 data.

    For distances outside the tabulated range, we clamp to
    the nearest endpoint to avoid extrapolation.
    """
    if np.isnan(d_oo):
        # If we cannot determine a bottleneck, assign a large barrier
        # so such hops are effectively blocked.
        return 2.0  # eV, artificially high
    # Clamp within the available data range
    d_clamped = min(max(d_oo, OO_SEP_DATA.min()), OO_SEP_DATA.max())
    # Interpolate
    Ea = np.interp(d_clamped, OO_SEP_DATA, OO_EA_DATA)
    return Ea

def compute_min_OO_bottleneck(structure: Structure,
                              li_i: int,
                              li_j: int,
                              o_neighbors_data) -> float:
    """
    Estimate the bottleneck O–O separation controlling a Li hop
    between Li sites li_i and li_j.

    Algorithm:
      * Get O neighbors within o_neighbor_cutoff for Li_i and Li_j.
      * Compute all pairwise O–O distances between O around Li_i
        and O around Li_j, including periodic images.
      * Take the minimum distance as the bottleneck separation.

    Returns:
      d_oo_min (Å) or np.nan if no O neighbors available.
    """
    o_list_i = [nb for nb in o_neighbors_data[li_i]
                if structure[nb.index].species.elements[0].symbol == "O"]
    o_list_j = [nb for nb in o_neighbors_data[li_j]
                if structure[nb.index].species.elements[0].symbol == "O"]

    if len(o_list_i) == 0 or len(o_list_j) == 0:
        return np.nan

    d_min = np.inf
    lat = structure.lattice
    for oi in o_list_i:
        for oj in o_list_j:
            # frac diff with images
            frac_diff = (structure[oj.index].frac_coords + oj.image
                         - (structure[oi.index].frac_coords + oi.image))
            cart_diff = lat.get_cartesian_coords(frac_diff)
            d = np.linalg.norm(cart_diff)
            if d < d_min:
                d_min = d

    if d_min is np.inf:
        return np.nan
    return d_min

# Build adjacency: for each Li site, list neighbors with hop vectors and hop-specific Ea
adj_list = {}
hop_barriers = {}  # (src, tgt) -> Ea

for i, site in enumerate(structure):
    if "Li" not in site.species.elements[0].symbol:
        continue
    neighbors = []
    for nb in neighbors_data[i]:
        if "Li" in structure[nb.index].species.elements[0].symbol:
            # Connectivity still defined purely by Li-Li distance cutoff
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            # Compute bottleneck O–O separation and corresponding Ea
            d_oo = compute_min_OO_bottleneck(structure, i, nb.index, o_neighbors_data)
            Ea_ij = bottleneck_Ea_from_OO_distance(d_oo)
            neighbors.append((nb.index, cart_disp, Ea_ij))
            hop_barriers[(i, nb.index)] = Ea_ij
    adj_list[i] = neighbors

print(f"Graph built (Li-Li cutoff={li_li_cutoff}A) with bottleneck-modulated barriers")

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
        self.kb = kb
        self.nu = params['nu']
        self.T = params['T']
        # We no longer use a single global E_a; instead we compute
        # hop-specific rates from hop-specific E_a (from bottleneck size).

    def hop_rate(self, Ea: float) -> float:
        """
        Arrhenius rate for a hop with activation energy Ea (eV).
        k = ν * exp(-Ea / (k_B T))
        """
        return self.nu * np.exp(-Ea / (self.kb * self.T))

    def run_step(self):
        events, cum_rates = [], []
        total = 0.0

        # Build the event list with hop-specific rates
        for src in self.li_indices:
            for tgt, vec, Ea_ij in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate_ij = self.hop_rate(Ea_ij)
                    if rate_ij <= 0.0:
                        continue
                    total += rate_ij
                    events.append((src, tgt, vec))
                    cum_rates.append(total)

        if total == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0.0, total)
        idx = np.searchsorted(cum_rates, r)
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
        msd = np.mean([np.sum((p['current'] - p['start'])**2)
                       for p in self.particle_positions.values()])  # Å^2
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
# Keep original prefactor and temperature; use structure volume.
sim_params = {
    'T': 300,
    # E_a is now hop-dependent; keep a nominal value here for reference only.
    'E_a': 0.30,
    'nu': 1e13,
    'volume': structure.volume
}
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

        # Check convergence
        if len(sigma_history) > 1000:
            sigma_history.pop(0)  # Keep last 1000

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0

            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")

            if rsd < 0.05:  # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

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