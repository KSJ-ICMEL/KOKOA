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

# Initialize Li sites with occupancy probability and keep mapping to structure indices
initial_sites = []
li_site_indices = []
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state, "struct_index": i})
        li_site_indices.append(i)

num_li_sites = len(initial_sites)
print(f"Li sites initialized: {num_li_sites}")

# Map between local Li-site index and global structure index
li_local_to_struct = {i: s["struct_index"] for i, s in enumerate(initial_sites)}
struct_to_li_local = {s["struct_index"]: i for i, s in enumerate(initial_sites)}

# === 2. Build Adjacency Graph on Li Sublattice ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {i: [] for i in range(num_li_sites)}  # adjacency in Li-local indices

for li_local, struct_idx in li_local_to_struct.items():
    site = structure[struct_idx]
    for nb in neighbors_data[struct_idx]:
        nb_struct_idx = nb.index
        # Only consider Li neighbors that are part of our Li site list
        if nb_struct_idx in struct_to_li_local:
            nb_local = struct_to_li_local[nb_struct_idx]
            frac_diff = structure[nb_struct_idx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adj_list[li_local].append((nb_local, cart_disp))

print(f"Graph built on Li sublattice (cutoff={cutoff}A)")

# === 2b. Assign simple site energies to encode 24d vs 96h preference ===
#
# We cannot robustly identify exact Wyckoff positions without full symmetry
# analysis, but we can introduce a minimal, physically motivated site-energy
# contrast using local Li environment as a proxy:
#
# - Tetrahedral 24d sites are lower energy for Li in cubic LLZO (Li prefers 24d).
# - More open, less crowded environments (fewer/ farther Li neighbors) are
#   therefore assigned lower site energies.
#
# This restores a non-flat energy landscape so that forward/backward rates
# differ and Li occupancy is biased toward low-energy (24d-like) sites.

# Compute a local environment metric for each Li site: average Li-Li distance
# within a moderate cutoff. Larger average distance => more open (24d-like).
env_cutoff = 3.5  # Angstrom, within Li sublattice
li_cart_coords = np.array([
    structure.lattice.get_cartesian_coords(s["coords"]) for s in initial_sites
])

# Distance matrix on Li sublattice
diff = li_cart_coords[:, None, :] - li_cart_coords[None, :, :]
dist_mat = np.linalg.norm(diff, axis=-1)

# For each site, get neighbors within env_cutoff (excluding self) and compute mean distance
env_metric = np.zeros(num_li_sites)
for i in range(num_li_sites):
    mask = (dist_mat[i] > 1e-5) & (dist_mat[i] <= env_cutoff)
    if np.any(mask):
        env_metric[i] = dist_mat[i, mask].mean()
    else:
        # Isolated site; treat as very open environment
        env_metric[i] = env_cutoff

# Normalize environment metric to [0, 1]
if env_metric.max() > env_metric.min():
    env_norm = (env_metric - env_metric.min()) / (env_metric.max() - env_metric.min())
else:
    env_norm = np.zeros_like(env_metric)

# Define site energies (in eV) from normalized environment:
# More open (larger env_metric => higher env_norm) -> lower site energy.
# We take a modest contrast so as not to freeze dynamics but restore
# thermodynamic bias: 24d-like lower by ~0.1 eV relative to 96h-like.
delta_E = 0.10  # total spread of site energies in eV
E_site = delta_E * (1.0 - env_norm)  # open env_norm ~1 -> E_site ~0; crowded -> higher

# Shift so that minimum is zero for convenience (only differences matter)
E_site -= E_site.min()

# Precompute pairwise site-energy differences for fast lookup
# ΔE_site(i->j) = E_site[j] - E_site[i]
delta_E_site_matrix = E_site[None, :] - E_site[:, None]


# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params,
                 li_cart_coords, delta_E_site_matrix):
        self.params = params
        self.adj_list = adj_list
        self.num_sites = len(initial_sites)
        self.delta_E_site_matrix = delta_E_site_matrix

        # Occupancy on Li sublattice
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map site -> particle id, and track particle trajectories
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for site_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start_cart = li_cart_coords[site_idx]
                self.site_to_particle[site_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start_cart, dtype=float),
                    'current': np.array(start_cart, dtype=float)
                }
                p_id += 1

        self.occupied_sites = set(self.site_to_particle.keys())
        self.num_particles = len(self.occupied_sites)
        self.current_time = 0.0
        self.step_count = 0

        # Base migration barrier (config-independent part)
        self.E_m = params['E_m']  # eV
        self.nu = params['nu']
        self.T = params['T']
        self.kb = 8.617e-5  # eV/K

    def _hop_rate(self, src, tgt):
        """
        Compute hop rate for Li moving from src -> tgt Li site.

        We include:
        - a common migration barrier E_m (saddle-point energy above some reference)
        - a site-energy difference term ΔE_site = E_site[tgt] - E_site[src]
          to enforce detailed balance:
              k_{i->j} ∝ exp( - (E_m + max(ΔE_site, 0)) / k_BT )
        This keeps the forward hop over the higher of the two site energies plus
        the migration barrier, while the backward hop over the reverse barrier
        is reduced when hopping back downhill in site energy.
        """
        dE_site = self.delta_E_site_matrix[src, tgt]  # E_site[tgt] - E_site[src]

        # Following a standard lattice kMC scheme for site-energy landscapes:
        # Use barrier: E_eff = E_m + max(dE_site, 0)
        # This ensures that:
        # - Uphill hops (to higher-energy site) pay the extra site-energy cost.
        # - Downhill hops keep the original barrier E_m, making them more favorable.
        E_eff = self.E_m + max(dE_site, 0.0)

        rate = self.nu * np.exp(-E_eff / (self.kb * self.T))
        return rate

    def run_step(self):
        events = []
        rates_cum = []
        total_rate = 0.0

        # Enumerate all possible hops from occupied to vacant neighboring sites
        for src in list(self.occupied_sites):
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    k_ij = self._hop_rate(src, tgt)
                    if k_ij <= 0.0:
                        continue
                    total_rate += k_ij
                    events.append((src, tgt, vec))
                    rates_cum.append(total_rate)

        if total_rate == 0.0:
            return False  # no available moves -> deadlock

        # BKL time advance
        r1 = np.random.rand()
        self.current_time += -np.log(r1) / total_rate
        self.step_count += 1

        # Select event
        r2 = np.random.rand() * total_rate
        idx = np.searchsorted(rates_cum, r2)
        src, tgt, vec = events[idx]

        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id
        self.occupied_sites.discard(src)
        self.occupied_sites.add(tgt)

        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        # Mean square displacement in Å^2
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])
        # Diffusivity (cm^2/s), using MSD(t) = 6 D t in 3D and 1 Å^2 = 1e-16 cm^2
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3); structure.volume is in Å^3
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein conductivity: σ = (n e^2 D)/(k_B T)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma


# === 4. Run Simulation ===
# Use a more realistic migration barrier (without site energies) for LLZO-like systems.
# The added site-energy contrasts will effectively increase the overall activation
# energy for long-range transport relative to a flat landscape model.
sim_params = {
    'T': 300,
    'E_m': 0.45,        # eV, baseline migration barrier (without site energy difference)
    'nu': 1e13,         # 1/s
    'volume': structure.volume
}

sim = KMCSimulator(
    structure=structure,
    adj_list=adj_list,
    initial_sites=initial_sites,
    params=sim_params,
    li_cart_coords=li_cart_coords,
    delta_E_site_matrix=delta_E_site_matrix
)

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
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            # Convergence criterion: RSD < 5%
            if rsd < 0.05:
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