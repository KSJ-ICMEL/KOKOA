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

# Map from structure index to Li-site index (0..n_Li-1)
struct_to_li_site = {s["struct_index"]: li_idx for li_idx, s in enumerate(initial_sites)}
li_site_to_struct = {li_idx: s["struct_index"] for li_idx, s in enumerate(initial_sites)}

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for li_idx, s in enumerate(initial_sites):
    struct_idx = s["struct_index"]
    neighbors = []
    for nb in neighbors_data[struct_idx]:
        nb_struct_idx = nb.index
        if nb_struct_idx in struct_to_li_site:
            tgt_li_idx = struct_to_li_site[nb_struct_idx]
            frac_diff = structure[nb_struct_idx].frac_coords - structure[struct_idx].frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((tgt_li_idx, cart_disp))
    adj_list[li_idx] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# Precompute Li–Li local environment metric: average Li–Li distance within r_env
r_env = 3.5  # Angstrom
# Build a neighbor list restricted to Li sites for environment calculation
li_positions_cart = np.array([
    structure.lattice.get_cartesian_coords(s["coords"]) for s in initial_sites
])
num_li_sites = len(initial_sites)
env_neighbors = [[] for _ in range(num_li_sites)]

# Simple O(N^2) environment neighbor determination; acceptable for moderate supercells
for i in range(num_li_sites):
    for j in range(i + 1, num_li_sites):
        diff = li_positions_cart[j] - li_positions_cart[i]
        dist = np.linalg.norm(diff)
        if dist <= r_env:
            env_neighbors[i].append((j, dist))
            env_neighbors[j].append((i, dist))

# Compute average neighbor distance for each site; use a large value if isolated
avg_env_dist = np.zeros(num_li_sites)
for i in range(num_li_sites):
    if env_neighbors[i]:
        avg_env_dist[i] = np.mean([d for (_, d) in env_neighbors[i]])
    else:
        avg_env_dist[i] = r_env  # isolated, treat as relatively open

# Normalize environment metric to [0,1]
env_min = np.min(avg_env_dist)
env_max = np.max(avg_env_dist)
if env_max > env_min:
    norm_env = (avg_env_dist - env_min) / (env_max - env_min)
else:
    norm_env = np.zeros_like(avg_env_dist)

# Precompute hop distances (cartesian) between Li sites for adjacency graph
hop_distances = {}
for src, nbs in adj_list.items():
    for tgt, vec in nbs:
        d = np.linalg.norm(vec)
        hop_distances[(src, tgt)] = d

# Normalize hop distances across all edges to [0,1]
if hop_distances:
    all_dists = np.array(list(hop_distances.values()))
    d_min = np.min(all_dists)
    d_max = np.max(all_dists)
    if d_max > d_min:
        norm_hop_dist = {k: (v - d_min) / (d_max - d_min) for k, v in hop_distances.items()}
    else:
        norm_hop_dist = {k: 0.0 for k in hop_distances.keys()}
else:
    norm_hop_dist = {}

# === 3. kMC Simulator (BKL Algorithm) with environment-dependent barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params,
                 norm_env, norm_hop_dist):
        self.params = params
        self.adj_list = adj_list
        self.norm_env = norm_env
        self.norm_hop_dist = norm_hop_dist

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

        self.kb = 8.617e-5  # eV/K

        # DFT/NEB-informed LLZO-like values for long-range transport
        # Overall activation energy for 3D Li migration channels ~0.45 eV (paper/expt)
        # Use a modest heterogeneous range bracketing this value
        self.Ea_min = 0.35  # eV, lower bound for most favorable local hops
        self.Ea_max = 0.60  # eV, upper bound for more constrained hops
        # Fallback global value representative of long-range pathway barrier
        self.Ea_global = 0.45  # eV
        self.nu = params['nu']

        # We no longer use a single base_rate; rates are hop-specific
        # self.base_rate = self.nu * np.exp(-self.Ea_global / (self.kb * params['T']))

        # Scales for how strongly distance and environment asymmetry affect Ea
        # Chosen modest to avoid over-penalizing as per diagnosis
        self.dist_scale = 0.4
        self.asym_scale = 0.3

    def _compute_Ea(self, src, tgt):
        """
        Compute a hop-specific activation energy based on:
          - normalized hop distance
          - local environment asymmetry (average Li–Li distance metric)
        """
        # Normalized hop distance; fall back to mid value if missing
        d_norm = self.norm_hop_dist.get((src, tgt), 0.5)

        # Normalized local environment around sites (proxy for crowding/strain)
        e_src = self.norm_env[src]
        e_tgt = self.norm_env[tgt]
        asym = abs(e_src - e_tgt)

        # Base barrier around global long-range value
        Ea = self.Ea_global

        # Distance: longer / more constricted hops slightly increase Ea
        Ea += self.dist_scale * (d_norm - 0.5)

        # Environment asymmetry: mismatched local environments modestly increase Ea
        Ea += self.asym_scale * asym

        # Clip to physically motivated range from LLZO context
        Ea = max(self.Ea_min, min(self.Ea_max, Ea))
        return Ea

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        T = self.params['T']

        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    Ea_ij = self._compute_Ea(src, tgt)
                    rate_ij = self.nu * np.exp(-Ea_ij / (self.kb * T))
                    if rate_ij <= 0.0:
                        continue
                    total_rate += rate_ij
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

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
        if self.current_time == 0:
            return 0.0, 0.0
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()
        ])  # Å^2
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
# Use attempt frequency from context; Ea now environment-dependent, not a single value
sim_params = {'T': 300, 'E_a': 0.45, 'nu': 1e13, 'volume': structure.volume}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params,
                   norm_env=norm_env, norm_hop_dist=norm_hop_dist)

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
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0

            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")

            if rsd < 0.05:
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

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