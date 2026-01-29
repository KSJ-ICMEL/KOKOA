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
        initial_sites.append({"index": i, "coords": site.frac_coords, "state": state})
        li_site_indices.append(i)

print(f"Li sites initialized: {len(initial_sites)}")

# === 2. Build Li–Li Adjacency Graph with Geometry Data ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Map from Li-site list position to structure index and back
li_list_to_struct = {li_pos: s["index"] for li_pos, s in enumerate(initial_sites)}
struct_to_li_list = {s["index"]: li_pos for li_pos, s in enumerate(initial_sites)}

# Build adjacency list on Li-site index space and store hop distances
adj_list = {li_pos: [] for li_pos in range(len(initial_sites))}
hop_distances = {li_pos: [] for li_pos in range(len(initial_sites))}

for li_pos, site_info in enumerate(initial_sites):
    struct_idx = site_info["index"]
    site = structure[struct_idx]
    neighbors = []
    dists = []
    for nb in neighbors_data[struct_idx]:
        nb_struct_idx = nb.index
        if "Li" in structure[nb_struct_idx].species.elements[0].symbol:
            if nb_struct_idx not in struct_to_li_list:
                continue
            tgt_li_pos = struct_to_li_list[nb_struct_idx]
            frac_diff = structure[nb_struct_idx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            dist = np.linalg.norm(cart_disp)
            neighbors.append((tgt_li_pos, cart_disp))
            dists.append(dist)
    adj_list[li_pos] = neighbors
    hop_distances[li_pos] = dists

print(f"Li–Li graph built (cutoff={cutoff}Å)")

# === 2b. Compute Simple Bottleneck Geometry Descriptor ===
# Proxy for bottleneck openness using Li–Li distances:
# - For each hop, the distance serves as a crude measure of path width.
# - We map distances to a normalized [0,1] scale to use in barrier calculation.

all_dists = []
for src, dlist in hop_distances.items():
    all_dists.extend(dlist)
all_dists = np.array(all_dists) if len(all_dists) > 0 else np.array([cutoff])

dist_min = float(np.min(all_dists))
dist_max = float(np.max(all_dists)) if np.max(all_dists) > dist_min else dist_min + 1e-6

def normalize_distance(d):
    """Normalize Li–Li hop distance to [0,1]."""
    return (d - dist_min) / (dist_max - dist_min)

# === 3. kMC Simulator (BKL Algorithm) with Geometry-Dependent Barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, hop_distances, params):
        self.params = params
        self.structure = structure
        self.adj_list = adj_list
        self.hop_distances = hop_distances

        # Occupancy on Li-site index space
        self.num_li_sites = len(initial_sites)
        self.occupancy = np.array([s["state"] for s in initial_sites], dtype=int)

        # Map Li-site -> particle, track particle positions in Cartesian space
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_pos, s in enumerate(initial_sites):
            if s["state"] == 1:
                start = structure.lattice.get_cartesian_coords(s["coords"])
                self.site_to_particle[li_pos] = p_id
                self.particle_positions[p_id] = {
                    "start": np.array(start, dtype=float),
                    "current": np.array(start, dtype=float),
                }
                p_id += 1

        self.li_indices = set(range(self.num_li_sites))
        self.num_particles = len(self.particle_positions)
        self.current_time = 0.0
        self.step_count = 0

        self.kb = 8.617e-5  # eV/K, Boltzmann constant

        # Geometry-dependent barrier model parameters (heuristic, LLZO-inspired)
        # Global "base" activation energy (eV), near MD-extracted LLZO value
        self.Ea_global = params.get("E_a", 0.345)

        # Allowable barrier window (eV), based on garnet-like ranges from context
        self.Ea_min = params.get("E_a_min", 0.20)
        self.Ea_max = params.get("E_a_max", 0.60)

        # Scaling factor for distance effect
        self.dist_scale = params.get("dist_scale", 0.3)

        # Precompute per-hop normalized distances for efficiency
        self.norm_hop_distances = {
            src: [normalize_distance(d) for d in dlist]
            for src, dlist in self.hop_distances.items()
        }

    def compute_Ea_for_hop(self, src, idx_in_list):
        """
        Geometry-dependent barrier:
        - Use Li–Li distance (proxy for bottleneck size).
        - Shorter distances (tighter bottleneck) → higher barrier.
        - Longer distances (more open path) → lower barrier.
        Map to [Ea_min, Ea_max] around Ea_global.
        """
        d_norm = self.norm_hop_distances[src][idx_in_list]  # in [0,1]
        # Center around 0.5 so that distances near the mean give Ea_global
        # d_norm < 0.5 (short path) → positive shift (increase barrier)
        # d_norm > 0.5 (longer path) → negative shift (decrease barrier)
        shift = self.dist_scale * (0.5 - d_norm)
        Ea = self.Ea_global + shift
        Ea_clipped = float(np.clip(Ea, self.Ea_min, self.Ea_max))
        return Ea_clipped

    def run_step(self):
        events = []
        cum_rates = []
        total_rate = 0.0

        T = self.params["T"]
        nu = self.params["nu"]

        # Enumerate all allowed Li hops and compute hop-specific rates
        for src in self.li_indices:
            if self.occupancy[src] != 1:
                continue
            neighbors = self.adj_list.get(src, [])
            for local_idx, (tgt, vec) in enumerate(neighbors):
                if self.occupancy[tgt] == 0:
                    Ea_ij = self.compute_Ea_for_hop(src, local_idx)
                    rate_ij = nu * np.exp(-Ea_ij / (self.kb * T))
                    if rate_ij <= 0.0:
                        continue
                    total_rate += rate_ij
                    events.append((src, tgt, vec))
                    cum_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock: no allowed events

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cum_rates, r)
        src, tgt, vec = events[idx]

        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]["current"] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id

        # li_indices is the set of Li sites capable of hosting Li;
        # membership doesn't change, only occupancy does.
        return True

    def calculate_properties(self):
        if self.current_time == 0 or self.num_particles == 0:
            return 0.0, 0.0
        msd = np.mean(
            [
                np.sum((p["current"] - p["start"]) ** 2)
                for p in self.particle_positions.values()
            ]
        )  # Å^2
        # Diffusivity (cm^2/s), using MSD(t) = 6 D t in 3D and 1 Å^2 = 1e-16 cm^2
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3), volume in Å^3 → cm^3
        n = self.num_particles / (self.params["volume"] * 1e-24)
        # Conductivity via Nernst–Einstein: σ = (n e^2 D) / (k_B T)
        e_charge = 1.602e-19  # C
        k_B_SI = 1.38e-23  # J/K
        sigma = (n * e_charge**2 * D) / (k_B_SI * self.params["T"])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    "T": 300,               # K
    "E_a": 0.345,           # eV, LLZO-like activation energy from MD/Arrhenius
    "nu": 1e13,             # 1/s, attempt frequency
    "volume": structure.volume,
    # Geometry-dependent barrier parameters
    "E_a_min": 0.20,        # eV, lower bound for favorable, wide bottlenecks
    "E_a_max": 0.60,        # eV, upper bound for tight bottlenecks
    "dist_scale": 0.3,      # strength of distance dependence
}

sim = KMCSimulator(structure, adj_list, initial_sites, hop_distances, sim_params)

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

        # Keep only last 1000 samples
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

            if rsd < 0.05:  # 5% convergence criterion
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
    "temperature_K": sim_params["T"],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")