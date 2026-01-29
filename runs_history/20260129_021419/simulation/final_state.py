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
li_site_indices = []  # map Li-site list index -> structure index
for idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(idx)

num_li_sites = len(initial_sites)
print(f"Li sites initialized: {num_li_sites}")

# === 2. Build Adjacency Graph on Li Sublattice ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Build adjacency in Li-site index space (0..num_li_sites-1)
adj_list = {li_idx: [] for li_idx in range(num_li_sites)}

# Map structure index -> local Li-site index
struct_to_li = {s_idx: li_idx for li_idx, s_idx in enumerate(li_site_indices)}

for li_local_i, struct_i in enumerate(li_site_indices):
    site_i = structure[struct_i]
    for nb in neighbors_data[struct_i]:
        struct_j = nb.index
        # only consider Li neighbors
        if struct_j not in struct_to_li:
            continue
        li_local_j = struct_to_li[struct_j]
        frac_diff = structure[struct_j].frac_coords - site_i.frac_coords + nb.image
        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
        # store neighbor in Li-site index space
        adj_list[li_local_i].append((li_local_j, cart_disp))

print(f"Li-Li graph built (cutoff={cutoff}A)")

# Precompute simple environment descriptor: local Li crowding for each Li site
env_cutoff = 3.5  # Angstrom
li_cart_coords = np.array(
    [structure.lattice.get_cartesian_coords(s["coords"]) for s in initial_sites]
)

env_metric = np.zeros(num_li_sites, dtype=float)
for i in range(num_li_sites):
    # distances to all other Li sites (including itself)
    disp = li_cart_coords - li_cart_coords[i]
    d2 = np.einsum("ij,ij->i", disp, disp)
    mask = (d2 > 1e-6) & (d2 < env_cutoff**2)
    if np.any(mask):
        # local crowding as count of neighbors within env_cutoff
        env_metric[i] = np.sum(mask)
    else:
        env_metric[i] = 0.0

# normalize environment metric to [0,1]
if env_metric.max() > env_metric.min():
    env_norm = (env_metric - env_metric.min()) / (env_metric.max() - env_metric.min())
else:
    env_norm = np.zeros_like(env_metric)

# Precompute a per-hop effective barrier that depends on static local environment
# This introduces heterogeneous, configuration-independent barriers that
# mimic blocking/correlation in an averaged way, without per-step recomputation.
Ea_global = 0.45  # eV, approximate LLZO long-range activation energy
Ea_min = 0.35     # eV lower bound
Ea_max = 0.60     # eV upper bound
env_scale = 0.15  # eV amplitude for environment penalty

# hop_Ea[li_i] = list of (li_j, disp, Ea_ij)
hop_Ea = {i: [] for i in range(num_li_sites)}
for i in range(num_li_sites):
    ei = env_norm[i]
    for j, disp in adj_list[i]:
        ej = env_norm[j]
        # symmetric environment factor for this static hop
        e_avg = 0.5 * (ei + ej)
        # more crowded environments (higher e_avg) get slightly higher barriers
        Ea_ij = Ea_global + env_scale * (e_avg - 0.5)
        # clamp to physical range
        Ea_ij = max(Ea_min, min(Ea_max, Ea_ij))
        hop_Ea[i].append((j, disp, Ea_ij))

# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, hop_Ea, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.hop_Ea = hop_Ea
        # occupancy only on Li sites (0..num_li_sites-1)
        self.occupancy = np.array([s["state"] for s in initial_sites], dtype=int)

        # mapping: Li-site index -> particle id (only for occupied sites)
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s["state"] == 1:
                start = structure.lattice.get_cartesian_coords(s["coords"])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    "start": np.array(start, dtype=float),
                    "current": np.array(start, dtype=float),
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        self.kb = 8.617e-5  # eV/K

    def run_step(self):
        events = []
        cum_rates = []
        total_rate = 0.0

        T = self.params["T"]
        nu = self.params["nu"]

        # Enumerate possible hops with precomputed static, environment-dependent barriers
        for src in self.li_indices:
            for (tgt, vec, Ea_ij) in self.hop_Ea.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = nu * np.exp(-Ea_ij / (self.kb * T))
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    cum_rates.append(total_rate)
                    events.append((src, tgt, vec))

        if total_rate == 0.0:
            return False  # No available events (deadlock)

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cum_rates, r)
        src, tgt, vec = events[idx]

        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]["current"] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id

        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0.0 or self.num_particles == 0:
            return 0.0, 0.0
        msd = np.mean(
            [
                np.sum((p["current"] - p["start"]) ** 2)
                for p in self.particle_positions.values()
            ]
        )  # Å^2
        # Diffusivity (cm^2/s), MSD(t)=6Dt, convert Å^2 -> cm^2 with 1e-16
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3), volume in Å^3 -> cm^3 with 1e-24
        n = self.num_particles / (self.params["volume"] * 1e-24)
        # Nernst-Einstein Equation: σ = (n*e^2*D)/(k*T) (S/cm)
        sigma = (
            n * (1.602e-19) ** 2 * D / (1.38e-23 * self.params["T"])
            if D > 0.0
            else 0.0
        )
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {"T": 300, "E_a": Ea_global, "nu": 1e13, "volume": structure.volume}
sim = KMCSimulator(structure, adj_list, hop_Ea, initial_sites, sim_params)

target_time = 1000e-9  # 1000 ns
log_interval = 100
sigma_history = []

while sim.current_time < target_time:
    if not sim.run_step():
        print("Deadlock - stopping")
        break
    if sim.step_count % log_interval == 0:
        msd, sigma = sim.calculate_properties()
        sigma_history.append(sigma)

        # Keep only last 1000 values for RSD calculation
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
            if rsd < 0.05:
                print(
                    f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns"
                )
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
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