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
        initial_sites.append({"coords": site.frac_coords, "state": state, "site_index": i})
        li_site_indices.append(i)

print(f"Li sites initialized: {len(initial_sites)}")

# === 1a. Classify Li Site Types (Environment-Dependent Kinetics) ===
# We approximate site types by local coordination environment (proxy for 24d vs 48g/96h).
# This is a focused change to introduce environment-dependent barriers/rates, as diagnosed.
from collections import defaultdict

# Map from structure index to local environment descriptor
site_env = {}

# Use a neighbor cutoff typical for Li-O coordination
env_cutoff = 3.0  # Angstrom
all_neighbors = structure.get_all_neighbors(r=env_cutoff)

for i, site in enumerate(structure):
    if "Li" not in [s.symbol for s in site.species.elements]:
        continue
    o_count = 0
    la_count = 0
    zr_count = 0
    other_cations = 0
    for nb in all_neighbors[i]:
        elem = structure[nb.index].species.elements[0].symbol
        if elem == "O":
            o_count += 1
        elif elem == "La":
            la_count += 1
        elif elem in ("Zr", "Ta", "Nb"):
            zr_count += 1
        else:
            other_cations += 1

    # Very simple heuristic: lower O coordination -> tetra-like; higher -> octa/interstitial
    # This is purely a structural classification, not adding new physics.
    if o_count <= 4:
        site_type = "tet"  # proxy for 24d
    else:
        site_type = "oct"  # proxy for 48g/96h-like / interstitial

    site_env[i] = {
        "o_count": o_count,
        "la_count": la_count,
        "zr_count": zr_count,
        "other_cations": other_cations,
        "type": site_type,
    }

# Define hop-type-dependent activation energies from literature guidance:
# - Local paths with lowest barriers ~0.26 eV (fast conducting channels).
# - Long-range effective activation ~0.45 eV in tetragonal LLZO.
# We therefore distinguish low-barrier (channel) hops from higher-barrier (bottleneck) hops.
def classify_hop_type(src_idx, tgt_idx):
    src_type = site_env[src_idx]["type"]
    tgt_type = site_env[tgt_idx]["type"]
    # Simple rules:
    # tet <-> tet: likely part of low-barrier channels (path C-like) -> 0.26 eV
    # tet <-> oct: mixed environment, moderate barrier -> 0.35 eV
    # oct <-> oct: higher barrier, often bottleneck -> 0.45 eV
    if src_type == "tet" and tgt_type == "tet":
        return "fast_channel"
    elif src_type == "oct" and tgt_type == "oct":
        return "slow_bottleneck"
    else:
        return "mixed"

HOP_EA = {
    "fast_channel": 0.26,      # eV, from lowest path C
    "mixed": 0.35,             # eV, intermediate between 0.26 and 0.45
    "slow_bottleneck": 0.45,   # eV, consistent with long-range activation
}

HOP_NU = {
    "fast_channel": 1e13,      # Hz, standard phonon attempt frequency
    "mixed": 1e13,
    "slow_bottleneck": 5e12,   # slightly reduced prefactor for unfavorable hops
}

# === 2. Build Adjacency Graph with Hop-Specific Kinetics ===
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
            hop_type = classify_hop_type(i, nb.index)
            Ea = HOP_EA[hop_type]
            nu = HOP_NU[hop_type]
            neighbors.append(
                {
                    "tgt": nb.index,
                    "disp": cart_disp,
                    "hop_type": hop_type,
                    "Ea": Ea,
                    "nu": nu,
                }
            )
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}A) with environment-dependent hop parameters")

# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list

        # Build array over full structure size so indexing matches structure indices
        max_index = max(s["site_index"] for s in initial_sites) + 1
        self.occupancy = np.zeros(max_index, dtype=int)
        for s in initial_sites:
            self.occupancy[s["site_index"]] = s["state"]

        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for s in initial_sites:
            idx = s["site_index"]
            if s["state"] == 1:
                start = structure.lattice.get_cartesian_coords(s["coords"])
                self.site_to_particle[idx] = p_id
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

    def hop_rate(self, Ea, nu):
        # Transition rate from harmonic transition state theory:
        # k = nu * exp(-Ea / (k_B * T))
        return nu * np.exp(-Ea / (self.kb * self.params["T"]))

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        for src in list(self.li_indices):
            for nb in self.adj_list.get(src, []):
                tgt = nb["tgt"]
                if tgt < len(self.occupancy) and self.occupancy[tgt] == 0:
                    rate = self.hop_rate(nb["Ea"], nb["nu"])
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, nb["disp"]))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance: Δt = -ln(ξ) / Σ_i k_i
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, vec = events[idx]

        # Execute event
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]["current"] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        msd = np.mean(
            [
                np.sum((p["current"] - p["start"]) ** 2)
                for p in self.particle_positions.values()
            ]
        )  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params["volume"] * 1e-24)  # ions/cm^3
        sigma = (
            n * (1.602e-19) ** 2 * D / (1.38e-23 * self.params["T"])
        )  # S/cm (Nernst-Einstein)
        return msd, sigma

# === 4. Run Simulation ===
# Base parameters remain but rates now depend on hop-specific Ea, nu.
sim_params = {"T": 300, "volume": structure.volume}
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

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criteria
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
    "temperature_K": sim_params["T"],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")