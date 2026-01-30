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
        initial_sites.append({"coords": site.frac_coords, "state": state, "index": i})
        li_site_indices.append(i)

print(f"Li sites initialized: {len(initial_sites)}")

# Map from structure index -> local Li index
struct_to_local_li = {s["index"]: i for i, s in enumerate(initial_sites)}
local_li_to_struct = {i: s["index"] for i, s in enumerate(initial_sites)}

# === 2. Build Adjacency Graph with hop classification ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

# Helper: classify Li site as "tet" or "oct" based on coordination as a simple proxy.
# This is a heuristic proxy for crystallographic 24d (tetrahedral) vs 48g (octahedral) Li sites.
def classify_li_site(structure, idx):
    site = structure[idx]
    # Count nearby O atoms within 2.6 Å as a crude coordination proxy
    o_count = 0
    for nb in neighbors_data[idx]:
        if "O" in structure[nb.index].species.elements[0].symbol and nb.nn_distance <= 2.6:
            o_count += 1
    # Tetrahedral (4) vs octahedral (6) coordination proxy
    if o_count <= 4:
        return "tet"
    else:
        return "oct"

li_site_type = {}
for sidx in li_site_indices:
    li_site_type[sidx] = classify_li_site(structure, sidx)

# Hop-type dependent activation energies (eV) and attempt prefactors (Hz)
# Based on internal LLZO paper: lowest local path ~0.26 eV but long-range 3D channels ~0.45 eV.
# We use a simple mapping:
#   oct<->oct and mixed hops assigned to 0.45 eV (dominant 3D transport barrier),
#   tet<->tet assigned slightly lower 0.30 eV as local low-barrier segments.
HOP_PARAMS = {
    ("tet", "tet"): {"E_a": 0.30, "nu": 1e13},
    ("tet", "oct"): {"E_a": 0.45, "nu": 1e13},
    ("oct", "tet"): {"E_a": 0.45, "nu": 1e13},
    ("oct", "oct"): {"E_a": 0.45, "nu": 1e13},
}

kb = 8.617e-5  # eV/K

# Build adjacency between local Li indices and store hop-type information
for local_src, entry in enumerate(initial_sites):
    src_struct_idx = entry["index"]
    neighbors = []
    for nb in neighbors_data[src_struct_idx]:
        nb_struct_idx = nb.index
        if nb_struct_idx not in struct_to_local_li:
            continue
        local_tgt = struct_to_local_li[nb_struct_idx]
        if local_tgt == local_src:
            continue
        # Build displacement vector in Cartesian from src to tgt including image shift
        frac_diff = structure[nb_struct_idx].frac_coords - structure[src_struct_idx].frac_coords + nb.image
        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)

        src_type = li_site_type[src_struct_idx]
        tgt_type = li_site_type[nb_struct_idx]
        hop_key = (src_type, tgt_type)
        hop_info = HOP_PARAMS.get(hop_key, HOP_PARAMS[("oct", "oct")])  # default to 0.45 eV
        neighbors.append(
            {
                "tgt": local_tgt,
                "disp": cart_disp,
                "src_type": src_type,
                "tgt_type": tgt_type,
                "E_a": hop_info["E_a"],
                "nu": hop_info["nu"],
            }
        )
    adj_list[local_src] = neighbors

print(f"Graph built with hop-type classification (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with path-specific barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list

        # Occupancy array over local Li sites
        self.occupancy = np.array([s["state"] for s in initial_sites], dtype=int)

        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for local_idx, s in enumerate(initial_sites):
            if s["state"] == 1:
                start = structure.lattice.get_cartesian_coords(s["coords"])
                self.site_to_particle[local_idx] = p_id
                self.particle_positions[p_id] = {
                    "start": np.array(start, dtype=float),
                    "current": np.array(start, dtype=float),
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # Precompute Arrhenius prefactors for hop types at this temperature
        self.hop_rate_cache = {}
        for key, vals in HOP_PARAMS.items():
            Ea = vals["E_a"]
            nu = vals["nu"]
            rate = nu * np.exp(-Ea / (kb * params["T"]))
            self.hop_rate_cache[key] = rate

    def _hop_rate(self, src_type, tgt_type):
        key = (src_type, tgt_type)
        return self.hop_rate_cache.get(key, self.hop_rate_cache[("oct", "oct")])

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Build event list with path-specific rate constants
        for src in self.li_indices:
            if self.occupancy[src] == 0:
                continue
            for hop in self.adj_list.get(src, []):
                tgt = hop["tgt"]
                if self.occupancy[tgt] == 0:
                    rate = self._hop_rate(hop["src_type"], hop["tgt_type"])
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, hop["disp"]))
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
        self.particle_positions[p_id]["current"] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
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
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params["T"])
        return msd, sigma

# === 4. Run Simulation ===
# Use 0.45 eV as the representative 3D transport activation energy at the macroscopic level.
sim_params = {"T": 300, "E_a": 0.45, "nu": 1e13, "volume": structure.volume}
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
    "conductivity": float(sigma),
    "diffusivity": float(D),
    "msd": float(msd),
    "simulation_time_ns": float(sim.current_time * 1e9),
    "temperature_K": sim_params["T"],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")