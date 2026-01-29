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
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(idx)

num_li_sites = len(initial_sites)
print(f"Li sites initialized: {num_li_sites}")

# Map from structure index to compressed Li-site index
struct_to_li = {}
li_to_struct = {}
li_counter = 0
for idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        struct_to_li[idx] = li_counter
        li_to_struct[li_counter] = idx
        li_counter += 1

# === 2. Build Adjacency Graph (Li-Li network) ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {i: [] for i in range(num_li_sites)}

for li_i in range(num_li_sites):
    struct_i = li_to_struct[li_i]
    site_i = structure[struct_i]
    neighbors = []
    for nb in neighbors_data[struct_i]:
        if "Li" in [s.symbol for s in structure[nb.index].species.elements]:
            li_j = struct_to_li[nb.index]
            frac_diff = structure[nb.index].frac_coords - site_i.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((li_j, cart_disp))
    adj_list[li_i] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 2b. Precompute coordination neighbors for cooperative penalties ===
# We use a short-range Li-Li distance to estimate local crowding around a target site
coord_cutoff = 3.0  # Angstrom, typical nearest-neighbor Li-Li distance scale
coord_neighbors = {i: set() for i in range(num_li_sites)}

# Build a list of Li Cartesian positions for neighbor search
li_cart_coords = [structure.lattice.get_cartesian_coords(s["coords"]) for s in initial_sites]
li_cart_coords = np.array(li_cart_coords)

for i in range(num_li_sites):
    for j in range(i + 1, num_li_sites):
        disp = li_cart_coords[j] - li_cart_coords[i]
        dist = np.linalg.norm(disp)
        if dist <= coord_cutoff:
            coord_neighbors[i].add(j)
            coord_neighbors[j].add(i)

print(f"Coordination neighbors built (cutoff={coord_cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with Correlation-Aware Rates ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params, coord_neighbors):
        self.params = params
        self.adj_list = adj_list
        self.coord_neighbors = coord_neighbors

        # Occupancy only on Li sublattice (compressed indexing)
        self.occupancy = np.array([s["state"] for s in initial_sites], dtype=int)

        # Map Li site -> particle id and track particle trajectories
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

        kb = 8.617e-5  # eV/K

        # Two distinct activation energies:
        # - E_single: isolated single-ion hops (higher barrier)
        # - E_coop  : hops into congested environments that require cooperative/rearrangement
        self.base_rate_single = params["nu"] * np.exp(-params["E_single"] / (kb * params["T"]))
        self.base_rate_coop = params["nu"] * np.exp(-params["E_coop"] / (kb * params["T"]))

        # Threshold for "congested" final environment in terms of occupied Li neighbors
        self.congestion_threshold = params.get("congestion_threshold", 2)

    def _local_occupancy_around(self, site_idx):
        """
        Count number of occupied coordination neighbors around a given Li site.
        """
        count = 0
        for nb in self.coord_neighbors[site_idx]:
            if self.occupancy[nb] == 1:
                count += 1
        return count

    def _event_rate(self, src, tgt):
        """
        Correlation-aware rate:
        If the target site is surrounded by many Li (high local occupancy),
        treat the hop as requiring cooperative motion / vacancy-push,
        and use a different effective barrier.
        """
        # Number of occupied Li neighbors around target after hop,
        # assuming source becomes vacant and target becomes occupied.
        occ_count = 0
        for nb in self.coord_neighbors[tgt]:
            if nb == src:
                # Source becomes empty after the hop, so do not count it
                continue
            if self.occupancy[nb] == 1:
                occ_count += 1

        # If environment is crowded, use cooperative barrier
        if occ_count >= self.congestion_threshold:
            return self.base_rate_coop
        else:
            return self.base_rate_single

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Build event list with correlation-aware rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    r = self._event_rate(src, tgt)
                    if r <= 0.0:
                        continue
                    total_rate += r
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # No possible events (deadlock)

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        rand_val = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cumulative_rates, rand_val)
        src, tgt, vec = events[idx]

        # Move particle
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
        )  # Mean Square Displacement (Å^2)
        D = msd / (6.0 * self.current_time) * 1e-16  # Diffusivity (cm^2/s)
        n = self.num_particles / (self.params["volume"] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params["T"])
        return msd, sigma


# === 4. Run Simulation ===
# For LLZO, correlated multi-ion pathways can lower the effective barrier of
# cooperative hops compared to isolated hops. We represent this via two barriers:
# - E_single ~ 0.50 eV (isolated / high-barrier hop)
# - E_coop   ~ 0.34 eV (concerted migration consistent with cubic LLZO data)
sim_params = {
    "T": 300,
    "E_single": 0.50,  # eV, isolated single-particle hop barrier
    "E_coop": 0.34,    # eV, effective barrier for cooperative / concerted hops
    "nu": 1e13,
    "volume": structure.volume,
    # A target site with >= congestion_threshold occupied neighbors
    # is treated as requiring a cooperative move.
    "congestion_threshold": 2,
}

sim = KMCSimulator(structure, adj_list, initial_sites, sim_params, coord_neighbors)

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

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criteria
                print(
                    f"Convergence reached (RSD < 5%) at "
                    f"{sim.current_time*1e9:.2f}ns"
                )
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
            )

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6.0 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

print("\n=== Simulation Complete ===")
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