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

# === 1b. Helpers ===
kb_eV = 8.617e-5  # eV/K

def is_li_site(site):
    return any(sp.symbol == "Li" for sp in site.species.elements)

def get_li_occupancy(site):
    return site.species.get("Li", 0.0)

# === 2. Identify Li Sites and Initialize Occupancies ===
li_site_indices = []
initial_sites = []  # list over Li sites only

for i, site in enumerate(structure):
    if is_li_site(site):
        li_site_indices.append(i)
        prob = get_li_occupancy(site)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({
            "struct_index": i,
            "coords_frac": site.frac_coords,
            "state": state
        })

num_li_sites = len(li_site_indices)
print(f"Li sites (in structure): {num_li_sites}")

# Map between Li-site index (0..num_li_sites-1) and global structure index
li_index_to_struct = {li_i: struct_i for li_i, struct_i in enumerate(li_site_indices)}
struct_to_li_index = {struct_i: li_i for li_i, struct_i in enumerate(li_site_indices)}

# === 3. Build Adjacency Graph on Li Sublattice ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Adjacency in Li-site index space; store neighbor Li-site index and displacement vector
adj_list = {li_i: [] for li_i in range(num_li_sites)}
# Also store hop distances for possible diagnostics/future use
hop_distances = {li_i: [] for li_i in range(num_li_sites)}

for li_i, struct_i in enumerate(li_site_indices):
    site = structure[struct_i]
    neighs = neighbors_data[struct_i]
    for nb in neighs:
        nb_struct_idx = nb.index
        if nb_struct_idx in struct_to_li_index:
            nb_li_i = struct_to_li_index[nb_struct_idx]
            frac_diff = structure[nb_struct_idx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            dist = np.linalg.norm(cart_disp)
            adj_list[li_i].append((nb_li_i, cart_disp))
            hop_distances[li_i].append(dist)

print(f"Li-Li graph built (cutoff={cutoff} Å) with {num_li_sites} Li sites")

# === 4. Coulomb-like Environment Descriptor for Li-Li Interactions ===
# We introduce a simple screened Coulomb penalty that depends on local Li crowding.
# For each Li site, compute an environment metric based on nearby Li sites (independent of occupancy).
# Later, during kMC, we will compute a configuration-dependent Coulomb correction by counting
# actually occupied neighboring Li sites.

env_cutoff = 4.0  # Å, same as adjacency to stay local

# Precompute pair distances between all Li sites within env_cutoff
# Use structure.get_all_neighbors on Li-site indices for environment
env_neighbors = {}
for li_i, struct_i in enumerate(li_site_indices):
    site = structure[struct_i]
    neighs = neighbors_data[struct_i]
    local_pairs = []
    for nb in neighs:
        nb_struct_idx = nb.index
        if nb_struct_idx in struct_to_li_index:
            nb_li_i = struct_to_li_index[nb_struct_idx]
            if nb_li_i == li_i:
                continue
            frac_diff = structure[nb_struct_idx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            dist = np.linalg.norm(cart_disp)
            if dist <= env_cutoff:
                local_pairs.append((nb_li_i, dist))
    env_neighbors[li_i] = local_pairs

# Define a simple distance-dependent Coulomb weight ~ 1/r, capped at short range
def coulomb_weight(r, r_min=1.5):
    r_eff = max(r, r_min)
    return 1.0 / r_eff

# Precompute static geometric weights w_ij = 1/r_ij for all Li-Li pairs within env_cutoff
geom_weights = {}
for li_i in range(num_li_sites):
    neighs = env_neighbors[li_i]
    geom_weights[li_i] = [(nb_li, coulomb_weight(dist)) for nb_li, dist in neighs]

# === 5. kMC Simulator (BKL Algorithm) with Coulomb-corrected Barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.structure = structure
        self.params = params
        self.adj_list = adj_list

        # Occupancy only on Li sites (0..num_li_sites-1)
        self.num_li_sites = len(initial_sites)
        self.occupancy = np.array([s["state"] for s in initial_sites], dtype=int)

        # Map: Li-site index -> particle id (for currently occupied sites)
        self.site_to_particle = {}
        self.particle_positions = {}  # particle_id -> {"start": vec, "current": vec}
        p_id = 0
        for li_i, s in enumerate(initial_sites):
            if s["state"] == 1:
                cart = structure.lattice.get_cartesian_coords(s["coords_frac"])
                self.site_to_particle[li_i] = p_id
                self.particle_positions[p_id] = {
                    "start": np.array(cart, dtype=float),
                    "current": np.array(cart, dtype=float)
                }
                p_id += 1

        self.li_indices = set(range(self.num_li_sites))
        self.num_particles = len(self.particle_positions)
        self.current_time = 0.0
        self.step_count = 0

        # Base NEB-like barrier (rigid lattice, no interactions)
        self.E_a0 = params["E_a0"]  # eV
        self.nu = params["nu"]      # s^-1
        self.T = params["T"]        # K

        # Coulomb penalty parameters
        self.alpha_coul = params.get("alpha_coul", 0.02)  # eV per (dimensionless env unit)
        self.max_coulomb_penalty = params.get("max_coulomb_penalty", 0.25)  # eV

    def _local_coulomb_env(self, li_site_index):
        """
        Compute a configuration-dependent local Coulomb environment metric
        for a given Li site as sum_j (n_j * w_ij), where n_j is occupancy.
        """
        env = 0.0
        for nb_li, weight in geom_weights[li_site_index]:
            if self.occupancy[nb_li] == 1:
                env += weight
        return env

    def _coulomb_correction_for_hop(self, src, tgt):
        """
        Compute ΔE_Coulomb for a hop src -> tgt based on difference in local
        Coulomb crowding between final and initial sites.
        """
        env_src = self._local_coulomb_env(src)
        env_tgt = self._local_coulomb_env(tgt)
        delta_env = env_tgt - env_src

        # If target is more crowded (env_tgt > env_src), add positive penalty.
        # If target is less crowded, allow a small barrier reduction but bound it.
        dE = self.alpha_coul * delta_env
        # Bound Coulomb correction to avoid freezing dynamics or negative barriers
        dE = max(-self.max_coulomb_penalty, min(self.max_coulomb_penalty, dE))
        return dE

    def _effective_barrier(self, src, tgt):
        """
        Effective activation barrier including Coulomb correction:
        E_a_eff = E_a0 + ΔE_Coulomb.
        Enforce a minimum positive barrier.
        """
        dE_coul = self._coulomb_correction_for_hop(src, tgt)
        E_eff = self.E_a0 + dE_coul
        # Ensure barrier remains positive
        E_eff = max(0.05, E_eff)
        return E_eff

    def run_step(self):
        events = []
        cum_rates = []
        total_rate = 0.0

        # Enumerate all possible hops from occupied to vacant neighbor Li sites
        for src in self.li_indices:
            if self.occupancy[src] == 0:
                continue
            for tgt, disp_vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 1:
                    continue
                # Compute configuration-dependent effective barrier
                E_eff = self._effective_barrier(src, tgt)
                rate = self.nu * np.exp(-E_eff / (kb_eV * self.T))
                if rate <= 0.0:
                    continue
                total_rate += rate
                events.append((src, tgt, disp_vec))
                cum_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # No available hops -> deadlock

        # Advance time
        u = np.random.rand()
        self.current_time += -np.log(u) / total_rate
        self.step_count += 1

        # Select event
        r_select = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cum_rates, r_select)
        src, tgt, disp_vec = events[idx]

        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]["current"] += disp_vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id

        return True

    def calculate_properties(self):
        if self.current_time == 0.0 or self.num_particles == 0:
            return 0.0, 0.0
        # Mean square displacement in Å^2
        msd = np.mean([
            np.sum((p["current"] - p["start"]) ** 2)
            for p in self.particle_positions.values()
        ])
        # Diffusivity in cm^2/s (1 Å^2 = 1e-16 cm^2)
        D = msd / (6.0 * self.current_time) * 1e-16

        # Ion concentration (ions/cm^3); structure.volume is in Å^3
        volume_cm3 = self.params["volume"] * 1e-24
        n = self.num_particles / volume_cm3 if volume_cm3 > 0 else 0.0

        # Nernst-Einstein conductivity σ = n e^2 D / (k_B T)
        e_C = 1.602e-19  # C
        k_B_J = 1.38e-23  # J/K
        sigma = (n * e_C ** 2 * D) / (k_B_J * self.T) if self.T > 0 else 0.0
        return msd, sigma

# === 6. Run Simulation ===
# Use a rigid-lattice NEB-like baseline barrier E_a0, then let Coulomb correction modulate it.
sim_params = {
    "T": 300,
    "E_a0": 0.35,        # eV, baseline migration barrier from NEB/MD-like data
    "nu": 1e13,          # s^-1
    "volume": structure.volume,
    # Coulomb correction parameters (kept moderate to avoid freezing)
    "alpha_coul": 0.02,          # eV per unit of env difference
    "max_coulomb_penalty": 0.25  # eV cap
}

sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

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

        # Keep only the last 1000 conductivity samples for convergence
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

            if rsd < 0.05:
                print(
                    f"Convergence reached (RSD < 5%) "
                    f"at {sim.current_time*1e9:.2f} ns"
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

print("\n=== Simulation Complete ===")
print(f"T={sim_params['T']} K, Time={sim.current_time*1e9:.2f} ns")
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
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f} ns "
        f"with Coulomb-corrected barriers (E_a0={sim_params['E_a0']} eV, "
        f"alpha_coul={sim_params['alpha_coul']} eV/unit)"
    ),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")