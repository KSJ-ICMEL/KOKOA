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

# Map from global structure index to Li sublattice index
global_to_li = {s["index"]: li_i for li_i, s in enumerate(initial_sites)}
li_to_global = {li_i: s["index"] for li_i, s in enumerate(initial_sites)}

# === 2. Build Adjacency Graph on Li Sublattice ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {li_i: [] for li_i in range(len(initial_sites))}

for li_i, s in enumerate(initial_sites):
    g_idx = s["index"]
    site = structure[g_idx]
    for nb in neighbors_data[g_idx]:
        nb_gidx = nb.index
        if nb_gidx in global_to_li:
            tgt_li = global_to_li[nb_gidx]
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adj_list[li_i].append((tgt_li, cart_disp))

print(f"Graph built on Li sublattice (cutoff={cutoff}A)")

# === 2b. Precompute second-neighbor pairs for concerted 2-ion exchanges ===
# We define a concerted event as two Li ions on sites i and j exchanging with two
# neighboring vacant sites k and l, where (i,k) and (j,l) are nearest neighbors
# and (k,l) are also neighbors (a local chain/loop of length 4). This is a simple
# proxy for cooperative motion documented for LLZO/LGPS/LATP (concerted chains/loops).
second_neighbor_pairs = {}  # key: li_i, value: list of (j, k, l, disp_i, disp_j)
for i in adj_list:
    pairs = []
    neigh_i = [n for n, _ in adj_list[i]]
    for j in neigh_i:
        # neighbors of j
        neigh_j = [n for n, _ in adj_list[j]]
        # candidate shared/connected neighbors for concerted motion
        for k, disp_ik in adj_list[i]:
            if k == j:
                continue
            for l, disp_jl in adj_list[j]:
                if l == i or l == k:
                    continue
                # require k and l to be neighbors (forming local loop/chain)
                if any(nn == l for nn, _ in adj_list[k]):
                    pairs.append((j, k, l, disp_ik, disp_jl))
    second_neighbor_pairs[i] = pairs

print("Precomputed local pairs for concerted 2-ion exchanges")

# === 3. kMC Simulator (BKL Algorithm) with configuration-dependent and concerted rates ===
class KMCSimulator:
    def __init__(self, structure, adj_list, second_neighbor_pairs, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.second_neighbor_pairs = second_neighbor_pairs

        # Occupancy on Li sublattice
        self.num_sites = len(initial_sites)
        self.occupancy = np.array([s["state"] for s in initial_sites], dtype=int)

        # Site -> particle mapping and trajectories
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_i, s in enumerate(initial_sites):
            if s["state"] == 1:
                start = structure.lattice.get_cartesian_coords(s["coords"])
                self.site_to_particle[li_i] = p_id
                self.particle_positions[p_id] = {
                    "start": np.array(start),
                    "current": np.array(start),
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        kb = 8.617e-5  # eV/K

        # Base attempt frequencies and activation energies
        # Single-ion parameters (classical isolated hop)
        self.nu_single = params["nu_single"]
        self.Ea_single = params["E_a_single"]

        # Concerted 2-ion exchange parameters.
        # Papers on LLZO/LGPS/LATP show lower migration barriers for concerted paths
        # than for isolated classical hops (flatter energy landscape along channels).
        self.nu_concerted = params["nu_concerted"]
        self.Ea_concerted = params["E_a_concerted"]

        # Local configuration penalty: additional barrier per occupied neighbor
        # around the target site, representing blocking/Coulomb cost in classical picture.
        # Cooperative motion partly relieves this; we keep it smaller for concerted hops.
        self.delta_E_neighbor_single = params.get("delta_E_neighbor_single", 0.02)
        self.delta_E_neighbor_concerted = params.get("delta_E_neighbor_concerted", 0.005)

        self.kb = kb

    def _count_occupied_neighbors(self, site_index):
        occ = 0
        for nb, _ in self.adj_list.get(site_index, []):
            if self.occupancy[nb] == 1:
                occ += 1
        return occ

    def _single_hop_rate(self, src, tgt):
        """
        Configuration-dependent single-particle hop rate.
        Rate = nu_single * exp(- (Ea_single + ΔE_env) / (kB T))
        where ΔE_env = delta_E_neighbor_single * (N_occ(target) - N_ref)
        Here we use N_ref = 4 as a loose reference coordination; only excess crowding
        increases the barrier, mimicking site blocking and Coulomb repulsion.
        """
        N_occ_tgt = self._count_occupied_neighbors(tgt)
        N_ref = 4.0
        extra_occ = max(0.0, N_occ_tgt - N_ref)
        deltaE = self.delta_E_neighbor_single * extra_occ
        Ea_eff = self.Ea_single + deltaE
        rate = self.nu_single * np.exp(-Ea_eff / (self.kb * self.params["T"]))
        return rate

    def _concerted_two_ion_rate(self, i, j, k, l):
        """
        Configuration-dependent rate for concerted two-ion exchange:
        i -> k and j -> l simultaneously.
        Cooperative motion reduces the penalty from local crowding;
        we still include a weaker dependence on occupied neighbors at k and l.
        """
        N_occ_k = self._count_occupied_neighbors(k)
        N_occ_l = self._count_occupied_neighbors(l)
        N_ref = 4.0
        extra_occ_k = max(0.0, N_occ_k - N_ref)
        extra_occ_l = max(0.0, N_occ_l - N_ref)
        deltaE = self.delta_E_neighbor_concerted * (extra_occ_k + extra_occ_l)
        Ea_eff = self.Ea_concerted + deltaE
        rate = self.nu_concerted * np.exp(-Ea_eff / (self.kb * self.params["T"]))
        return rate

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # --- Single-ion hops with configuration-dependent rates ---
        for src in list(self.li_indices):
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._single_hop_rate(src, tgt)
                    if rate <= 0:
                        continue
                    total_rate += rate
                    events.append(("single", src, tgt, vec, rate))
                    cumulative_rates.append(total_rate)

        # --- Concerted 2-ion exchange events ---
        for i in list(self.li_indices):
            # i must be occupied
            if self.occupancy[i] == 0:
                continue
            for (j, k, l, disp_ik, disp_jl) in self.second_neighbor_pairs.get(i, []):
                # j must also be occupied, k and l vacant
                if (
                    j in self.li_indices
                    and self.occupancy[j] == 1
                    and self.occupancy[k] == 0
                    and self.occupancy[l] == 0
                ):
                    rate = self._concerted_two_ion_rate(i, j, k, l)
                    if rate <= 0:
                        continue
                    total_rate += rate
                    events.append(
                        (
                            "concerted2",
                            i,
                            j,
                            k,
                            l,
                            disp_ik,
                            disp_jl,
                            rate,
                        )
                    )
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        event = events[idx]

        # Execute event
        if event[0] == "single":
            _, src, tgt, vec, _rate = event
            p_id = self.site_to_particle.pop(src)
            self.particle_positions[p_id]["current"] += vec
            self.occupancy[src] = 0
            self.occupancy[tgt] = 1
            self.site_to_particle[tgt] = p_id
            self.li_indices.discard(src)
            self.li_indices.add(tgt)
        elif event[0] == "concerted2":
            _, i, j, k, l, disp_ik, disp_jl, _rate = event
            # First particle: i -> k
            p_i = self.site_to_particle.pop(i)
            self.particle_positions[p_i]["current"] += disp_ik
            # Second particle: j -> l
            p_j = self.site_to_particle.pop(j)
            self.particle_positions[p_j]["current"] += disp_jl

            # Update occupancy and indices
            self.occupancy[i] = 0
            self.occupancy[j] = 0
            self.occupancy[k] = 1
            self.occupancy[l] = 1

            self.site_to_particle[k] = p_i
            self.site_to_particle[l] = p_j

            self.li_indices.discard(i)
            self.li_indices.discard(j)
            self.li_indices.add(k)
            self.li_indices.add(l)
        else:
            # Unknown event type
            return False

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
        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s)
        n = self.num_particles / (self.params["volume"] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params["T"])
        return msd, sigma


# === 4. Run Simulation ===
# Parameters chosen to reflect lower barrier for concerted hops versus single hops,
# consistent with NEB/AIMD findings for LLZO/LGPS/LATP:
# Ea_concerted < Ea_single, and cooperative motion partially mitigates
# the environment penalty (delta_E_neighbor_concerted < delta_E_neighbor_single).
sim_params = {
    "T": 300,
    "E_a_single": 0.35,       # eV, classical isolated hop barrier (larger)
    "E_a_concerted": 0.22,    # eV, lower barrier for concerted mechanism
    "nu_single": 1e13,        # Hz
    "nu_concerted": 5e12,     # Hz, slightly smaller attempt frequency for multi-ion move
    "delta_E_neighbor_single": 0.02,   # eV per excess occupied neighbor
    "delta_E_neighbor_concerted": 0.005,  # eV per excess occupied neighbor (weaker)
    "volume": structure.volume,
}

sim = KMCSimulator(structure, adj_list, second_neighbor_pairs, initial_sites, sim_params)

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

        # Check convergence using a running window
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
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns with configuration-dependent and concerted events",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")