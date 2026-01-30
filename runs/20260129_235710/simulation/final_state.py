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
li_site_indices = []  # map local Li index -> structure index
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(i)

num_li_sites = len(initial_sites)
print(f"Li sites initialized: {num_li_sites}")

# Build inverse map: structure index -> local Li-site index
struct_to_li = {sidx: li_idx for li_idx, sidx in enumerate(li_site_indices)}

# === 2. Build Adjacency Graph (using only Li-Li network) ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {li_idx: [] for li_idx in range(num_li_sites)}

for li_idx, sidx in enumerate(li_site_indices):
    site = structure[sidx]
    for nb in neighbors_data[sidx]:
        nb_sidx = nb.index
        if nb_sidx in struct_to_li:
            tgt_li_idx = struct_to_li[nb_sidx]
            frac_diff = structure[nb_sidx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adj_list[li_idx].append((tgt_li_idx, cart_disp))

print(f"Graph built for Li sublattice (cutoff={cutoff}A)")

# === 2b. Precompute Li-Li neighbor shells for configurational energy ===
# We introduce a configuration-dependent interaction energy that penalizes
# high local Li density via pairwise repulsion: E_conf = 0.5 * V0 * N_pairs,
# with N_pairs counting occupied Li-Li neighbor pairs within r_int.
#
# In the absence of a detailed DFT-derived potential for LLZO here,
# we use a generic short-range pair count to modulate the barrier, consistent
# with the qualitative behavior described in the literature: higher local Li
# density -> higher effective activation barrier and reduced mobility.
#
# Implementation: for each Li site i, we store a list of neighboring Li sites
# j within r_int (interaction cutoff). This is independent of the hopping
# adjacency (which is already limited by 4 Å) and can be chosen somewhat larger
# to capture Coulombic effects.

interaction_cutoff = 4.0  # Å, use same as hop cutoff for simplicity
neighbors_int_data = structure.get_all_neighbors(r=interaction_cutoff)

li_neighbor_shells = {li_idx: [] for li_idx in range(num_li_sites)}

for li_idx, sidx in enumerate(li_site_indices):
    for nb in neighbors_int_data[sidx]:
        nb_sidx = nb.index
        if nb_sidx in struct_to_li:
            j = struct_to_li[nb_sidx]
            if j != li_idx:
                li_neighbor_shells[li_idx].append(j)

print(f"Li-Li interaction neighbor shells built (cutoff={interaction_cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with configuration-dependent barriers ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params,
                 li_neighbor_shells, li_site_indices):
        self.params = params
        self.adj_list = adj_list
        self.li_neighbor_shells = li_neighbor_shells
        self.li_site_indices = li_site_indices

        # Occupancy only on Li sites; index i runs over Li network
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map: Li site index -> particle id; particle positions in Cartesian coords
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                cart = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(cart, dtype=float),
                    'current': np.array(cart, dtype=float)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.E0 = params['E_a']  # base activation barrier (eV)

        # Interaction strength for Li-Li pair penalty at saddle (eV per occupied neighbor)
        # This is the key new parameter controlling how strongly local Li density
        # raises the barrier. Literature on concerted migration in garnets reports
        # barriers ~0.2–0.3 eV; we keep E0 in that range and use a modest penalty
        # so crowded configurations are suppressed without freezing dynamics.
        self.V_pair = params.get('V_pair', 0.05)

    def _count_occupied_neighbors(self, site_idx):
        """Count occupied Li neighbors within interaction shell of a given Li site."""
        neighs = self.li_neighbor_shells.get(site_idx, [])
        if not neighs:
            return 0
        occ = self.occupancy[neighs]
        return int(occ.sum())

    def _hop_barrier(self, src, tgt):
        """
        Configuration-dependent activation barrier for hop src -> tgt.

        We model the saddle energy as:
            E_act(src->tgt) = E0 + V_pair * (n_src_env + n_tgt_env)

        where n_src_env and n_tgt_env count occupied Li neighbors around the
        initial and final sites, excluding the moving ion itself. This reflects
        the Coulomb repulsion: high local Li density around the path raises the
        effective barrier and reduces the hop rate.
        """
        # Occupied neighbors around src (excluding the particle at src)
        n_src_env = 0
        for j in self.li_neighbor_shells.get(src, []):
            if j == src or j == tgt:
                continue
            if self.occupancy[j] == 1:
                n_src_env += 1

        # Occupied neighbors around tgt (site currently empty; count other Li)
        n_tgt_env = 0
        for j in self.li_neighbor_shells.get(tgt, []):
            if j == src or j == tgt:
                continue
            if self.occupancy[j] == 1:
                n_tgt_env += 1

        E_act = self.E0 + self.V_pair * (n_src_env + n_tgt_env)
        return E_act

    def _hop_rate(self, src, tgt):
        """Arrhenius rate for hop src -> tgt with configuration-dependent barrier."""
        E_act = self._hop_barrier(src, tgt)
        return self.nu * np.exp(-E_act / (self.kb * self.params['T']))

    def run_step(self):
        # Build event list with configuration-dependent rates
        events = []
        cumulative_rates = []
        total_rate = 0.0

        for src in self.li_indices:
            if self.occupancy[src] != 1:
                continue
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._hop_rate(src, tgt)
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # No available events -> deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, vec = events[idx]

        # Execute hop
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
        # Mean square displacement (Å^2)
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])
        # Diffusivity (cm^2/s); MSD(t) = 6 D t
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein: σ = (n e^2 D)/(k_B T)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
# Base parameters: E_a from literature (concerted migration barriers ~0.26 eV).
# V_pair introduces additional configuration-dependent penalty due to Li-Li repulsion.
sim_params = {
    'T': 300,
    'E_a': 0.26,         # eV, base barrier consistent with LLZO literature
    'nu': 1e13,          # attempt frequency (1/s)
    'volume': structure.volume,
    'V_pair': 0.04       # eV per occupied neighbor; tunes suppression in dense regions
}

sim = KMCSimulator(structure, adj_list, initial_sites, sim_params,
                   li_neighbor_shells, li_site_indices)

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

        # Keep only last 1000 entries
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criterion
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