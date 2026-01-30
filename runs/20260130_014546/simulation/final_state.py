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
for site in structure:
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

print(f"Li sites initialized: {len(initial_sites)}")

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

# We also precompute Li–O coordination for each Li site to build an
# environment-dependent barrier model that mimics host-lattice relaxation effects.
li_indices_all = []
li_o_coord_numbers = {}

# Map: structure index -> index in initial_sites (Li-only indexing)
li_structure_to_li_list = {}
li_counter = 0
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        li_structure_to_li_list[i] = li_counter
        li_indices_all.append(i)
        li_counter += 1

# Precompute Li–O neighbors with a reasonable Li–O cutoff
li_o_cutoff = 2.6  # Å, typical Li–O bond-length scale in oxides
all_neighbors_lo = structure.get_all_neighbors(r=li_o_cutoff)

for i in li_indices_all:
    site = structure[i]
    # Count O neighbors within Li–O cutoff
    o_count = 0
    for nb in all_neighbors_lo[i]:
        if "O" in [s.symbol for s in nb.site.species.elements]:
            o_count += 1
    li_o_coord_numbers[i] = o_count

for i, site in enumerate(structure):
    if "Li" not in [s.symbol for s in site.species.elements]:
        continue
    neighbors = []
    for nb in neighbors_data[i]:
        if "Li" in [s.symbol for s in structure[nb.index].species.elements]:
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((nb.index, cart_disp))
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. Environment-Dependent Barrier Model ===
#
# Diagnosis: A single, rigid-lattice barrier E_a = 0.30 eV ignores how Li migration
# couples to local host relaxations (La/Zr–O polyhedra and O-framework distortions),
# which in crowded environments increases barriers and suppresses some pathways.
#
# Consistent with the provided context (NEB barriers in oxides typically ~0.5–0.7 eV),
# we construct a simple local-environment correction that increases the effective
# migration barrier for hops involving highly coordinated (crowded) Li sites.
#
# We do NOT introduce any new physics beyond: barriers depend on local environment,
# and higher local strain/crowding -> larger barrier.

def compute_local_env_barrier(src_index, tgt_index, base_Ea, li_o_coord_numbers):
    """
    Environment-dependent migration barrier:
    - Use Li–O coordination as a proxy for local crowding/strain that couples to
      host-lattice relaxation.
    - Increase the barrier if either the initial or final site is highly coordinated.
    """
    # Li–O coordinations
    cn_src = li_o_coord_numbers.get(src_index, 0)
    cn_tgt = li_o_coord_numbers.get(tgt_index, 0)

    # Reference coordination chosen as a typical value in garnet-like oxides.
    # This is a heuristic mapping, remaining within the energy scale suggested
    # by NEB barriers in the knowledge context (~0.5–0.7 eV).
    cn_ref = 6.0

    # Coordination excess relative to reference for initial and final sites
    excess_src = max(0.0, cn_src - cn_ref)
    excess_tgt = max(0.0, cn_tgt - cn_ref)

    # Penalty per excess O neighbor (eV).
    # With, e.g., 2–3 extra neighbors, the correction is ~0.4–0.6 eV, pushing
    # overall barriers into the realistic range.
    penalty_per_excess = 0.15  # eV per extra O neighbor

    delta_E = penalty_per_excess * (excess_src + excess_tgt)

    # Effective barrier
    E_eff = base_Ea + delta_E
    return E_eff


# === 4. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params,
                 li_o_coord_numbers, li_structure_to_li_list):
        self.params = params
        self.adj_list = adj_list
        self.structure = structure
        self.li_o_coord_numbers = li_o_coord_numbers
        self.li_structure_to_li_list = li_structure_to_li_list

        # Occupancy is defined on the Li-sub-lattice (initial_sites index)
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map from Li-sub-lattice site (index in initial_sites) to structure index
        self.li_list_to_structure = {li_idx: struct_idx
                                     for struct_idx, li_idx in li_structure_to_li_list.items()}

        # Build mapping from structure-based adjacency to Li-sub-lattice adjacency
        self.li_adj_list = {}
        for struct_i, neighbors in adj_list.items():
            if struct_i not in li_structure_to_li_list:
                continue
            li_i = li_structure_to_li_list[struct_i]
            li_neighbors = []
            for struct_j, vec in neighbors:
                if struct_j in li_structure_to_li_list:
                    li_j = li_structure_to_li_list[struct_j]
                    li_neighbors.append((li_j, vec, struct_j))
            self.li_adj_list[li_i] = li_neighbors

        # Build particle mapping
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start),
                    'current': np.array(start)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        kb = 8.617e-5  # eV/K
        self.kb = kb

        # Store base attempt frequency and base activation energy
        self.nu = params['nu']
        self.base_Ea = params['E_a']
        self.T = params['T']

    def _event_rate(self, src_li_idx, tgt_li_idx):
        """
        Compute rate for a hop using environment-dependent barrier.
        """
        # Convert Li-sub-lattice indices back to structure indices to evaluate local environment
        src_struct_idx = self.li_list_to_structure[src_li_idx]
        tgt_struct_idx = self.li_list_to_structure[tgt_li_idx]

        E_eff = compute_local_env_barrier(
            src_struct_idx,
            tgt_struct_idx,
            self.base_Ea,
            self.li_o_coord_numbers
        )

        rate = self.nu * np.exp(-E_eff / (self.kb * self.T))
        return rate

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Enumerate all possible hops and their (env-dependent) rates
        for src_li_idx in list(self.li_indices):
            for tgt_li_idx, vec, struct_j in self.li_adj_list.get(src_li_idx, []):
                if self.occupancy[tgt_li_idx] == 0:
                    rate = self._event_rate(src_li_idx, tgt_li_idx)
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src_li_idx, tgt_li_idx, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src_li_idx, tgt_li_idx, vec = events[idx]

        p_id = self.site_to_particle.pop(src_li_idx)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src_li_idx], self.occupancy[tgt_li_idx] = 0, 1
        self.site_to_particle[tgt_li_idx] = p_id
        self.li_indices.discard(src_li_idx)
        self.li_indices.add(tgt_li_idx)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma


# === 5. Run Simulation ===
# Use the original base barrier as the "unrelaxed" value; the environment-dependent
# corrections will raise many barriers toward realistic values.
sim_params = {
    'T': 300,
    'E_a': 0.30,     # base (unrelaxed) barrier in eV
    'nu': 1e13,      # attempt frequency (1/s)
    'volume': structure.volume
}

sim = KMCSimulator(
    structure=structure,
    adj_list=adj_list,
    initial_sites=initial_sites,
    params=sim_params,
    li_o_coord_numbers=li_o_coord_numbers,
    li_structure_to_li_list=li_structure_to_li_list
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

        # Check convergence (keep last 1000 samples)
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