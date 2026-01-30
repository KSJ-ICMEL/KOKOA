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
li_site_indices = []  # map reduced Li index -> structure index
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(i)

num_li_sites = len(initial_sites)
print(f"Li sites initialized: {num_li_sites}")

# Build mapping: structure index -> Li index (compressed)
struct_to_li = {s_idx: li_idx for li_idx, s_idx in enumerate(li_site_indices)}

# === 2. Build Adjacency Graph (between Li sites only) ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {li_idx: [] for li_idx in range(num_li_sites)}

for li_idx, s_idx in enumerate(li_site_indices):
    site = structure[s_idx]
    for nb in neighbors_data[s_idx]:
        nb_struct_idx = nb.index
        # Only consider Li neighbors
        if "Li" in structure[nb_struct_idx].species.elements[0].symbol:
            if nb_struct_idx not in struct_to_li:
                continue
            tgt_li_idx = struct_to_li[nb_struct_idx]
            frac_diff = structure[nb_struct_idx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adj_list[li_idx].append((tgt_li_idx, cart_disp))

print(f"Graph built (cutoff={cutoff}A)")

# --- 2b. Build short-range Li–Li neighbor list for Coulombic penalty ---
# We use the same Li–Li cutoff, but this could be tuned separately if needed.
li_neighbor_list = {li_idx: [] for li_idx in range(num_li_sites)}
for li_idx, s_idx in enumerate(li_site_indices):
    site = structure[s_idx]
    for nb in neighbors_data[s_idx]:
        nb_struct_idx = nb.index
        if "Li" in structure[nb_struct_idx].species.elements[0].symbol:
            if nb_struct_idx not in struct_to_li:
                continue
            n_li_idx = struct_to_li[nb_struct_idx]
            if n_li_idx == li_idx:
                continue
            li_neighbor_list[li_idx].append(n_li_idx)

print("Local Li–Li environment list built for Coulombic penalty")

# === 3. kMC Simulator (BKL Algorithm) with environment-dependent barrier ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params,
                 li_neighbor_list, li_site_indices):
        self.params = params
        self.adj_list = adj_list
        self.li_neighbor_list = li_neighbor_list
        self.li_site_indices = li_site_indices

        # Occupancy on Li sublattice (compressed index)
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map sites to particles and track positions
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                # Convert fractional coords of Li site index in structure to Cartesian
                struct_idx = li_site_indices[li_idx]
                start = structure[struct_idx].coords
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

        # Constants
        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.base_Ea = params['E_a']
        # Strength of Coulombic penalty per additional Li neighbor (eV)
        # This is a phenomenological parameter, consistent with
        # environment-dependent barriers used in kMCpy-like approaches.
        self.alpha = params.get('alpha', 0.05)  # eV per neighbor difference

    def _count_li_neighbors(self, li_idx):
        """Count occupied Li neighbors around site li_idx (short-range)."""
        n_occ = 0
        for n_idx in self.li_neighbor_list.get(li_idx, []):
            if self.occupancy[n_idx] == 1:
                n_occ += 1
        return n_occ

    def _hop_rate(self, src, tgt):
        """
        Environment-dependent hop rate:
        E_migration = E_a + alpha * max(0, n_tgt - n_src)
        where n_src and n_tgt are the counts of occupied Li neighbors
        (excluding the hopping ion) around source and target sites.
        This penalizes moves into more Li-crowded environments,
        mimicking Coulombic repulsion in LLZO as in cluster-expansion
        based kMC (e.g., kMCpy, Morgan 2017, Catlow 1983).
        """
        n_src = self._count_li_neighbors(src)
        n_tgt = self._count_li_neighbors(tgt)
        delta_n = n_tgt - n_src
        # Only penalize moves into more crowded environments
        E_m = self.base_Ea + self.alpha * max(0, delta_n)
        rate = self.nu * np.exp(-E_m / (self.kb * self.params['T']))
        return rate

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Build event list with environment-dependent rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    r = self._hop_rate(src, tgt)
                    if r <= 0:
                        continue
                    total_rate += r
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        rnd = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, rnd)
        src, tgt, vec = events[idx]

        # Move particle
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
        # Mean Square Displacement (Å^2)
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])
        # Diffusivity (cm^2/s), MSD(t)=6Dt
        D = msd / (6 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein Equation: σ = (n*e^2*D)/(k*T) (S/cm)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma


# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    'E_a': 0.30,     # base migration barrier (eV)
    'nu': 1e13,      # attempt frequency (1/s)
    'volume': structure.volume,
    # Environment-dependent Coulombic penalty coefficient (eV per extra Li neighbor)
    # Chosen to suppress hops into highly crowded regions, aligning
    # with LLZO NEB / cluster-expansion observations where local
    # Li-rich configurations are energetically penalized.
    'alpha': 0.10
}

sim = KMCSimulator(
    structure=structure,
    adj_list=adj_list,
    initial_sites=initial_sites,
    params=sim_params,
    li_neighbor_list=li_neighbor_list,
    li_site_indices=li_site_indices
)

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
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns "
        f"with environment-dependent migration barriers (alpha={sim_params['alpha']} eV)."
    ),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")