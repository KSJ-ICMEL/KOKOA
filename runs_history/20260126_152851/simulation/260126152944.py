import os, sys, json
import numpy as np
from pymatgen.core import Structure

# Load structure (CIF is in current directory)
structure = Structure.from_file("LLZO.cif")
N = 4
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

for i, site in enumerate(structure):
    if "Li" not in site.species.elements[0].symbol:
        continue
    neighbors = []
    for nb in neighbors_data[i]:
        if "Li" in structure[nb.index].species.elements[0].symbol:
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((nb.index, cart_disp))
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.structure = structure
        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.T = params['T']
        self.base_Ea = params['E_a_base']

        # Mapping from global Li site index -> occupancy array index
        self.site_to_occ_idx = {}
        self.occ_idx_to_site = {}
        occ_list = []
        occ_counter = 0
        for idx, site in enumerate(structure):
            if "Li" in site.species.elements[0].symbol:
                self.site_to_occ_idx[idx] = occ_counter
                self.occ_idx_to_site[occ_counter] = idx
                occ_list.append(initial_sites[occ_counter]['state'])
                occ_counter += 1
        self.occupancy = np.array(occ_list, dtype=int)

        # Particle bookkeeping
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for occ_idx, state in enumerate(self.occupancy):
            if state == 1:
                site_idx = self.occ_idx_to_site[occ_idx]
                start = structure.lattice.get_cartesian_coords(initial_sites[occ_idx]['coords'])
                self.site_to_particle[site_idx] = p_id
                self.particle_positions[p_id] = {'start': np.array(start), 'current': np.array(start)}
                p_id += 1
        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

    def get_barrier(self, src, tgt):
        """Return a configuration‑dependent activation energy for a hop src→tgt.
        Simple model: base_Ea + 0.10 eV * (number of occupied neighboring Li sites of the target).
        """
        occupied_neighbors = 0
        for nb_idx, _ in self.adj_list.get(tgt, []):
            if nb_idx == src:
                continue
            occ_idx = self.site_to_occ_idx.get(nb_idx)
            if occ_idx is not None and self.occupancy[occ_idx] == 1:
                occupied_neighbors += 1
        return self.base_Ea + 0.10 * occupied_neighbors

    def run_step(self):
        events = []          # (src, tgt, vec, rate)
        cumulative = []
        total_rate = 0.0
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                occ_idx_tgt = self.site_to_occ_idx.get(tgt)
                if occ_idx_tgt is None:
                    continue
                if self.occupancy[occ_idx_tgt] == 0:
                    Ea = self.get_barrier(src, tgt)
                    rate = self.nu * np.exp(-Ea / (self.kb * self.T))
                    total_rate += rate
                    events.append((src, tgt, vec, rate))
                    cumulative.append(total_rate)
        if total_rate == 0.0:
            return False  # deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Choose event
        r = np.random.rand() * total_rate
        idx = np.searchsorted(cumulative, r)
        src, tgt, vec, _ = events[idx]

        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        occ_idx_src = self.site_to_occ_idx[src]
        occ_idx_tgt = self.site_to_occ_idx[tgt]
        self.occupancy[occ_idx_src] = 0
        self.occupancy[occ_idx_tgt] = 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0, 0
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 298,
    'E_a_base': 0.30,  # base activation energy (eV)
    'nu': 1e13,
    'volume': structure.volume
}

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
            sigma_history.pop(0)
        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")
            if rsd < 0.05:
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0

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
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns"
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")