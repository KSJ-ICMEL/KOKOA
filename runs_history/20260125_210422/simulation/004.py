"""KOKOA Simulation #4 - 2026-01-25 21:07:39"""
import os, sys, json
import numpy as np
from pymatgen.core import Structure

# Pre-loaded structure with supercell
structure = Structure.from_file("C:/Users/sjkim/KOKOA/LLZO.cif")
N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

'''kMC Simulation for Li-ion Conductivity in Solid Electrolyte with Concerted Two‑Ion Swaps'''

# === 1. Structure Loading ===
# Use the pre‑loaded `structure` variable directly (do not reload CIF)
# The variable `structure` is assumed to be provided in the runtime environment.

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
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[idx] = p_id
                self.particle_positions[p_id] = {'start': np.array(start), 'current': np.array(start)}
                p_id += 1
        
        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0
        
        # Physical constants
        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.base_Ea = params['E_a']
        self.delta_Ea = params.get('delta_Ea', 0.0)  # linear correction per occupied neighbor
        
        # Parameters for concerted two‑ion swaps
        self.nu_swap = params.get('nu_swap', 1e13)
        self.Ea_swap = params.get('Ea_swap', 0.20)  # lower barrier for swaps
        
        # Pre‑compute neighbor lists for barrier correction
        self.neighbor_sites = {site_idx: [tgt for tgt, _ in nbrs] for site_idx, nbrs in adj_list.items()}

    def compute_rate(self, src, tgt):
        """Calculate hop rate with occupancy‑dependent activation energy.
        The effective barrier is:
            Ea_eff = base_Ea + delta_Ea * n_occ
        where n_occ is the number of occupied Li neighbors of both src and tgt
        (excluding the moving ion itself and the target site).
        """
        occ_neighbors = 0
        for neigh in self.neighbor_sites.get(src, []):
            if neigh != tgt and self.occupancy[neigh] == 1:
                occ_neighbors += 1
        for neigh in self.neighbor_sites.get(tgt, []):
            if neigh != src and self.occupancy[neigh] == 1:
                occ_neighbors += 1
        Ea_eff = self.base_Ea + self.delta_Ea * occ_neighbors
        rate = self.nu * np.exp(-Ea_eff / (self.kb * self.params['T']))
        return rate

    def compute_swap_rate(self, src, tgt):
        """Rate for a concerted two‑ion swap between two occupied neighboring sites.
        A simplified constant barrier Ea_swap is used, optionally corrected by
        the same neighbor‑occupation term as single hops.
        """
        # Apply the same neighbor‑occupation correction if desired
        occ_neighbors = 0
        for neigh in self.neighbor_sites.get(src, []):
            if neigh != tgt and self.occupancy[neigh] == 1:
                occ_neighbors += 1
        for neigh in self.neighbor_sites.get(tgt, []):
            if neigh != src and self.occupancy[neigh] == 1:
                occ_neighbors += 1
        Ea_eff = self.Ea_swap + self.delta_Ea * occ_neighbors
        rate = self.nu_swap * np.exp(-Ea_eff / (self.kb * self.params['T']))
        return rate

    def run_step(self):
        events = []          # each entry: (type, src, tgt, vec)
        cum_rates = []
        total = 0.0
        # --- Single‑ion hops ---
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self.compute_rate(src, tgt)
                    if rate > 0:
                        total += rate
                        events.append(('hop', src, tgt, vec))
                        cum_rates.append(total)
        # --- Concerted two‑ion swaps ---
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                # consider each unordered pair once
                if src < tgt and self.occupancy[tgt] == 1:
                    rate = self.compute_swap_rate(src, tgt)
                    if rate > 0:
                        total += rate
                        # For a swap we store both displacement vectors (src->tgt and tgt->src)
                        events.append(('swap', src, tgt, vec))
                        cum_rates.append(total)
        
        if total == 0.0:
            return False  # Deadlock
        
        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1
        
        # Select event
        r = np.random.rand() * total
        idx = np.searchsorted(cum_rates, r)
        ev_type, src, tgt, vec = events[idx]
        
        if ev_type == 'hop':
            # Execute hop (same as original code)
            p_id = self.site_to_particle.pop(src)
            self.particle_positions[p_id]['current'] += vec
            self.occupancy[src], self.occupancy[tgt] = 0, 1
            self.site_to_particle[tgt] = p_id
            self.li_indices.discard(src)
            self.li_indices.add(tgt)
        elif ev_type == 'swap':
            # Execute concerted swap: move both particles
            p_id_src = self.site_to_particle[src]
            p_id_tgt = self.site_to_particle[tgt]
            # Update positions
            self.particle_positions[p_id_src]['current'] += vec
            self.particle_positions[p_id_tgt]['current'] -= vec
            # Occupancy unchanged, but site‑to‑particle mapping swapped
            self.site_to_particle[src], self.site_to_particle[tgt] = p_id_tgt, p_id_src
            # li_indices set remains the same (both sites stay occupied)
        else:
            raise RuntimeError('Unknown event type')
        
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
    'E_a': 0.30,          # base activation energy (eV)
    'nu': 1e13,
    'volume': structure.volume,
    'delta_Ea': 0.05,      # additional barrier per occupied neighbor (eV)
    # Parameters for concerted swaps
    'Ea_swap': 0.20,       # lower barrier for two‑ion swaps (eV)
    'nu_swap': 1e13,
}

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
