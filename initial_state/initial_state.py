"""kMC Simulation for Li-ion Conductivity in Solid Electrolyte
   - Li sites: occupied (state=1)
   - He sites: vacancy placeholder (state=0)
   - Li can hop to He sites (vacancy migration)
"""
import numpy as np
from pymatgen.core import Structure
import os

# === 1. Structure Loading ===
cif_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLZO_with_vacancy.cif")
if not os.path.exists(cif_path):
    raise FileNotFoundError(f"CIF file not found: {cif_path}")

structure = Structure.from_file(cif_path)

N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

# === 2. Initialize Li and He(vacancy) sites ===
initial_sites = []
site_to_idx = {}  # Map structure index to initial_sites index

li_count = 0
he_count = 0

for idx, site in enumerate(structure):
    element = site.species.elements[0].symbol
    if element == "Li":
        # Li site: occupied (state=1)
        initial_sites.append({"coords": site.frac_coords, "state": 1, "type": "Li"})
        site_to_idx[idx] = len(initial_sites) - 1
        li_count += 1
    elif element == "He":
        # He site: vacancy (state=0)
        initial_sites.append({"coords": site.frac_coords, "state": 0, "type": "He"})
        site_to_idx[idx] = len(initial_sites) - 1
        he_count += 1

print(f"Li sites (occupied): {li_count}")
print(f"He sites (vacancy): {he_count}")
print(f"Total hop sites: {len(initial_sites)}")

# === 3. Build Adjacency Graph (Li-Li and Li-He connections) ===
# adj_list uses initial_sites indices (not structure indices)
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for i, site in enumerate(structure):
    element = site.species.elements[0].symbol
    if element not in ["Li", "He"]:
        continue
    
    src_idx = site_to_idx.get(i)
    if src_idx is None:
        continue
    
    neighbors = []
    for nb in neighbors_data[i]:
        nb_element = structure[nb.index].species.elements[0].symbol
        if nb_element in ["Li", "He"]:
            tgt_idx = site_to_idx.get(nb.index)
            if tgt_idx is not None:
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                neighbors.append((tgt_idx, cart_disp))  # Use initial_sites index
    adj_list[src_idx] = neighbors

print(f"Graph built (cutoff={cutoff}A, includes Li-He connections)")

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
        
        kb = 8.617e-5  # eV/K
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

    def run_step(self):
        events, rates, total = [], [], 0.0
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    total += self.base_rate
                    events.append((src, tgt, vec))
                    rates.append(total)
        
        if total == 0:
            return False  # Deadlock
        
        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1
        
        # Select and execute event
        idx = np.searchsorted(rates, np.random.uniform(0, total))
        src, tgt, vec = events[idx]
        
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0, 0
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()]) # Mean Square Displacement (Å^2)
        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s), MSD(t)=6Dt
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # Nernst-Einstein Equation: σ = (n*e^2*D)/(k*T) (S/cm)
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {'T': 298, 'E_a': 0.30, 'nu': 1e13, 'volume': structure.volume}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

target_time = 5e-9  # 5ns (DO NOT MODIFY)
log_interval = 2000

while sim.current_time < target_time:
    if not sim.run_step():
        print("Deadlock - stopping")
        break
    if sim.step_count % log_interval == 0:
        msd, sigma = sim.calculate_properties()
        print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

# Final result
msd, sigma = sim.calculate_properties()
D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0

print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D={D:.4e} cm^2/s")
print(f"Conductivity: {sigma:.4e} S/cm")

# Save result to JSON
import json
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
print(f"\n📁 결과 저장: {result_path}")