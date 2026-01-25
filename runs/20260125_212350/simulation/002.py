"""KOKOA Simulation #2 - 2026-01-25 21:24:44"""
import os, sys, json
import numpy as np
from pymatgen.core import Structure

# Pre-loaded structure with supercell
structure = Structure.from_file("C:/Users/sjkim/KOKOA/LLZO.cif")
N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

initial_sites = []
for site in structure:
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

# Build adjacency list (already created as adj_list in the original script)

class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        
        # Map site index -> particle id and store particle trajectories
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
        self.T = params['T']
        self.repulsion_coeff = params.get('repulsion_coeff', 0.10)  # eV per occupied neighbor
        
        # Base barriers for different hop types (eV). In a real implementation these would be read from DFT data.
        # Here we distinguish only by whether the source site is a 24d or 96h Wyckoff position.
        # For simplicity we infer the type from the coordination number (24d sites have 4 Li neighbors, 96h have 6).
        self.base_barriers = {
            '24d-96h': params.get('E_a_base', 0.45),
            '96h-24d': params.get('E_a_base', 0.45),
            '96h-96h': params.get('E_a_base', 0.55),
            '24d-24d': params.get('E_a_base', 0.60)
        }
        
        # Pre‑compute a simple site‑type dictionary based on neighbor count (heuristic).
        self.site_type = {}
        for i in range(len(structure)):
            li_neighbors = sum(1 for nb_idx, _ in self.adj_list.get(i, []) if "Li" in structure[nb_idx].species.elements)
            self.site_type[i] = '24d' if li_neighbors <= 4 else '96h'

    def _hop_rate(self, src, tgt):
        """Calculate the rate for a hop from src to tgt using environment‑dependent barriers."""
        src_type = self.site_type.get(src, '96h')
        tgt_type = self.site_type.get(tgt, '96h')
        key = f"{src_type}-{tgt_type}"
        E_base = self.base_barriers.get(key, self.params['E_a'])
        # Count occupied Li neighbors around the target site (excluding the source site which will become vacant)
        occupied_neighbors = 0
        for nb_idx, _ in self.adj_list.get(tgt, []):
            if nb_idx == src:
                continue
            if self.occupancy[nb_idx] == 1:
                occupied_neighbors += 1
        E_rep = self.repulsion_coeff * occupied_neighbors
        E_a = E_base + E_rep
        return self.nu * np.exp(-E_a / (self.kb * self.T))

    def run_step(self):
        events = []          # list of (src, tgt, vec, cumulative_rate)
        cumulative = 0.0
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._hop_rate(src, tgt)
                    if rate <= 0:
                        continue
                    cumulative += rate
                    events.append((src, tgt, vec, cumulative))
        if cumulative == 0.0:
            return False  # deadlock
        # Advance time using total rate
        self.current_time += -np.log(np.random.rand()) / cumulative
        self.step_count += 1
        # Choose event
        r = np.random.rand() * cumulative
        # Linear search (acceptable for modest system size)
        for src, tgt, vec, cum_rate in events:
            if r <= cum_rate:
                break
        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
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

# Simulation parameters – extended with new barrier model entries
sim_params = {
    'T': 298,
    'nu': 1e13,
    'volume': structure.volume,
    'E_a_base': 0.45,          # base barrier for the most common hop (eV)
    'repulsion_coeff': 0.10    # additional penalty per occupied neighbor (eV)
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
print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D={D:.4e} cm^2/s")
print(f"Conductivity: {sigma:.4e} S/cm")

# Save result to JSON (wrapper already imported json)
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
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")
