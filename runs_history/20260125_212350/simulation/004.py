"""KOKOA Simulation #4 - 2026-01-25 21:30:01"""
import os, sys, json
import numpy as np
from pymatgen.core import Structure

# Pre-loaded structure with supercell
structure = Structure.from_file("C:/Users/sjkim/KOKOA/LLZO.cif")
N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

# 1. Identify initial Li occupancy
initial_sites = []
for site in structure:
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

# 2. Build adjacency list using crystallographic neighbors (24d‑96h)
cutoff = 3.0  # Å, captures true Li‑Li hops without periodic images
adj_list = {}
for i, site in enumerate(structure):
    neighbors = structure.get_neighbors(site, cutoff)
    entries = []
    for neighbor in neighbors:
        # consider only Li sites and only the primary image
        if "Li" not in neighbor.species.elements:
            continue
        if neighbor.image != (0, 0, 0):
            continue
        j = neighbor.index
        vec = neighbor.coords - site.coords
        entries.append((j, vec))
    if entries:
        adj_list[i] = entries

# 3. KMC Simulator class with barrier lookup
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

        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.T = params['T']
        self.repulsion_coeff = params.get('repulsion_coeff', 0.10)

        # Pre‑compute hop‑specific barriers from geometric distance (proxy for bottleneck size)
        self.barrier_lookup = {}
        for src, edges in self.adj_list.items():
            for tgt, vec in edges:
                distance = np.linalg.norm(vec)  # Å
                # Simple model: baseline 0.40 eV, increase 0.05 eV per Å beyond 2 Å
                E_barrier = 0.40 + max(0.0, distance - 2.0) * 0.05
                self.barrier_lookup[(src, tgt)] = E_barrier

    def _hop_rate(self, src, tgt):
        E_base = self.barrier_lookup.get((src, tgt), self.params.get('E_a', 0.45))
        # repulsive contribution from occupied neighbors around target
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
        events = []
        cumulative = 0.0
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if tgt < 0 or tgt >= len(self.occupancy):
                    raise IndexError(f"Target index {tgt} out of bounds")
                if self.occupancy[tgt] == 0:
                    rate = self._hop_rate(src, tgt)
                    if rate <= 0:
                        continue
                    cumulative += rate
                    events.append((src, tgt, vec, cumulative))
        if cumulative == 0.0:
            return False
        self.current_time += -np.log(np.random.rand()) / cumulative
        self.step_count += 1
        r = np.random.rand() * cumulative
        for src, tgt, vec, cum_rate in events:
            if r <= cum_rate:
                break
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
        msd = np.mean([np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()])
        D = msd / (6 * self.current_time) * 1e-16
        n = self.num_particles / (self.params['volume'] * 1e-24)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# 4. Simulation parameters (unchanged)
sim_params = {
    'T': 298,
    'nu': 1e13,
    'volume': structure.volume,
    'E_a': 0.45,
    'repulsion_coeff': 0.10
}

# 5. Run the KMC simulation (unchanged logic)
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

target_time = 1000e-9
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

msd, sigma = sim.calculate_properties()
D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0
print("\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D={D:.4e} cm^2/s")
print(f"Conductivity: {sigma:.4e} S/cm")
