"""KOKOA Simulation #4 - 2026-01-25 20:49:28"""
import os, sys, json
import numpy as np
from pymatgen.core import Structure

# Pre-loaded structure
structure = Structure.from_file("C:/Users/sjkim/KOKOA/LLZO.cif")
print(f"Structure loaded: {len(structure)} atoms")

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
        
        # kinetic parameters
        self.nu = params['nu']          # original attempt frequency (1/s) – kept for backward compatibility
        self.nu_eff = params.get('nu_eff', self.nu)  # vibrational prefactor (~1e13 s⁻¹)
        self.T = params['T']            # temperature (K)
        self.kb = 8.617e-5              # eV/K
        
        # phonon‑assisted barrier reduction parameters
        self.phonon_mean = params.get('phonon_mean', 0.02)   # mean reduction in eV
        self.phonon_std = params.get('phonon_std', 0.01)    # std dev of reduction in eV
        
        # simple barrier database (eV) for distinct hop types
        self.barrier_dict = {
            ("24d", "96h"): 0.35,
            ("96h", "24d"): 0.35,
            ("96h", "96h"): 0.45,
            ("24d", "24d"): 0.45
        }
        
    def get_site_type(self, idx):
        """Placeholder site‑type assignment.
        In a real implementation this would query the crystallographic Wyckoff position.
        Here we use even indices as 24d and odd indices as 96h.
        """
        return "24d" if idx % 2 == 0 else "96h"
    
    def occupied_neighbors(self, idx):
        """Count occupied Li neighbours of a given site (excluding the site itself)."""
        count = 0
        for nb_idx, _ in self.adj_list.get(idx, []):
            if self.occupancy[nb_idx] == 1:
                count += 1
        return count
    
    def get_barrier(self, src, tgt):
        """Return an environment‑dependent activation barrier (eV)."""
        src_type = self.get_site_type(src)
        tgt_type = self.get_site_type(tgt)
        base = self.barrier_dict.get((src_type, tgt_type), 0.45)
        # simple penalty for occupied neighbours (electrostatic repulsion)
        occ_src = self.occupied_neighbors(src)
        occ_tgt = self.occupied_neighbors(tgt)
        penalty = 0.05 * (occ_src + occ_tgt)
        return base + penalty
    
    def run_step(self):
        events = []          # list of (src, tgt, vec, rate)
        cumulative = []      # cumulative rates for BKL selection
        total_rate = 0.0
        
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    Ea_static = self.get_barrier(src, tgt)
                    # phonon‑assisted reduction (Gaussian draw, never negative reduction)
                    delta_E = np.random.normal(self.phonon_mean, self.phonon_std)
                    delta_E = max(delta_E, 0.0)
                    Ea_eff = max(Ea_static - delta_E, 0.0)
                    rate = self.nu_eff * np.exp(-Ea_eff / (self.kb * self.T))
                    if rate <= 0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec, rate))
                    cumulative.append(total_rate)
        
        if total_rate == 0.0:
            return False  # deadlock
        
        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1
        
        # select event
        r = np.random.rand() * total_rate
        idx = np.searchsorted(cumulative, r)
        src, tgt, vec, _ = events[idx]
        
        # execute hop
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
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 298,
    'E_a': 0.30,
    'nu': 1e13,          # original attempt frequency (kept for reference)
    'nu_eff': 1e13,      # vibrational prefactor used after relaxation
    'phonon_mean': 0.02, # average barrier reduction due to phonons (eV)
    'phonon_std': 0.01,  # fluctuation of barrier reduction (eV)
    'volume': structure.volume
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
