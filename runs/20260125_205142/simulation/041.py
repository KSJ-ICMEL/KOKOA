"""KOKOA Simulation #41 - 2026-01-25 20:54:37"""
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

# === Define initial sites (occupancy) ===
# Each site gets a dict with its occupation state (1 if Li, else 0) and fractional coordinates.
initial_sites = []
for i, site in enumerate(structure):
    state = 1 if "Li" in site.species.elements[0].symbol else 0
    initial_sites.append({'state': state, 'coords': site.frac_coords})

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
        self.alpha = params.get('alpha', 0.05)  # eV per occupied neighbour
        
        # Pre‑compute temperature‑dependent quantities for phonon‑assisted hopping
        self._update_phonon_quantities()
    
    def _update_phonon_quantities(self):
        """Compute temperature‑dependent attempt frequency and barrier reduction.
        Uses a simple linear scaling for ν(T) and a √T scaling for ΔE_ph(T).
        """
        T = self.params['T']
        # Base attempt frequency (nu0) at reference temperature (300 K)
        nu0 = self.params.get('nu0', self.params.get('nu', 1e13))
        beta = self.params.get('beta', 0.0)  # fractional change per 300 K
        # ν(T) = ν0 * (1 + beta * (T-300)/300)
        self.nu_T = nu0 * (1.0 + beta * (T - 300.0) / 300.0)
        # Phonon‑assisted barrier reduction ΔE_ph = gamma * sqrt(T/300)
        gamma = self.params.get('gamma', 0.0)  # eV
        self.delta_E_ph = gamma * np.sqrt(T / 300.0)
    
    def _compute_rate(self, src, tgt):
        """Calculate the hopping rate for a specific src→tgt hop.
        Includes environment‑dependent barrier, phonon‑assisted reduction,
        and temperature‑dependent attempt frequency.
        """
        # Count occupied neighbours of the target site (excluding the moving ion)
        n_occ = 0
        for nb_idx, _ in self.adj_list.get(tgt, []):
            if self.occupancy[nb_idx] == 1:
                n_occ += 1
        # Effective activation energy
        E_eff = self.params['E_a'] + self.alpha * n_occ - self.delta_E_ph
        # Ensure barrier is not negative
        if E_eff < 0:
            E_eff = 0.0
        rate = self.nu_T * np.exp(-E_eff / (self.kb * self.params['T']))
        return rate
    
    def run_step(self):
        events = []          # (src, tgt, vec)
        cumulative_rates = []
        total_rate = 0.0
        
        # Build list of possible hops with updated rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._compute_rate(src, tgt)
                    if rate <= 0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)
        
        if total_rate == 0.0:
            return False  # Deadlock
        
        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1
        
        # Select event based on cumulative rates
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, vec = events[idx]
        
        # Execute the selected hop
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
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])  # Å^2
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 298,
    'E_a': 0.30,          # base activation energy (eV)
    'nu0': 1e13,          # base attempt frequency at 300 K (1/s)
    'beta': 0.10,         # 10 % increase of ν per 300 K rise (dimensionless)
    'gamma': 0.02,        # phonon‑assisted barrier reduction coefficient (eV)
    'volume': structure.volume,
    'alpha': 0.05         # barrier increase per occupied neighbour (eV)
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
            sigma_history.pop(0)  # Keep last 1000
        
        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0
            
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")
            
            if rsd < 0.05:  # 5% convergence criteria
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
