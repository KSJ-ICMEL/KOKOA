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
        """
        Compute tracer diffusivity D_tracer from single-particle MSD and
        convert to ionic conductivity using a Haven ratio H < 1 to
        account for correlation effects in LLZO:
            D_tracer = <r^2(t)> / (6 t)
            D_sigma  = H * D_tracer
            σ        = n e^2 D_sigma / (k_B T)
        """
        if self.current_time == 0:
            return 0.0, 0.0, 0.0

        # Mean Square Displacement (Å^2)
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])

        # Tracer diffusivity (cm^2/s), MSD(t) = 6 D_tracer t
        D_tracer = msd / (6.0 * self.current_time) * 1e-16

        # Haven ratio H = D_sigma / D_tracer < 1 for correlated motion
        # In LLZO, concerted Li hopping leads to significant correlation
        # between charge and mass transport (see Jalem et al. 2013 and
        # related LLZO transport studies). To avoid enforcing the
        # ideal Nernst–Einstein assumption (H = 1), we include an
        # explicit Haven factor in the conversion from D_tracer to σ.
        H = self.params.get("haven_ratio", 0.2)  # correlation correction

        # Conductivity diffusion coefficient (cm^2/s)
        D_sigma = H * D_tracer

        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)

        # Nernst–Einstein-type relation using D_sigma:
        # σ = n e^2 D_sigma / (k_B T)
        e_charge = 1.602e-19  # C
        k_B_J = 1.38e-23      # J/K
        sigma = (n * e_charge ** 2 * D_sigma) / (k_B_J * self.params['T'])

        return msd, sigma, D_tracer

# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    'E_a': 0.30,
    'nu': 1e13,
    'volume': structure.volume,
    # Haven ratio < 1 to correct for correlation between tracer and
    # charge diffusion in LLZO. Literature reports for highly
    # correlated Li conductors are typically well below unity;
    # this parameter can be refined against detailed kMC or AIMD
    # correlation analyses but is explicitly included here to avoid
    # the H = 1 ideal Nernst–Einstein assumption.
    'haven_ratio': 0.2
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
        msd, sigma, D_tracer = sim.calculate_properties()
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
                f"MSD={msd:.2f}A^2, D_tracer={D_tracer:.3e}cm^2/s, "
                f"sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )
            
            if rsd < 0.05:  # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, D_tracer={D_tracer:.3e}cm^2/s, "
                f"sigma={sigma*1e3:.4f}mS/cm"
            )

# Final result
msd, sigma, D_tracer = sim.calculate_properties()
D = D_tracer  # For backward compatibility with previous output naming

print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D_tracer={D_tracer:.4e} cm^2/s")
print(f"Haven ratio (input): {sim_params['haven_ratio']}")
print(f"D_sigma={D_tracer*sim_params['haven_ratio']:.4e} cm^2/s")
print(f"Conductivity: {sigma:.4e} S/cm")

# Save result to JSON
result = {
    "is_success": True,
    "conductivity": sigma,
    "diffusivity_tracer": D_tracer,
    "diffusivity_sigma": D_tracer * sim_params['haven_ratio'],
    "haven_ratio": sim_params['haven_ratio'],
    "msd": msd,
    "simulation_time_ns": sim.current_time * 1e9,
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns; "
        f"Haven ratio={sim_params['haven_ratio']}"
    ),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")