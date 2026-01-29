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

# === 3. kMC Simulator (BKL Algorithm) with phonon-assisted hopping ===
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
        
        # Constants
        self.kb = 8.617e-5  # eV/K
        
        # Store base attempt frequency; barrier is now phonon-renormalized per hop
        self.nu0 = params['nu']
        
        # Precompute phonon-related factors based on Debye frequency and coupling
        # Inspired by Fig. 12F: linear relation between Debye frequency and activation barrier.
        # We introduce a configuration-averaged phonon factor that reduces the effective rate.
        self.debye_freq = params.get('omega_D', 1.0e13)  # s^-1, rough scale of phonon frequencies
        self.phonon_coupling = params.get('phonon_coupling', 1.0)  # dimensionless, >1 strengthens reduction
        self._setup_phonon_assisted_rate_parameters()

    def _setup_phonon_assisted_rate_parameters(self):
        """
        Encode a simple phonon-assisted hopping correction consistent with:
        - Lattice softness lowers Debye frequency and activation barrier.
        - Only a subset of vibrational configurations are conducive to hopping.
        
        We keep a static "bare" barrier E_a and renormalize it by lattice vibrations via:
            E_eff = E_a + alpha * (ħ*omega_D)
        and reduce the effective prefactor via a phonon-configuration factor f_vib(T) < 1:
            f_vib(T) = exp(-lambda_ph * (ħ*omega_D) / (k_B T))
        
        This captures that phonon reorganization costs energy and only some phonon modes assist hops.
        """
        kb = self.kb
        T = self.params['T']
        E_a = self.params['E_a']
        
        # Convert Debye frequency to energy scale: E_D = ħ * ω_D (in eV)
        hbar = 6.582119569e-16  # eV*s
        E_D = hbar * self.debye_freq
        
        # Phonon-induced barrier renormalization factor alpha:
        # Stronger coupling -> more reorganization energy -> higher apparent barrier.
        alpha = 0.5 * self.phonon_coupling  # moderate linear scaling with coupling
        self.E_renorm = E_a + alpha * E_D
        
        # Effective fraction of vibrational configurations that enable a hop.
        # This is an exponential suppression factor based on a phonon reorganization energy ~ E_D.
        lambda_ph = self.phonon_coupling
        self.f_vib = np.exp(-lambda_ph * E_D / (kb * T))
        
        # Store an effective, phonon-renormalized base rate used as a reference.
        self.base_rate_eff = self.nu0 * self.f_vib * np.exp(-self.E_renorm / (kb * T))

    def _compute_hop_rate(self, src, tgt, vec):
        """
        Configuration-dependent phonon-assisted rate for a hop.
        
        We allow mild dependence of the renormalized barrier on local Li configuration
        by modulating E_renorm based on the number of Li neighbors at the source.
        This reflects that nearby mobile ions and local lattice distortion alter the
        energy landscape (Fig. 12A, 12E, 12G, 12H).
        """
        kb = self.kb
        T = self.params['T']
        
        # Count Li neighbors around source site to mimic local distortion / phonon environment.
        li_neighbor_count = 0
        for nb_idx, _ in self.adj_list.get(src, []):
            if self.occupancy[nb_idx] == 1:
                li_neighbor_count += 1
        
        # Local barrier modulation: more Li neighbors -> more repulsion and larger reorganization,
        # hence a slightly higher effective barrier.
        # Keep this modest to avoid unphysical changes; 0.01 eV per Li neighbor as a scale.
        delta_E_local = 0.01 * li_neighbor_count  # eV
        
        E_loc = self.E_renorm + delta_E_local
        
        # Use the same phonon configuration factor f_vib; it depends only on T and ω_D here.
        rate = self.nu0 * self.f_vib * np.exp(-E_loc / (kb * T))
        return rate

    def run_step(self):
        events, cum_rates = [], []
        total = 0.0
        
        # Build event list with configuration-dependent, phonon-assisted rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._compute_hop_rate(src, tgt, vec)
                    if rate <= 0.0:
                        continue
                    total += rate
                    events.append((src, tgt, vec))
                    cum_rates.append(total)
        
        if total == 0:
            return False  # Deadlock
        
        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1
        
        # Select and execute event
        r = np.random.uniform(0, total)
        idx = np.searchsorted(cum_rates, r)
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
        msd = np.mean(
            [np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()]
        )  # Mean Square Displacement (Å^2)
        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s), MSD(t)=6Dt
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # Nernst-Einstein Equation
        return msd, sigma

# === 4. Run Simulation ===
# Parameters now include Debye frequency and phonon coupling to encode phonon-assisted hopping effects.
sim_params = {
    'T': 300,
    'E_a': 0.30,        # bare migration barrier (eV)
    'nu': 1e13,         # attempt frequency (s^-1)
    'omega_D': 5e13,    # Debye frequency (s^-1), LLZO-scale lattice vibrations
    'phonon_coupling': 2.0,  # dimensionless, tunes strength of phonon reorganization penalty
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
        
        # Check convergence
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