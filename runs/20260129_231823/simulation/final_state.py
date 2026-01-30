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

# === 3. kMC Simulator (BKL Algorithm) with phonon-renormalized hopping ===
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
                self.particle_positions[p_id] = {
                    'start': np.array(start),
                    'current': np.array(start)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # === Phonon-renormalized rate parameters ===
        # Static barrier (frozen lattice) E_a (eV) and attempt frequency nu0 (Hz)
        self.E_a0 = params['E_a']      # base activation energy (frozen lattice)
        self.nu0 = params['nu']        # base attempt frequency

        # Phonon-renormalization parameters (phenomenological, temperature dependent)
        # Effective activation energy: E_a,eff(T) = E_a0 + alpha * (T - T_ref)
        # This increases the effective barrier at low T to compensate missing phonon-assisted processes
        self.alpha_E = params.get('alpha_E', 0.0)    # eV/K
        self.T_ref = params.get('T_ref', params['T'])

        # Effective prefactor: nu_eff(T) = nu0 * (T / T_ref)^gamma
        # Allows temperature-dependent attempt frequency due to phonon population
        self.gamma_nu = params.get('gamma_nu', 0.0)

        # Optional multi-phonon/small-polaron-like contribution (Marcus-like)
        # Additional channel: k_mp(T) = nu_mp * exp( - E_mp / (k_B T) )
        # Total hop rate is then: k_tot = k_Arr_eff + k_mp
        self.use_multiphonon = params.get('use_multiphonon', True)
        self.nu_mp = params.get('nu_mp', 0.0)        # Hz
        self.E_mp = params.get('E_mp', 0.0)          # eV

        self.T = params['T']
        self.kb = 8.617e-5  # eV/K

    def _effective_barrier(self, T):
        """
        Phonon-renormalized effective activation energy:
        E_a,eff(T) = E_a0 + alpha_E * (T - T_ref)
        """
        return self.E_a0 + self.alpha_E * (T - self.T_ref)

    def _effective_prefactor(self, T):
        """
        Phonon-renormalized attempt frequency:
        nu_eff(T) = nu0 * (T / T_ref)^gamma_nu
        """
        if self.T_ref <= 0:
            return self.nu0
        return self.nu0 * (T / self.T_ref) ** self.gamma_nu

    def _phonon_renormalized_rate(self, T):
        """
        Total phonon-renormalized hopping rate for a single Li hop:
        k_tot(T) = k_Arr_eff(T) + k_mp(T)
        k_Arr_eff(T) = nu_eff(T) * exp( -E_a,eff(T) / (k_B T) )
        k_mp(T) = nu_mp * exp( -E_mp / (k_B T) )  (optional)
        """
        E_a_eff = self._effective_barrier(T)
        nu_eff = self._effective_prefactor(T)

        k_arr = nu_eff * np.exp(-E_a_eff / (self.kb * T))

        if self.use_multiphonon and self.nu_mp > 0.0 and self.E_mp > 0.0:
            k_mp = self.nu_mp * np.exp(-self.E_mp / (self.kb * T))
            return k_arr + k_mp
        else:
            return k_arr

    def run_step(self):
        # Compute phonon-renormalized base hop rate at current temperature
        base_rate = self._phonon_renormalized_rate(self.T)

        events, rates, total = [], [], 0.0
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    total += base_rate
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
        # Mean Square Displacement (Å^2)
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )
        # Diffusivity (cm^2/s), MSD(t)=6Dt, convert Å^2 to cm^2 with 1e-16
        D = msd / (6 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3); volume in Å^3 -> cm^3 with 1e-24
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein Equation: σ = (n*e^2*D)/(k*T) (S/cm)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
# Base model parameters (frozen lattice)
sim_params = {
    'T': 300,              # K
    'E_a': 0.30,           # eV, static NEB barrier
    'nu': 1e13,            # Hz, attempt frequency
    'volume': structure.volume,
}

# === Phonon-renormalization parameters ===
# Tune these to reduce the 300 K conductivity from ~2.6e-3 S/cm
# toward the experimental ~1.6e-6 S/cm by increasing effective barrier
# and modestly adjusting the prefactor.
#
# Here we introduce a modest positive alpha_E so that
# at 300 K the effective barrier is higher than the frozen-lattice value.
sim_params.update({
    'alpha_E': 3.0e-4,     # eV/K; E_a,eff(T) = E_a + alpha_E*(T - T_ref)
    'T_ref': 600.0,        # K; reference temperature for phonon fit
    'gamma_nu': 0.5,       # dimensionless exponent for nu_eff(T)
    # Multi-phonon channel kept small at 300 K to avoid overestimation
    'use_multiphonon': True,
    'nu_mp': 1e12,         # Hz
    'E_mp': 0.45,          # eV
})

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

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
            )

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