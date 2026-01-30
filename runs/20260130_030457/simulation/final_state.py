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

# === 3. kMC Simulator with non-Poissonian waiting times ===
class KMCSimulator:
    """
    Kinetic Monte Carlo simulator with site-disordered, configuration-dependent,
    non-exponential waiting times implemented via a continuous-time random walk (CTRW)
    / subordination scheme.

    Physics-motivated modifications (within provided context):
      - Each Li site has a static local "trap" energy U[Li-m] drawn from a distribution.
      - The local rate is configuration-dependent: r_i = nu * exp(-E_a / (k_B T)) * exp(-U_i / (k_B T)).
      - To model broad waiting time distributions from disorder and correlated dynamics,
        we use a Weibull (stretched-exponential) waiting time distribution for each hop:
            ψ_i(t) = (β / τ_i) (t / τ_i)^{β-1} exp[ -(t/τ_i)^β ]
        where τ_i = 1 / r_i and 0 < β <= 1.
        β < 1 produces broad, non-Poissonian waiting times (subdiffusive CTRW-like).
      - Global time is advanced by the *smallest* of all scheduled hop times (next-event
        CTRW), rather than a single exponential with total rate as in standard BKL.

    This directly relaxes the homogeneous Poisson assumption A7.
    """
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.structure = structure
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map sites to particles and track particle trajectories
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start, dtype=float),
                    'current': np.array(start, dtype=float)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # Physical constants
        self.kb = 8.617e-5  # eV/K

        # Base attempt frequency and activation barrier
        self.nu0 = params['nu']
        self.Ea = params['E_a']  # eV
        self.T = params['T']

        # Exponent β for Weibull waiting time distribution (β=1 => exponential/Poisson)
        # To introduce broad waiting-time distributions (subdiffusion), set β < 1.
        self.beta = params.get('beta_weibull', 0.6)

        # Static site disorder energies U[Li-m] (trap depths), e.g. Gaussian with std dev
        # U has units of eV; positive U slows down hopping (deeper trap).
        disorder_std = params.get('U_disorder_std', 0.05)  # eV
        rng = np.random.default_rng(params.get('seed', None))
        self.site_energy = rng.normal(loc=0.0, scale=disorder_std, size=len(initial_sites))

        # Event structure:
        # For each occupied site i, maintain a scheduled hop event:
        #   - next_event_time[i]: absolute time of next hop from site i
        #   - next_event_target[i]: target site index j
        #   - next_event_vec[i]: displacement vector (cartesian) for that hop
        self.next_event_time = {}
        self.next_event_target = {}
        self.next_event_vec = {}
        self.rng = rng

        # Initialize events for all currently occupied sites
        for i in self.li_indices:
            self._schedule_next_event_for_site(i)

    def _local_rate(self, site_idx):
        """
        Compute configuration-dependent local rate for a hop from site_idx:
            r_i = nu0 * exp( - (E_a + U_i) / (k_B T) )
        where U_i is the site disorder energy U[Li-m].
        """
        U_i = self.site_energy[site_idx]  # eV
        return self.nu0 * np.exp(-(self.Ea + U_i) / (self.kb * self.T))

    def _sample_weibull_waiting_time(self, rate):
        """
        Sample a waiting time from a Weibull distribution with scale τ = 1/rate
        and exponent β:
            F(t) = 1 - exp[ - (t/τ)^β ]
        Using inverse transform:
            t = τ * [ -ln(ξ) ]^(1/β)
        """
        if rate <= 0:
            return np.inf
        tau = 1.0 / rate
        xi = self.rng.random()
        return tau * (-np.log(xi)) ** (1.0 / self.beta)

    def _schedule_next_event_for_site(self, src):
        """
        For a given occupied site src, choose a target and schedule a hop time
        based on the local rate and a non-exponential waiting time distribution.
        If no available neighbor, schedule at infinity (effectively frozen).
        """
        neighbors = self.adj_list.get(src, [])
        free_neighbors = [(tgt, vec) for (tgt, vec) in neighbors if self.occupancy[tgt] == 0]

        if not free_neighbors:
            # No possible hop; set event to never happen
            self.next_event_time[src] = np.inf
            self.next_event_target[src] = None
            self.next_event_vec[src] = None
            return

        # Simple choice: randomly select one of the available neighbors
        tgt, vec = free_neighbors[self.rng.integers(len(free_neighbors))]

        # Local rate including site energy disorder
        r_loc = self._local_rate(src)

        # Sample non-Poissonian waiting time for this event
        dt = self._sample_weibull_waiting_time(r_loc)
        event_time = self.current_time + dt

        self.next_event_time[src] = event_time
        self.next_event_target[src] = tgt
        self.next_event_vec[src] = vec

    def run_step(self):
        """
        Continuous-time random-walk kMC step:
          - Find event with smallest next_event_time among all occupied sites.
          - Advance global time to that event time.
          - Execute hop (src -> tgt) if still valid.
          - Reschedule events for src (now empty) and tgt (now occupied).
        This replaces the standard BKL global exponential time advance and
        thus relaxes the homogeneous Poisson assumption.
        """
        if not self.li_indices:
            return False

        # Find site with earliest scheduled event
        # (this is O(N) but acceptable for moderate system sizes)
        min_time = np.inf
        min_site = None
        for i in self.li_indices:
            t_evt = self.next_event_time.get(i, np.inf)
            if t_evt < min_time:
                min_time = t_evt
                min_site = i

        if min_site is None or np.isinf(min_time):
            # No more events can occur (blocked configuration)
            return False

        # Advance global time to the next event
        self.current_time = min_time
        self.step_count += 1

        src = min_site
        tgt = self.next_event_target.get(src)
        vec = self.next_event_vec.get(src)

        # If event is invalid (e.g. tgt was taken due to another event, though
        # our scheme avoids competing events from same site), check and handle.
        if tgt is None or self.occupancy[src] == 0 or self.occupancy[tgt] == 1:
            # Reschedule event for this site based on current configuration
            self._schedule_next_event_for_site(src)
            return True

        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)

        # After hop, reschedule events for:
        #   - src: now empty, should have no event
        self.next_event_time[src] = np.inf
        self.next_event_target[src] = None
        self.next_event_vec[src] = None

        #   - tgt: now occupied, needs a new scheduled hop
        self._schedule_next_event_for_site(tgt)

        # Also, neighbors whose availability changed may have to be updated.
        # To capture configuration dependence of rates, we reschedule any
        # occupied neighbor events that involved src or tgt as possible targets.
        for nb_idx, _ in self.adj_list.get(src, []):
            if self.occupancy[nb_idx] == 1:
                self._schedule_next_event_for_site(nb_idx)
        for nb_idx, _ in self.adj_list.get(tgt, []):
            if self.occupancy[nb_idx] == 1:
                self._schedule_next_event_for_site(nb_idx)

        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )  # Mean Square Displacement (Å^2)
        # MSD(t) = 6 D t  (Einstein relation)
        D = msd / (6.0 * self.current_time) * 1e-16  # Diffusivity (cm^2/s)
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
        # Nernst-Einstein: σ = (n e^2 D) / (k_B T)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma


# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    'E_a': 0.30,           # eV
    'nu': 1e13,            # s^-1
    'volume': structure.volume,
    # Parameters controlling non-Poissonian waiting times and disorder:
    'beta_weibull': 0.6,   # 0<β<=1; β<1 -> broad waiting times (trapping, subdiffusion)
    'U_disorder_std': 0.05,  # eV, width of site energy disorder U[Li-m]
    'seed': 42,
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
D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

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