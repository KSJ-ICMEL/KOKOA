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
li_global_indices = []  # mapping from Li-site index -> global structure index
for g_idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_global_indices.append(g_idx)

num_li_sites = len(initial_sites)
print(f"Li sites initialized: {num_li_sites}")

# Build reverse mapping: global index -> Li index (or -1)
global_to_li = {g_idx: li_idx for li_idx, g_idx in enumerate(li_global_indices)}

# === 2. Build Adjacency Graph on Li Sublattice ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {li_idx: [] for li_idx in range(num_li_sites)}

for li_idx, g_idx in enumerate(li_global_indices):
    site = structure[g_idx]
    for nb in neighbors_data[g_idx]:
        nb_g_idx = nb.index
        if nb_g_idx in global_to_li:
            tgt_li_idx = global_to_li[nb_g_idx]
            frac_diff = structure[nb_g_idx].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adj_list[li_idx].append((tgt_li_idx, cart_disp))

print(f"Li-sublattice graph built (cutoff={cutoff}A)")

# === 2b. Precompute static local environment metrics and hop distances ===
# Local environment metric: average Li-Li distance within r_env (proxy for local openness/strain)
r_env = 3.5  # Angstrom
li_coords_cart = np.array(
    [structure.lattice.get_cartesian_coords(initial_sites[i]['coords']) for i in range(num_li_sites)]
)

env_metric = np.zeros(num_li_sites, dtype=float)
for i in range(num_li_sites):
    dists = []
    for j in range(num_li_sites):
        if i == j:
            continue
        disp = li_coords_cart[j] - li_coords_cart[i]
        d = np.linalg.norm(disp)
        if d <= r_env:
            dists.append(d)
    if dists:
        env_metric[i] = np.mean(dists)
    else:
        env_metric[i] = r_env  # if isolated, assign max openness

# Normalize environment metric to [0, 1]
e_min, e_max = env_metric.min(), env_metric.max()
if e_max > e_min:
    env_norm = (env_metric - e_min) / (e_max - e_min)
else:
    env_norm = np.zeros_like(env_metric)

# Precompute hop distances and normalize
hop_dist = {}
all_dists = []
for i in range(num_li_sites):
    for j, vec in adj_list[i]:
        d = np.linalg.norm(vec)
        hop_dist[(i, j)] = d
        all_dists.append(d)

if all_dists:
    d_min, d_max = min(all_dists), max(all_dists)
else:
    d_min, d_max = 0.0, 1.0

def norm_dist(d):
    if d_max > d_min:
        return (d - d_min) / (d_max - d_min)
    return 0.0

# === 3. kMC Simulator (BKL Algorithm) with phonon-assisted barrier modulation ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list

        # Occupancy on Li sites
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Particle tracking
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start, dtype=float),
                    'current': np.array(start, dtype=float),
                }
                p_id += 1

        self.li_indices = set(range(len(initial_sites)))
        self.num_particles = len(self.particle_positions)
        self.current_time = 0.0
        self.step_count = 0

        # Phonon-modulated attempt frequency and effective barriers
        self.kb = 8.617e-5  # eV/K
        self._precompute_temperature_dependent_params()

    def _precompute_temperature_dependent_params(self):
        """
        Incorporate phonon-assisted corrections:
        - Temperature-dependent effective attempt frequency nu_eff(T)
        - Temperature-dependent barrier softening via vibrational free energy
        Model:
          nu_eff(T) = nu0 * (T / T_ref)^alpha
          E_a,eff(T) = E_a0 - lambda * k_B * T
        This stays within known Arrhenius-like corrections and avoids non-physical forms.
        """
        T = self.params['T']
        nu0 = self.params['nu']
        E_a0 = self.params['E_a']  # base 0 K barrier (eV)

        T_ref = 300.0
        alpha = 0.5  # mild increase of prefactor with T
        self.nu_eff = nu0 * (T / T_ref) ** alpha

        # Vibrational free-energy correction to the barrier:
        # E_a,eff(T) = E_a0 - lambda * k_B * T, with small lambda to avoid over-softening
        lambda_ph = 1.5  # dimensionless; moderate phonon assistance
        self.Ea_eff_base = max(E_a0 - lambda_ph * self.kb * T, 0.05)  # floor to keep barrier positive

        # For heterogeneous barriers we will not use a single base_rate; we keep these for reference.
        self.base_rate_ref = self.nu_eff * np.exp(-self.Ea_eff_base / (self.kb * T))

        # Parameters for heterogeneous, environment-dependent barriers
        # Constrained to a realistic LLZO-like range, following NEB-informed values:
        # cubic LLZO ~0.34 eV, tetragonal ~0.6–0.7 eV; we keep a narrow window around target.
        self.Ea_min = 0.25  # eV, lower bound for phonon-softened favorable hops
        self.Ea_max = 0.70  # eV, upper bound for crowded/unfavorable local environments

        # Scales for distance and asymmetry contributions (dimensionless)
        self.dist_scale = 0.25
        self.asym_scale = 0.20

    def _compute_Ea_ij(self, i, j):
        """
        Compute heterogeneous, phonon-assisted activation barrier for hop i -> j:
          E_a,ij(T) = clamp( E_a,eff_base
                             + dist_scale * (d_ij_norm - 0.5)
                             + asym_scale * |e_i - e_j| ,
                             [Ea_min, Ea_max] )
        where:
          - d_ij_norm : normalized hop distance (proxy for bottleneck width)
          - e_i, e_j  : normalized local environment metrics (proxy for lattice openness/strain)
        """
        # geometric contributions
        d = hop_dist.get((i, j), d_min)
        d_norm = norm_dist(d)

        e_i = env_norm[i]
        e_j = env_norm[j]
        asym = abs(e_i - e_j)

        Ea = self.Ea_eff_base \
             + self.dist_scale * (d_norm - 0.5) \
             + self.asym_scale * asym

        # clamp to physically reasonable range
        if Ea < self.Ea_min:
            Ea = self.Ea_min
        elif Ea > self.Ea_max:
            Ea = self.Ea_max

        return Ea

    def run_step(self):
        T = self.params['T']
        kbT = self.kb * T

        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Enumerate all possible hops and compute phonon-modulated, hop-specific rates
        for src in self.li_indices:
            if self.occupancy[src] == 0:
                continue
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    Ea_ij = self._compute_Ea_ij(src, tgt)
                    rate_ij = self.nu_eff * np.exp(-Ea_ij / kbT)
                    if rate_ij <= 0.0:
                        continue
                    total_rate += rate_ij
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0 or not events:
            return False  # Deadlock or no available moves

        # BKL time advance
        dt = -np.log(np.random.rand()) / total_rate
        self.current_time += dt
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, vec = events[idx]

        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id

        return True

    def calculate_properties(self):
        if self.current_time == 0.0 or self.num_particles == 0:
            return 0.0, 0.0
        # Mean-square displacement in Å^2
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )
        # Diffusivity: MSD(t) = 6 D t in 3D, convert Å^2 -> cm^2 (1 Å^2 = 1e-16 cm^2)
        D = msd / (6.0 * self.current_time) * 1e-16

        # Ion concentration n (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)

        # Nernst-Einstein conductivity: sigma = n e^2 D / (k_B T)
        e_charge = 1.602e-19  # C
        k_B_SI = 1.380649e-23  # J/K
        sigma = (n * e_charge ** 2 * D) / (k_B_SI * self.params['T'])

        return msd, sigma

# === 4. Run Simulation ===
# Use a slightly higher static barrier, but allow phonons to soften favorable hops.
sim_params = {
    'T': 300,
    'E_a': 0.45,            # 0 K reference barrier (eV), ~LLZO cubic long-range value
    'nu': 1e13,             # base attempt frequency (Hz)
    'volume': structure.volume
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

        # Keep last 1000 conductivity samples
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0.0

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
D = msd / (6.0 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

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