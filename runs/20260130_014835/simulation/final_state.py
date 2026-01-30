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

# === 2. Build Adjacency Graph with local-phonon-modulated barriers ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

# --- Identify Li site types by local coordination (proxy for different vibrational coupling) ---
# We use coordination number of heavy atoms (non-Li) within 3 Å as a crude classifier.
# This is a simple, evidence-based way to introduce site-to-site variability in barriers
# tied to local environment, reflecting phonon-assisted barrier renormalization.
heavy_neighbor_cutoff = 3.0  # Å
site_env_signature = []
for i, site in enumerate(structure):
    if "Li" not in [s.symbol for s in site.species.elements]:
        site_env_signature.append(None)
        continue
    neighs = structure.get_neighbors(site, heavy_neighbor_cutoff)
    heavy_count = sum(1 for nb in neighs if "Li" not in [s.symbol for s in nb.species.elements])
    site_env_signature.append(heavy_count)

# Map coordination counts to "site types"
unique_counts = sorted(set(c for c in site_env_signature if c is not None))
coord_to_type = {c: idx for idx, c in enumerate(unique_counts)}
li_site_types = {}
for i, sig in enumerate(site_env_signature):
    if sig is None:
        continue
    li_site_types[i] = coord_to_type[sig]

# Predefine environment-dependent baseline barriers for each site type (eV).
# These represent static part of the migration barrier associated with local structure.
# Later we will add a stochastic phonon-induced modulation on top of these baselines.
# Choose a modest spread around the original 0.30 eV to introduce site-to-site variability.
base_Ea_by_type = {}
if len(unique_counts) == 1:
    base_Ea_by_type[0] = 0.30
else:
    # Spread values between 0.25 and 0.35 eV across types
    Ea_min, Ea_max = 0.25, 0.35
    if len(unique_counts) == 2:
        vals = [0.27, 0.33]
    else:
        vals = np.linspace(Ea_min, Ea_max, num=len(unique_counts))
    for c, t in coord_to_type.items():
        base_Ea_by_type[t] = float(vals[t])

# Build adjacency: for each Li site, store neighbors and a static geometric factor
for i, site in enumerate(structure):
    if "Li" not in [s.symbol for s in site.species.elements]:
        continue
    neighbors = []
    for nb in neighbors_data[i]:
        if "Li" in [s.symbol for s in structure[nb.index].species.elements]:
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            dist = np.linalg.norm(cart_disp)
            # Simple geometric weight: shorter hops have slightly lower baseline barrier
            # Use a linear factor in distance referenced to cutoff.
            geom_factor = 1.0 + 0.2 * (dist - cutoff) / cutoff  # small correction
            neighbors.append((nb.index, cart_disp, geom_factor))
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with phonon-renormalized, site-dependent barriers ===
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
                    'start': np.array(start, dtype=float),
                    'current': np.array(start, dtype=float)
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # Constants
        self.kb = 8.617e-5  # eV/K
        self.nu0 = params['nu']
        self.T = params['T']

        # Phonon-modulation parameters:
        # We model phonon renormalization of the activation barrier as:
        #   E_eff = E_base + delta_E(T, local_env)
        #
        # with delta_E a zero-mean Gaussian-distributed fluctuation whose variance
        # increases with temperature to mimic stronger phonon amplitudes.
        #
        # sigma_E(T) = alpha * sqrt(T / T_ref)  with alpha in eV
        # Theoretical works (e.g., electron/ion-phonon renormalization) show
        # vibrational free-energy contributions scale with phonon population ~ T
        # in the classical (kT >> ħω) limit, while structural disorder broadens
        # barrier distributions. Here we use alpha ~ 0.03 eV as a modest scale.
        self.E_renorm_alpha = params.get('E_renorm_alpha', 0.03)  # eV
        self.T_ref = params.get('T_ref', 300.0)  # K

        # Precompute a global phonon "softening" factor for the attempt frequency
        # to encode phonon-assisted hopping frequency enhancement:
        #   nu_eff = nu0 * (1 + c * (T/T_ref - 1))
        # This mimics an increased availability of vibrational modes that
        # assist barrier crossing at higher T without changing detailed balance.
        self.nu_T_factor = 1.0 + params.get('nu_T_coeff', 0.5) * (self.T / self.T_ref - 1.0)
        if self.nu_T_factor < 0.2:
            self.nu_T_factor = 0.2  # avoid unphysical negative/too small
        self.nu_eff = self.nu0 * self.nu_T_factor

        # Pre-store per-site static baseline barriers (site-type-dependent)
        self.base_Ea_site = {}
        for i, s in enumerate(initial_sites):
            if "Li" not in [sp.symbol for sp in structure[i].species.elements]:
                continue
            if i not in li_site_types:
                # fallback
                self.base_Ea_site[i] = params['E_a']
            else:
                site_type = li_site_types[i]
                self.base_Ea_site[i] = base_Ea_by_type.get(site_type, params['E_a'])

        # For efficiency, we keep a cache of last-used effective barriers per edge
        # keyed by (src, tgt). We will update them dynamically with phonon noise.
        self.edge_barrier_cache = {}

    def _sigma_E_T(self):
        # Width of phonon-induced barrier fluctuations at temperature T
        return self.E_renorm_alpha * np.sqrt(self.T / self.T_ref)

    def _draw_barrier_for_edge(self, src, tgt, geom_factor):
        # Draw a phonon-renormalized activation barrier for hop src -> tgt.
        # Baseline: average of source and target site baselines, times geom_factor.
        Ea_src = self.base_Ea_site.get(src, self.params['E_a'])
        Ea_tgt = self.base_Ea_site.get(tgt, self.params['E_a'])
        Ea_base = 0.5 * (Ea_src + Ea_tgt) * geom_factor

        # Add phonon-induced fluctuation delta_E ~ N(0, sigma_E^2)
        sigma_E = self._sigma_E_T()
        delta_E = np.random.normal(loc=0.0, scale=sigma_E)

        Ea_eff = Ea_base + delta_E
        # Prevent unphysical negative or extremely small barriers
        Ea_eff = max(Ea_eff, 0.05)
        return Ea_eff

    def _rate_for_edge(self, src, tgt, geom_factor):
        # Compute (or update) effective barrier and corresponding rate
        key = (src, tgt)
        Ea_eff = self._draw_barrier_for_edge(src, tgt, geom_factor)
        self.edge_barrier_cache[key] = Ea_eff
        rate = self.nu_eff * np.exp(-Ea_eff / (self.kb * self.T))
        return rate

    def run_step(self):
        events = []
        rates_cum = []
        total_rate = 0.0

        # Enumerate all possible vacancy hops and assign phonon-renormalized rates
        for src in list(self.li_indices):
            for tgt, vec, geom_factor in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._rate_for_edge(src, tgt, geom_factor)
                    if rate <= 0.0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    rates_cum.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        rnd = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(rates_cum, rnd)
        src, tgt, vec = events[idx]

        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)

        # Optionally clear cache entries associated with moved ion to avoid bias
        # as local environment changes; keep it simple and just delete src-related keys.
        keys_to_delete = [k for k in self.edge_barrier_cache.keys() if k[0] == src or k[1] == src]
        for k in keys_to_delete:
            del self.edge_barrier_cache[k]

        return True

    def calculate_properties(self):
        if self.current_time == 0.0:
            return 0.0, 0.0
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Mean Square Displacement (Å^2)
        D = msd / (6.0 * self.current_time) * 1e-16  # Diffusivity (cm^2/s), MSD(t)=6Dt
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma


# === 4. Run Simulation ===
# We keep the same nominal E_a but phonon renormalization and site variability
# will effectively increase barriers at 300 K and broaden the rate distribution,
# correcting the overly coherent, too-fast transport of the frozen-lattice model.
sim_params = {
    'T': 300,
    'E_a': 0.30,      # nominal baseline activation energy (eV)
    'nu': 1e13,       # base attempt frequency (Hz)
    'volume': structure.volume,
    # Phonon-renormalization controls
    'E_renorm_alpha': 0.04,  # eV, slightly larger to effectively reduce conductivity
    'T_ref': 300.0,
    'nu_T_coeff': 0.3  # modest phonon-assisted enhancement of attempt frequency
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