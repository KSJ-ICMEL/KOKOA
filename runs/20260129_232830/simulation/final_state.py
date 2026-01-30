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

# === 2. Build Adjacency Graph with Distance-Dependent Barriers ===
# We retain a geometric cutoff, but assign hop-specific activation energies
# based on Li-Li distance using a linear fit to Table 2 (paths A-D).

cutoff = 4.0  # Angstrom, still used only to identify candidate neighbors
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Precompute Li site indices for fast lookup
li_indices = [i for i, site in enumerate(structure)
              if "Li" in [s.symbol for s in site.species.elements]]

# Linear fit of Ea vs Li-Li distance from internal paper (paths A–D):
# (d, Ea): (2.45, 0.44), (2.52, 0.35), (2.58, 0.26), (2.59, 0.45)
# Simple least-squares fit gives approximately:
# Ea(d) [eV] = m * d + b
# We'll compute m, b here explicitly from those four points to keep it transparent.

dist_samples = np.array([2.45, 2.52, 2.58, 2.59])
Ea_samples = np.array([0.44, 0.35, 0.26, 0.45])
A_fit = np.vstack([dist_samples, np.ones_like(dist_samples)]).T
m_fit, b_fit = np.linalg.lstsq(A_fit, Ea_samples, rcond=None)[0]

def activation_energy_from_distance(d):
    """
    Map Li-Li distance to a hop-specific activation energy using the linear fit
    to the DFT data (paths A–D). Clamp to the min/max of the sampled Ea range
    to avoid unphysical extrapolation.
    """
    Ea_raw = m_fit * d + b_fit
    Ea_clamped = float(np.clip(Ea_raw, Ea_samples.min(), Ea_samples.max()))
    return Ea_clamped

# Build adjacency list with hop-specific barriers and distances
adj_list = {}

for i in li_indices:
    site_i = structure[i]
    neighbors = []
    for nb in neighbors_data[i]:
        j = nb.index
        site_j = structure[j]
        if j == i:
            continue
        if "Li" not in [s.symbol for s in site_j.species.elements]:
            continue

        # Cartesian displacement including image
        frac_diff = site_j.frac_coords - site_i.frac_coords + nb.image
        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
        d = np.linalg.norm(cart_disp)

        # Skip self or extremely close numerical artifacts
        if d < 1e-3:
            continue

        # Assign hop-specific activation energy from distance
        Ea_ij = activation_energy_from_distance(d)

        neighbors.append(
            {
                "tgt": j,
                "disp": cart_disp,
                "dist": d,
                "Ea": Ea_ij,
            }
        )
    adj_list[i] = neighbors

print(f"Graph built with distance-dependent barriers (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with Hop-Specific Rates ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map from site index to particle id and track positions
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

        # Precompute k0 = nu(T) prefactor
        self.kb = 8.617e-5  # eV/K
        self.nu0 = params['nu']
        self.T = params['T']

    def hop_rate(self, Ea):
        """
        Arrhenius rate k = nu0 * exp(-Ea / (kB T)) using hop-specific Ea.
        """
        return self.nu0 * np.exp(-Ea / (self.kb * self.T))

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Enumerate all possible vacancy-mediated Li hops
        for src in list(self.li_indices):
            for nb in self.adj_list.get(src, []):
                tgt = nb["tgt"]
                if self.occupancy[tgt] == 0:
                    k_ij = self.hop_rate(nb["Ea"])
                    if k_ij <= 0.0:
                        continue
                    total_rate += k_ij
                    events.append((src, tgt, nb["disp"]))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # No further hops possible

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, disp = events[idx]

        # Move particle
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += disp

        # Update occupancy and mappings
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        # Mean square displacement (Å^2)
        msd = np.mean(
            [
                np.sum((p['current'] - p['start']) ** 2)
                for p in self.particle_positions.values()
            ]
        )
        # Diffusivity (cm^2/s): MSD(t) = 6 D t
        D = msd / (6.0 * self.current_time) * 1e-16
        # Ion concentration (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)
        # Nernst-Einstein Equation: σ = (n * e^2 * D) / (k_B * T)
        e = 1.602e-19  # C
        k_B_SI = 1.38e-23  # J/K
        sigma = (n * e * e * D) / (k_B_SI * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
# Use a representative activation energy scale for parameter inspection only;
# actual hop barriers come from the distance-based mapping above.
sim_params = {
    'T': 300,
    'E_a': 0.45,     # Not used directly for rates now, kept for record
    'nu': 1e13,
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
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, "
                f"RSD={rsd*100:.2f}%"
            )

            # 5% convergence criterion on running conductivity
            if rsd < 0.05:
                print(
                    f"Convergence reached (RSD < 5%) at "
                    f"{sim.current_time*1e9:.2f}ns"
                )
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
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns "
        f"with distance-dependent migration barriers"
    ),
    "Ea_fit_slope_eV_per_A": float(m_fit),
    "Ea_fit_intercept_eV": float(b_fit),
    "Ea_sample_min_eV": float(Ea_samples.min()),
    "Ea_sample_max_eV": float(Ea_samples.max()),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")