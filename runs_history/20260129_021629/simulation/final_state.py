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
li_site_indices = []  # indices of Li sites in the structure
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(i)

num_li_sites = len(li_site_indices)
print(f"Li sites initialized: {num_li_sites}")

# === 2. Build Adjacency Graph Restricting to Li Sublattice ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)

# Map from global structure index -> local Li-site index
global_to_li = {g_idx: li_idx for li_idx, g_idx in enumerate(li_site_indices)}

# Build adjacency list on Li-site indices
adj_list = {li_idx: [] for li_idx in range(num_li_sites)}

for li_idx, g_idx in enumerate(li_site_indices):
    site = structure[g_idx]
    neighbors = []
    for nb in neighbors_data[g_idx]:
        nb_site = structure[nb.index]
        if "Li" in [s.symbol for s in nb_site.species.elements]:
            if nb.index not in global_to_li:
                continue
            tgt_li_idx = global_to_li[nb.index]
            frac_diff = nb_site.frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((tgt_li_idx, cart_disp))
    adj_list[li_idx] = neighbors

print(f"Li-only graph built (cutoff={cutoff} Å)")

# === 2b. Precompute Environment Metrics for Haven Ratio Estimation ===
# We will need the distinct part of the van Hove function to estimate H,
# but that requires trajectories, not static geometry. Static structure
# is still used for mapping sites and neighbors only.

# === 3. kMC Simulator (BKL Algorithm) with Haven Ratio Estimation ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, li_site_indices, params):
        self.params = params
        self.adj_list = adj_list
        self.li_site_indices = li_site_indices

        # Occupancy defined over Li sites only
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map Li-site index -> particle id, and track positions
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for li_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start_cart = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[li_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start_cart, dtype=float),
                    'current': np.array(start_cart, dtype=float)
                }
                p_id += 1

        self.num_particles = p_id
        self.li_indices = set(self.site_to_particle.keys())

        self.current_time = 0.0
        self.step_count = 0

        kb = 8.617e-5  # eV/K
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

        # For Haven ratio: track center of mass and distinct displacements
        # COM of all Li ions at each step
        self.initial_com = self._compute_com()
        self.current_com = self.initial_com.copy()

        # For collective MSD: we will accumulate squared COM displacement
        self.com_msd_samples = []

        # For tracer MSD, we already track individual positions

    def _compute_com(self):
        if self.num_particles == 0:
            return np.zeros(3)
        coords = np.array([p['current'] for p in self.particle_positions.values()])
        return np.mean(coords, axis=0)

    def run_step(self):
        events = []
        rates_cum = []
        total_rate = 0.0

        # Build event list using Li-site graph and current occupancies
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    total_rate += self.base_rate
                    events.append((src, tgt, vec))
                    rates_cum.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(rates_cum, r)
        src, tgt, vec = events[idx]

        # Update particle mapping and positions
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)

        # Update COM and record COM MSD sample
        self.current_com = self._compute_com()
        com_disp = self.current_com - self.initial_com
        com_msd = np.dot(com_disp, com_disp)
        self.com_msd_samples.append((self.current_time, com_msd))

        return True

    def calculate_properties(self):
        if self.current_time == 0 or self.num_particles == 0:
            return 0.0, 0.0, 1.0, 0.0

        # Tracer MSD
        msd_tracer = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )  # Å^2

        # Tracer diffusion coefficient (per ion)
        D_tracer = msd_tracer / (6.0 * self.current_time) * 1e-16  # cm^2/s

        # Collective (charge-transport) MSD via COM motion
        # For N identical ions, COM displacement R_cm obeys:
        # <|Σ_i r_i|^2> = N^2 <|R_cm|^2>, and collective diffusion coefficient D_sigma is:
        # D_sigma = (1/(6tN)) <|Σ_i r_i|^2> = (N/(6t)) <|R_cm|^2>
        # so we can use COM MSD to estimate D_sigma.
        if len(self.com_msd_samples) > 1:
            # Use latest COM MSD value
            t_com, com_msd = self.com_msd_samples[-1]
            D_sigma = (self.num_particles * com_msd / (6.0 * t_com)) * 1e-16  # cm^2/s
        else:
            D_sigma = 0.0

        # Haven ratio H = D_sigma / D_tracer
        H = D_sigma / D_tracer if D_tracer > 0 else 1.0

        # Use modified Nernst–Einstein with Haven ratio:
        # σ = n e^2 D_sigma / (k_B T) = n e^2 H D_tracer / (k_B T)
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        e_charge = 1.602e-19  # C
        k_B_SI = 1.38e-23  # J/K

        sigma = (n * e_charge ** 2 * D_sigma) / (k_B_SI * self.params['T'])  # S/cm

        return msd_tracer, sigma, H, D_tracer

# === 4. Run Simulation ===
sim_params = {'T': 300, 'E_a': 0.30, 'nu': 1e13, 'volume': structure.volume}
sim = KMCSimulator(structure, adj_list, initial_sites, li_site_indices, sim_params)

target_time = 1000e-9  # 1000 ns
log_interval = 100
sigma_history = []
haven_history = []

while sim.current_time < target_time:
    if not sim.run_step():
        print("Deadlock - stopping")
        break
    if sim.step_count % log_interval == 0:
        msd_tracer, sigma, H, D_tracer = sim.calculate_properties()
        sigma_history.append(sigma)
        haven_history.append(H)

        # Keep last 1000 conductivity samples
        if len(sigma_history) > 1000:
            sigma_history.pop(0)
        if len(haven_history) > 1000:
            haven_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0.0

            avg_H = np.mean(haven_history)

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f} ns, "
                f"MSD_tracer={msd_tracer:.2f} Å^2, "
                f"D_tracer={D_tracer:.3e} cm^2/s, "
                f"Haven={avg_H:.3f}, "
                f"sigma={sigma*1e3:.4f} mS/cm, "
                f"RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f} ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f} ns, "
                f"MSD_tracer={msd_tracer:.2f} Å^2, "
                f"D_tracer={D_tracer:.3e} cm^2/s, "
                f"Haven={H:.3f}, "
                f"sigma={sigma*1e3:.4f} mS/cm"
            )

# Final result
msd_tracer, sigma, H_final, D_tracer_final = sim.calculate_properties()
D_sigma_final = H_final * D_tracer_final

print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']} K, Time={sim.current_time*1e9:.2f} ns")
print(f"Tracer D = {D_tracer_final:.4e} cm^2/s")
print(f"Charge-transport D_sigma = {D_sigma_final:.4e} cm^2/s")
print(f"Haven ratio H = {H_final:.3f}")
print(f"Conductivity (modified Nernst–Einstein): {sigma:.4e} S/cm")

# Save result to JSON
result = {
    "is_success": True,
    "conductivity_S_per_cm": float(sigma),
    "diffusivity_tracer_cm2_per_s": float(D_tracer_final),
    "diffusivity_sigma_cm2_per_s": float(D_sigma_final),
    "haven_ratio": float(H_final),
    "msd_tracer_A2": float(msd_tracer),
    "simulation_time_ns": float(sim.current_time * 1e9),
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f} ns; "
        f"Haven ratio {H_final:.3f}"
    ),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")