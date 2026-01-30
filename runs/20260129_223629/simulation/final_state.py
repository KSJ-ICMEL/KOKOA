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

# === 3. kMC Simulator with non-Poisson waiting times (CTRW-style) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Map sites to particles and track positions
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

        # Global simulation time and step counter
        self.current_time = 0.0
        self.step_count = 0

        # Microscopic attempt frequency and barriers retained for reference
        kb = 8.617e-5  # eV/K
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

        # Parameters for non-Poissonian waiting-time statistics (Weibull/CTRW)
        # ψ(τ) = (k/τ0) (τ/τ0)^(k-1) exp[-(τ/τ0)^k]
        # τ0 is chosen as 1/base_rate to connect with microscopic hopping
        k_shape = params.get('waiting_shape_k', 0.7)  # k<1 gives subdiffusive, non-Poisson behavior
        self.waiting_shape_k = k_shape
        # Guard against unphysical values; k=1 would reduce to exponential
        if self.waiting_shape_k <= 0:
            raise ValueError("waiting_shape_k must be positive")

        # Set scale τ0 from microscopic rate (τ0 = 1 / base_rate)
        if self.base_rate <= 0:
            raise ValueError("base_rate must be positive for waiting-time construction")
        self.waiting_scale_tau0 = 1.0 / self.base_rate

    def _sample_non_poisson_wait(self):
        """
        Draw a waiting time from a non-exponential (Weibull) distribution.

        Inverse transform for Weibull CDF:
          F(τ) = 1 - exp[-(τ/τ0)^k]
          u in (0,1) -> τ = τ0 * (-ln(1-u))^(1/k)
        """
        u = np.random.rand()
        # Protect against u=0; but np.random.rand() is in [0,1), so u=1 is more relevant
        if u == 0.0:
            u = np.nextafter(0, 1)
        return self.waiting_scale_tau0 * (-np.log(1.0 - u)) ** (1.0 / self.waiting_shape_k)

    def run_step(self):
        # Enumerate all possible single-ion hops (events)
        events = []
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    events.append((src, tgt, vec))

        if not events:
            return False  # Deadlock: no accessible moves

        # Non-Poissonian time advance: draw a single global waiting time
        # consistent with specified CTRW waiting-time statistics
        dt = self._sample_non_poisson_wait()
        self.current_time += dt
        self.step_count += 1

        # Select event uniformly among allowed hops (topological CTRW)
        idx = np.random.randint(len(events))
        src, tgt, vec = events[idx]

        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src], self.occupancy[tgt] = 0, 1
        self.site_to_particle[tgt] = p_id
        self.li_indices.discard(src)
        self.li_indices.add(tgt)
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        # Mean square displacement in Å^2
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])
        # For non-Poisson CTRW, MSD(t) ∝ t^α; here we still report an
        # effective D via MSD(t) = 6 D_eff t for comparison.
        D_eff = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s

        # Ion concentration n (ions/cm^3)
        n = self.num_particles / (self.params['volume'] * 1e-24)

        # Nernst–Einstein relation for an effective conductivity
        sigma = (n * (1.602e-19) ** 2 * D_eff) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    'E_a': 0.30,
    'nu': 1e13,
    'volume': structure.volume,
    # Shape parameter for non-Poisson waiting times: k<1 -> subdiffusive,
    # consistent with correlated, non-Poisson hopping statistics.
    'waiting_shape_k': 0.7
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

        # Keep only the last 1000 samples
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

            if rsd < 0.05:  # 5% convergence criterion
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
            )

# Final result
msd, sigma = sim.calculate_properties()
D_eff = msd / (6.0 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0

print(f"\n=== Simulation Complete ===")
print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
print(f"D_eff={D_eff:.4e} cm^2/s")
print(f"Conductivity (effective): {sigma:.4e} S/cm")

# Save result to JSON
result = {
    "is_success": True,
    "conductivity": sigma,
    "diffusivity": D_eff,
    "msd": msd,
    "simulation_time_ns": sim.current_time * 1e9,
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns "
        f"with non-Poisson waiting times (Weibull k={sim.waiting_shape_k})"
    ),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")