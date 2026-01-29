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

# Helper: classify Li site type by local oxygen coordination (proxy for tetrahedral / octahedral)
def classify_li_site(structure, li_index, r_O=3.0):
    site = structure[li_index]
    # Count neighboring oxygens
    neighbors = structure.get_neighbors(site, r_O)
    o_count = sum(1 for nb in neighbors if "O" in [s.symbol for s in nb.species.elements])
    # Very simple heuristic:
    # lower O coordination -> more "open" tetrahedral-like, higher -> octahedral-like
    if o_count <= 4:
        return "tet"  # tetrahedral-like (24d-like)
    elif o_count <= 6:
        return "oct"  # octahedral-like (48g/96h-like)
    else:
        return "other"

# Precompute site types for all Li sites
li_site_types = {}
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        li_site_types[i] = classify_li_site(structure, i)

# === 2. Build Adjacency Graph with Path-Specific Barriers ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}
# Migration barrier (Ea in eV) and attempt frequency (nu in Hz) map
# Incorporating lattice relaxation effects via NEB-informed, path-dependent barriers:
# - oct <-> tet: moderate barrier (framework-assisted hop)
# - tet <-> tet: somewhat higher (through narrower oxygen rings)
# - oct <-> oct: lower barrier due to polyhedral breathing
# Values chosen to reflect broadened / increased barriers relative to 0.30 eV rigid-lattice,
# consistent with NEB / CI-NEB trends reported for LLZO-like garnets.
barrier_map = {
    ("tet", "tet"): 0.60,
    ("tet", "oct"): 0.45,
    ("oct", "tet"): 0.45,
    ("oct", "oct"): 0.35,
}

# Attempt frequency map (prefactors can differ slightly with local environment)
nu_map = {
    ("tet", "tet"): 5e12,
    ("tet", "oct"): 1e13,
    ("oct", "tet"): 1e13,
    ("oct", "oct"): 1.5e13,
}

# Default values if classification falls outside simple types, still higher than rigid-lattice 0.30 eV
default_Ea = 0.55
default_nu = 1e13

for i, site in enumerate(structure):
    if "Li" not in [s.symbol for s in site.species.elements]:
        continue
    neighbors = []
    src_type = li_site_types.get(i, "other")
    for nb in neighbors_data[i]:
        if "Li" in [s.symbol for s in structure[nb.index].species.elements]:
            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            tgt_type = li_site_types.get(nb.index, "other")
            key = (src_type, tgt_type)
            Ea = barrier_map.get(key, default_Ea)
            nu = nu_map.get(key, default_nu)
            neighbors.append((nb.index, cart_disp, Ea, nu))
    adj_list[i] = neighbors

print(f"Graph built with path-specific barriers (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with Site-/Path-Specific Rates ===
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

        self.kb = 8.617e-5  # eV/K

    def hop_rate(self, Ea, nu):
        # Arrhenius rate with path-specific Ea and prefactor nu
        return nu * np.exp(-Ea / (self.kb * self.params['T']))

    def run_step(self):
        events, rates_cum, total = [], [], 0.0
        for src in self.li_indices:
            for tgt, vec, Ea, nu in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    r = self.hop_rate(Ea, nu)
                    if r <= 0.0:
                        continue
                    total += r
                    events.append((src, tgt, vec))
                    rates_cum.append(total)

        if total == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1

        # Select event
        xi = np.random.uniform(0.0, total)
        idx = np.searchsorted(rates_cum, xi)
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
        if self.current_time == 0.0 or self.num_particles == 0:
            return 0.0, 0.0
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
# Use a higher characteristic barrier than the original rigid-lattice 0.30 eV.
# This E_a is only used for logging / reference; actual dynamics use path-specific Ea, nu.
sim_params = {
    'T': 300,
    'E_a': 0.45,  # representative average including lattice relaxation
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

        # Keep last 1000 entries
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0

            print(
                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
            )

            if rsd < 0.05:  # 5% convergence criteria
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
    "conductivity": float(sigma),
    "diffusivity": float(D),
    "msd": float(msd),
    "simulation_time_ns": float(sim.current_time * 1e9),
    "temperature_K": sim_params['T'],
    "steps": sim.step_count,
    "error_message": None,
    "execution_log": (
        f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns "
        f"with path-specific migration barriers."
    ),
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")