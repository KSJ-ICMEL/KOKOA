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

# ----------------------------------------------------------------------
# 1a. Identify Li Wyckoff types and assign site energies
# ----------------------------------------------------------------------
# We use LLZO crystallography: Li1 on 24d are low-energy sites,
# Li2 on 96h are higher-energy, following the internal paper.
# Assign a simple two-level site energy model (in eV) to break flat landscape.
#
# E(24d) = 0
# E(96h) = +ΔE
#
# ΔE is an adjustable parameter; choose modest value so that 24d are preferred
# but 96h still participate in diffusion. This directly introduces the
# thermodynamic bias missing in the original model.

def classify_li_site(structure, site_index, r_tol=0.1):
    """
    Classify Li sites using approximate fractional coordinates to distinguish
    24d (Li1) from 96h (Li2) based on reported positions:

        Li1 (24d): (1/8, 0, 1/4)
        Li2 (96h): (~0.098, ~0.686, ~0.576)

    In supercells, all equivalent sites share these fractional coordinates
    modulo lattice translations. We use a distance check in fractional space.
    """
    frac = structure[site_index].frac_coords
    # Reduce to [0,1) for comparison
    frac = frac - np.floor(frac)

    # Ideal prototype positions from the paper
    li1_frac = np.array([1/8, 0.0, 1/4])
    li2_frac = np.array([0.0980, 0.6859, 0.5764])

    # Compute minimum image difference in fractional space
    def min_image_dist(a, b):
        d = a - b
        d -= np.round(d)  # Wrap into [-0.5,0.5]
        return np.linalg.norm(d)

    d1 = min_image_dist(frac, li1_frac)
    d2 = min_image_dist(frac, li2_frac)

    if d1 < r_tol:
        return "24d"
    if d2 < r_tol:
        return "96h"
    return "unknown"

# Assign per-site energies (relative, in eV)
# Parameter: energy difference between 24d and 96h
DELTA_E_96H = 0.05  # eV, 96h higher than 24d; tune as needed

site_energies = []
li_site_types = []
for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        stype = classify_li_site(structure, i)
        li_site_types.append(stype)
        if stype == "24d":
            site_energies.append(0.0)
        elif stype == "96h":
            site_energies.append(DELTA_E_96H)
        else:
            # For unidentified Li sites (if any), assign intermediate energy
            site_energies.append(DELTA_E_96H * 0.5)
    else:
        li_site_types.append(None)
        site_energies.append(0.0)

site_energies = np.array(site_energies)

# === 1b. Initialize Li sites with occupancy probability ===
initial_sites = []
li_site_indices = []
for idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})
        li_site_indices.append(idx)

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

# === 3. kMC Simulator (BKL Algorithm with site energy landscape) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params, site_energies, li_site_types):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        self.site_energies = site_energies
        self.li_site_types = li_site_types

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

        # Base attempt frequency (s^-1)
        self.nu0 = params['nu']
        self.kb = 8.617e-5  # eV/K

        # Base barrier (for reference, from original code)
        self.E_a0 = params['E_a']

        # We introduce a simple energy-dependent barrier model that
        # respects detailed balance:
        #
        # Let E_i, E_j be site energies.
        # Symmetric Butler–Volmer–type splitting with β = 0.5:
        #
        # E_ij^‡ = E_a0 + 0.5 * max(0, E_j - E_i)
        # E_ji^‡ = E_a0 + 0.5 * max(0, E_i - E_j)
        #
        # Then:
        # k_ij = ν0 * exp[-(E_ij^‡ + max(0, E_j - E_i))/kT]
        # Simplifies (with our choice) to:
        # k_ij = ν0 * exp(-E_a0/(kT)) * exp(-(E_j - E_i)/(2kT))
        #
        # So:
        # k_ij = base_rate * exp(-ΔE / (2 kT))
        #
        # This ensures k_ij / k_ji = exp(-(E_j - E_i)/kT),
        # enforcing the correct Boltzmann equilibrium distribution
        # over sites while introducing uphill/downhill modulation
        # of the rates.

        self.base_rate = self.nu0 * np.exp(-self.E_a0 / (self.kb * params['T']))

    def hop_rate(self, src, tgt):
        """Compute transition rate from src to tgt using site energies."""
        dE = self.site_energies[tgt] - self.site_energies[src]
        # Symmetric splitting, see comment in __init__
        factor = np.exp(-dE / (2.0 * self.kb * self.params['T']))
        return self.base_rate * factor

    def run_step(self):
        events = []
        cumulative_rates = []
        total = 0.0

        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self.hop_rate(src, tgt)
                    if rate <= 0.0:
                        continue
                    total += rate
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total)

        if total == 0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0.0, total)
        idx = np.searchsorted(cumulative_rates, r)
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
            return 0.0, 0.0
        msd = np.mean([
            np.sum((p['current'] - p['start']) ** 2)
            for p in self.particle_positions.values()
        ])  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 300,
    'E_a': 0.30,      # base barrier (eV)
    'nu': 1e13,       # attempt frequency (s^-1)
    'volume': structure.volume
}

sim = KMCSimulator(structure, adj_list, initial_sites, sim_params, site_energies, li_site_types)

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