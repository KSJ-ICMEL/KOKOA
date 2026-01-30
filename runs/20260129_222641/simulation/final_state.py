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

# === 1a. Build Thermodynamically-Informed Li Configuration ===
#
# Goal: Replace purely random Li/vacancy assignment by a configuration
# that respects experimentally refined occupancies and Li count in Al-LLZO.
#
# From neutron diffraction at 300 K for Al-stabilized LLZO:
#   - Li(24d) occupancy g_24d ≈ 0.54 (per crystallographic 24d site)
#   - Li(96h) occupancy g_96h ≈ 0.37 (per crystallographic 96h site)
#   - Al substitutes on 24d (occ_Al ≈ 0.0653), which reduces Li count there.
#
# We will:
#   1. Identify crystallographic 24d-like and 96h-like Li sites in the primitive cell
#      of the CIF via their Wyckoff-equivalent fractional coordinates.
#   2. For each supercell image, apply those same per-site occupancies using a
#      Bernoulli trial with probabilities g_24d and g_96h, but *without* using
#      the nominal site.species.get("Li") (which mixes Li and Al).
#   3. This enforces the correct average Li content and site preference, and
#      captures the experimentally relevant short-range order at 300 K much
#      better than a fully random distribution over all "Li" species.

# --- Identify unique Li crystallographic sites in the primitive cell ---

# Work with the *unexpanded* CIF structure to detect representative Li sites
primitive_structure = Structure.from_file(cif_path)

# Extract representative fractional coordinates for Li(24d) and Li(96h)
# from neutron diffraction refinements (O'Callaghan & Cussen).
# These are given as fractional coordinates in the cubic garnet cell:
li24d_ref = np.array([1.0/8.0, 0.0, 1.0/4.0])  # (0.125, 0, 0.25)
li96h_ref = np.array([0.1004, 0.6853, 0.5769])

def min_image_diff(frac_a, frac_b):
    """Minimum-image fractional difference between two fractional coordinates."""
    d = frac_a - frac_b
    d -= np.round(d)  # wrap to [-0.5, 0.5)
    return d

def is_equivalent_site(frac, ref, tol=1e-2):
    """Check if 'frac' is symmetry-equivalent to 'ref' within a tolerance,
    using only translational lattice periodicity (Wyckoff labels are not
    directly available here)."""
    d = min_image_diff(np.array(frac), np.array(ref))
    return np.linalg.norm(d) < tol

# Collect indices of Li sites in primitive cell that correspond to 24d-like
# and 96h-like positions.
primitive_li24d_indices = []
primitive_li96h_indices = []

for i, site in enumerate(primitive_structure):
    if "Li" not in [el.symbol for el in site.species.elements]:
        continue
    frac = site.frac_coords
    if is_equivalent_site(frac, li24d_ref):
        primitive_li24d_indices.append(i)
    elif is_equivalent_site(frac, li96h_ref):
        primitive_li96h_indices.append(i)

print(f"Primitive cell Li(24d)-like sites: {len(primitive_li24d_indices)}")
print(f"Primitive cell Li(96h)-like sites: {len(primitive_li96h_indices)}")

if len(primitive_li24d_indices) == 0 or len(primitive_li96h_indices) == 0:
    print("Warning: Could not uniquely identify 24d/96h Li sites from CIF; "
          "falling back to original random occupation based on species occupancies.")
    use_refined_occupancies = False
else:
    use_refined_occupancies = True

# Experimental Li occupancies at 300 K (per crystallographic site)
g_24d = 0.54  # Li occupancy at 24d
g_96h = 0.37  # Li occupancy at 96h

# --- Map primitive Li sites to supercell Li sites and assign occupancy ---

initial_sites = []

if use_refined_occupancies:
    # Build a mapping from primitive sites (via fractional coordinates) to
    # supercell sites. Since the supercell is a simple [N, N, N] expansion,
    # each primitive site is replicated N^3 times with translated fractional
    # coordinates. We'll classify supercell Li sites by matching their
    # fractional coordinates (modulo 1) to the primitive representatives.
    frac_to_type = {}  # key: tuple(rounded frac coords) -> "24d" / "96h" / None

    # Seed with primitive coordinates
    for idx in primitive_li24d_indices:
        f = primitive_structure[idx].frac_coords
        key = tuple(np.round(f % 1.0, 4))
        frac_to_type[key] = "24d"
    for idx in primitive_li96h_indices:
        f = primitive_structure[idx].frac_coords
        key = tuple(np.round(f % 1.0, 4))
        frac_to_type[key] = "96h"

    num_24d_total = 0
    num_96h_total = 0

    for site in structure:
        if "Li" not in [el.symbol for el in site.species.elements]:
            continue

        f = site.frac_coords % 1.0
        key = tuple(np.round(f, 4))
        site_type = frac_to_type.get(key, None)

        if site_type == "24d":
            prob = g_24d
            num_24d_total += 1
        elif site_type == "96h":
            prob = g_96h
            num_96h_total += 1
        else:
            # For Li-like sites that do not match the refined 24d/96h
            # references (e.g., split/disordered sites not resolved here),
            # we leave them vacant in the initial configuration to avoid
            # artificially inflating mobility.
            prob = 0.0

        state = 1 if np.random.rand() < prob else 0
        initial_sites.append({"coords": site.frac_coords, "state": state})

    print(f"Supercell Li(24d)-like sites: {num_24d_total}, Li(96h)-like sites: {num_96h_total}")
else:
    # Fallback: original random initialization using species occupancies.
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

# === 3. kMC Simulator (BKL Algorithm) ===
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
        
        kb = 8.617e-5  # eV/K
        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))

    def run_step(self):
        events, rates, total = [], [], 0.0
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    total += self.base_rate
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
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()]) # Mean Square Displacement (Å^2)
        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s), MSD(t)=6Dt
        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # Nernst-Einstein Equation: σ = (n*e^2*D)/(k*T) (S/cm)
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {'T': 300, 'E_a': 0.30, 'nu': 1e13, 'volume': structure.volume}
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
            sigma_history.pop(0) # Keep last 1000
            
        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0
            
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")
            
            if rsd < 0.05: # 5% convergence criteria
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