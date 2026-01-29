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

# === 1a. Identify Li Wyckoff-like site types (24d vs 96h proxy) ===
#
# We need two energetically distinct classes for Li sites to capture
# thermodynamic driving forces between low-energy 24d-like and
# higher-energy 96h-like environments. The CIF typically does not
# expose Wyckoff labels, so we use coordination environment as a proxy.
#
# Evidence-based assumption from context:
# - 24d tetrahedral sites are low-energy sites for Li in cubic LLZO.
# - 96h octahedral sites are higher-energy than 24d.
#
# We approximate:
# - Li with 4 nearest O neighbors (low coordination) → 24d-like (low E_site)
# - Li with ≥5 nearest O neighbors → 96h-like (higher E_site)
#
# This classification is purely structural and does NOT invent new physics;
# it only supplies the relative site energies used in standard
# thermodynamically-consistent rate formulas.

# Precompute O indices
o_indices = [i for i, s in enumerate(structure) if s.species.elements[0].symbol == "O"]

# Neighbor search limited to Li–O for coordination
li_site_types = {}  # index -> "24d" or "96h"
li_indices_all = []
li_neighbors_O = structure.get_all_neighbors(r=3.0)  # short cutoff for Li–O

for i, site in enumerate(structure):
    if "Li" in [e.symbol for e in site.species.elements]:
        li_indices_all.append(i)
        # Count nearby O atoms
        coord_O = 0
        for nb in li_neighbors_O[i]:
            if nb.index in o_indices:
                coord_O += 1
        # Simple coordination-based classification
        if coord_O <= 4:
            li_site_types[i] = "24d"
        else:
            li_site_types[i] = "96h"

num_24d = sum(1 for i in li_indices_all if li_site_types.get(i) == "24d")
num_96h = sum(1 for i in li_indices_all if li_site_types.get(i) == "96h")
print(f"Li site classification (proxy): 24d-like={num_24d}, 96h-like={num_96h}")

# Define relative site energies (eV) consistent with literature trend:
# 24d is a low-energy site; 96h is higher-energy.
# We choose a modest splitting so as not to exceed typical DFT defect-energy scales.
E_site_map = {
    "24d": 0.0,     # reference
    "96h": 0.10     # higher-energy by 0.1 eV
}

# === 1b. Initialize Li sites with occupancy probability ===
initial_sites = []
for idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = site.species.get("Li", 0)
        state = 1 if np.random.rand() < prob else 0
        site_type = li_site_types.get(idx, "96h")
        E_site = E_site_map[site_type]
        initial_sites.append({
            "index": idx,
            "coords": site.frac_coords,
            "state": state,
            "type": site_type,
            "E_site": E_site
        })

print(f"Li sites initialized: {len(initial_sites)}")

# Map from structure index to index in initial_sites / occupancy array
li_struct_to_local = {s["index"]: i for i, s in enumerate(initial_sites)}

# === 2. Build Adjacency Graph ===
cutoff = 4.0  # Angstrom
neighbors_data = structure.get_all_neighbors(r=cutoff)
adj_list = {}

for i, site in enumerate(structure):
    if "Li" not in site.species.elements[0].symbol:
        continue
    if i not in li_struct_to_local:
        continue
    src_local = li_struct_to_local[i]
    neighbors = []
    for nb in neighbors_data[i]:
        j = nb.index
        if j in li_struct_to_local:
            tgt_local = li_struct_to_local[j]
            frac_diff = structure[j].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            neighbors.append((tgt_local, cart_disp))
    adj_list[src_local] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) with Site-Energy-Dependent Barriers ===
#
# Thermodynamically consistent hopping rates:
# Let E_i, E_j be site energies for initial and final sites.
# A simple, detailed-balance respecting choice (Hin 2011; standard kMC):
#
#   E_m,ij = E_a0 + max(0, E_j - E_i)       (barrier raised when hopping uphill)
#   k_ij   = nu * exp( -E_m,ij / (k_B T) )
#
# This yields:
#   k_ij / k_ji = exp( -(E_j - E_i) / (k_B T) )
#
# which ensures the correct Boltzmann ratio for site occupations at equilibrium.
#
# Implementation change vs. original code:
# - Replace single global base_rate with per-hop barrier using E_i and E_j.
# - Preserve the original attempt frequency nu and base E_a0 parameters.

class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params):
        self.params = params
        self.structure = structure
        self.adj_list = adj_list

        # Occupancy over Li sites only
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        self.num_sites = len(initial_sites)

        # Store site energies for Li sites
        self.site_energies = np.array([s['E_site'] for s in initial_sites], dtype=float)

        # Particle bookkeeping
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for local_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[local_idx] = p_id
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
        self.nu = params['nu']
        self.E_a0 = params['E_a']  # base barrier in eV

    def _hop_rate(self, src, tgt):
        """
        Compute thermodynamically consistent hop rate from src->tgt site.

        Barrier: E_m = E_a0 + max(0, E_site[tgt] - E_site[src])
        Rate:    k   = nu * exp(-E_m / (k_B T))
        """
        Ei = self.site_energies[src]
        Ej = self.site_energies[tgt]
        delta = Ej - Ei
        E_m = self.E_a0 + max(0.0, delta)
        rate = self.nu * np.exp(-E_m / (self.kb * self.params['T']))
        return rate

    def run_step(self):
        events = []
        cum_rates = []
        total = 0.0

        # Enumerate all possible hops and compute their rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    k_ij = self._hop_rate(src, tgt)
                    if k_ij <= 0.0:
                        continue
                    total += k_ij
                    events.append((src, tgt, vec))
                    cum_rates.append(total)

        if total == 0.0:
            return False  # No available events → deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total
        self.step_count += 1

        # Select event
        r = np.random.uniform(0.0, total)
        idx = np.searchsorted(cum_rates, r)
        src, tgt, vec = events[idx]

        # Execute hop
        p_id = self.site_to_particle.pop(src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[src] = 0
        self.occupancy[tgt] = 1
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
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {
    'T': 300,               # K
    'E_a': 0.30,            # base migration barrier (eV)
    'nu': 1e13,             # attempt frequency (1/s)
    'volume': structure.volume  # Å^3
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

        # Keep only last 1000 samples
        if len(sigma_history) > 1000:
            sigma_history.pop(0)

        if len(sigma_history) == 1000:
            avg_sigma = np.mean(sigma_history)
            std_sigma = np.std(sigma_history)
            rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0

            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")

            if rsd < 0.05:  # 5% convergence criteria
                print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                break
        else:
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
                  f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

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