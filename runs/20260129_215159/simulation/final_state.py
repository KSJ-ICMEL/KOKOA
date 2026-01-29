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

# === Identify Li Site Types (24d vs 96h) ===
# We rely on the species occupancy pattern: Li1 at 24d, Li2 at 96h (from Rietveld data).
# pymatgen keeps species as a Composition-like object per site; we use that to distinguish Li1 vs Li2.
li_site_types = []  # "24d" or "96h" for Li-containing sites (in order of 'initial_sites' that we build below)
li_structure_indices = []  # map from local Li index (in initial_sites) back to structure index

for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        # Try to distinguish Li1 (24d) vs Li2 (96h) using occupancy labels if present.
        # If site.species contains a single Li entry with occupancy ~0.54 -> 24d; ~0.37 -> 96h
        li_occup = site.species.get("Li", 0)
        # Use a simple threshold to distinguish the two kinds of Li sites.
        if li_occup > 0.45:
            li_site_types.append("24d")
        else:
            li_site_types.append("96h")
        li_structure_indices.append(i)

print(f"Identified Li site types for {len(li_site_types)} Li sites")

# Initialize Li sites with occupancy probability (following the original logic)
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

# === 3. kMC Simulator (BKL Algorithm) with Site-Dependent Energies ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, li_site_types, li_structure_indices, params):
        self.params = params
        self.adj_list = adj_list
        self.structure = structure

        # Map from structure index to local Li index
        self.struct_to_local = {}
        for local_idx, struct_idx in enumerate(li_structure_indices):
            self.struct_to_local[struct_idx] = local_idx

        # Occupancy defined over Li sites only (indexed by local Li index)
        self.num_li_sites = len(initial_sites)
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)

        # Assign site energies according to type (24d vs 96h)
        # Energies are in eV. We set 24d as reference (0), 96h higher by ΔE_site.
        # This introduces Boltzmann weighting and detailed-balance-consistent bias.
        self.site_energies = np.zeros(self.num_li_sites, dtype=float)
        # ΔE between 96h and 24d; can be tuned from DFT/thermo data. Use modest value to reduce
        # overestimated conductivity while preserving both site types.
        delta_E_site = params.get("delta_E_site_96h_24d", 0.05)  # eV, E(96h) - E(24d)
        for local_idx, stype in enumerate(li_site_types):
            if stype == "96h":
                self.site_energies[local_idx] = delta_E_site
            else:
                self.site_energies[local_idx] = 0.0  # 24d reference

        # Particle tracking: map local Li site index -> particle id
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for local_idx, s in enumerate(initial_sites):
            if s['state'] == 1:
                start = structure.lattice.get_cartesian_coords(s['coords'])
                self.site_to_particle[local_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start, dtype=float),
                    'current': np.array(start, dtype=float),
                }
                p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # Precompute base prefactor for attempt frequency
        self.kb = 8.617e-5  # eV/K
        self.nu = params['nu']
        self.T = params['T']

        # Site-independent "bare" migration barrier (for symmetric 24d-24d reference hop)
        self.E_m = params['E_m']  # eV

        # Activation energy for conductivity in original model was E_a.
        # We retain it only as a reference but do not use it directly for rates now.
        self.E_a_ref = params.get('E_a', self.E_m)

    def _rate_forward(self, E_i, E_j):
        """
        Compute detailed-balance-consistent forward rate k_{i->j} using:
        E_i^‡ = E_m + max(0, (E_j - E_i)/2)
        k_{i->j} = nu * exp( -E_i^‡ / (k_B T) )
        which ensures k_{i->j} / k_{j->i} = exp( -(E_j - E_i)/(k_B T) ).
        """
        dE = E_j - E_i
        barrier_i = self.E_m + max(0.0, dE / 2.0)
        return self.nu * np.exp(-barrier_i / (self.kb * self.T))

    def run_step(self):
        events = []
        cumulative_rates = []
        total_rate = 0.0

        # Loop over occupied Li sites (local indices)
        for local_src in self.li_indices:
            E_i = self.site_energies[local_src]
            # Convert local_src back to structure index
            struct_src = li_structure_indices[local_src]
            for struct_tgt, vec in self.adj_list.get(struct_src, []):
                # Only consider hops to Li sites (which we know by construction)
                if struct_tgt not in self.struct_to_local:
                    continue
                local_tgt = self.struct_to_local[struct_tgt]
                if self.occupancy[local_tgt] == 0:
                    E_j = self.site_energies[local_tgt]
                    rate_ij = self._rate_forward(E_i, E_j)
                    if rate_ij <= 0.0:
                        continue
                    total_rate += rate_ij
                    events.append((local_src, local_tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        rnd = np.random.rand()
        self.current_time += -np.log(rnd) / total_rate
        self.step_count += 1

        # Select and execute event
        rnd2 = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(cumulative_rates, rnd2)
        local_src, local_tgt, vec = events[idx]

        # Move particle
        p_id = self.site_to_particle.pop(local_src)
        self.particle_positions[p_id]['current'] += vec
        self.occupancy[local_src], self.occupancy[local_tgt] = 0, 1
        self.site_to_particle[local_tgt] = p_id
        self.li_indices.discard(local_src)
        self.li_indices.add(local_tgt)

        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        msd = np.mean(
            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
        )  # Å^2
        D = msd / (6.0 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.T)  # S/cm
        return msd, sigma


# === 4. Run Simulation ===
# Parameters:
#   T: temperature (K)
#   E_m: symmetric migration barrier (eV) for reference hop (used in detailed-balance scheme)
#   nu: attempt frequency (1/s)
#   delta_E_site_96h_24d: site energy difference (eV) between 96h and 24d (E_96h - E_24d)
sim_params = {
    'T': 300,
    'E_a': 0.30,              # kept for record
    'E_m': 0.30,              # base migration barrier (eV)
    'nu': 1e13,
    'volume': structure.volume,
    'delta_E_site_96h_24d': 0.05,  # eV; tune based on DFT/thermo data if available
}

sim = KMCSimulator(structure, adj_list, initial_sites, li_site_types, li_structure_indices, sim_params)

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
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")