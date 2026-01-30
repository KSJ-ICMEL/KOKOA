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

# === 1b. Identify Li Wyckoff Sites and Assign Site Energies ===
#
# Evidence-based modeling choice:
# Internal and web context establish that:
#   - 24d sites are tetrahedral Li(1) (Li1) sites
#   - 96h sites are octahedral / split Li(2) (Li2) sites
#   - 24d and 96h have different site energies (ΔE_site != 0)
#
# We therefore:
#   - Detect Li sites belonging to Wyckoff 24d vs 96h using symmetry analysis
#   - Assign a lower reference energy to tetrahedral 24d sites (E_24d = 0)
#   - Assign a higher energy to 96h sites (E_96h = ΔE_site > 0)
#
# Only relative energy differences matter for detailed balance. A reasonable
# order-of-magnitude splitting consistent with literature is of order
# ~0.05–0.1 eV; we keep this as a user-tunable parameter ΔE_site.
#
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sga = SpacegroupAnalyzer(structure, symprec=1e-3)
symm_struct = sga.get_symmetrized_structure()

# Map symmetry-equivalent indices to their Wyckoff symbol
wyckoff_label_by_index = {}
for wyckoff_symbol, equiv_indices in zip(symm_struct.wyckoff_symbols, symm_struct.equivalent_indices):
    for idx in equiv_indices:
        wyckoff_label_by_index[idx] = wyckoff_symbol

# Assign site energies (in eV) to Li sites based on Wyckoff symbol
# Reference: E_24d = 0.0; E_96h = ΔE_site > 0
Delta_E_site = 0.08  # eV; tunable parameter capturing 24d–96h energy difference

site_energies = np.zeros(len(structure), dtype=float)
li_site_indices = []

for i, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        li_site_indices.append(i)
        wyck = wyckoff_label_by_index.get(i, "")
        # Tetrahedral 24d: low-energy
        if wyck.strip().lower() == "24d":
            site_energies[i] = 0.0
        # Octahedral 96h: higher energy
        elif wyck.strip().lower() == "96h":
            site_energies[i] = Delta_E_site
        else:
            # For any other Li Wyckoff (should be rare), assign intermediate energy
            site_energies[i] = 0.5 * Delta_E_site

print("Assigned site energies to Li sites based on Wyckoff positions.")
print(f"Example energies (first 10 Li sites): {[(i, site_energies[i]) for i in li_site_indices[:10]]}")

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

# === 3. kMC Simulator (BKL Algorithm) with Site-Energy-Dependent Rates ===
#
# We now incorporate thermodynamic site-energy differences into the hop rates,
# enforcing detailed balance:
#
# Let:
#   E_i, E_j  : site energies at origin and destination (in eV)
#   ΔE_ij     = E_j - E_i
#   E_a       : symmetric part of barrier (in eV)
#   nu        : attempt frequency (Hz)
#   k_B       : Boltzmann constant (eV/K)
#
# A standard, evidence-based detailed-balance-consistent choice is:
#
#   k_{i→j} = ν * exp( - [ E_a + max(0, ΔE_ij) ] / (k_B T) )
#   k_{j→i} = ν * exp( - [ E_a + max(0, -ΔE_ij) ] / (k_B T) )
#
# This construction guarantees:
#
#   k_{i→j} / k_{j→i} = exp( -ΔE_ij / (k_B T) )
#
# so that equilibrium occupancies follow:
#
#   p_i ∝ exp( -E_i / (k_B T) )
#
# and hopping into lower-energy (e.g., 24d) sites is favored,
# capturing trapping and reduced mobility.

class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params, site_energies, li_site_indices):
        self.params = params
        self.adj_list = adj_list
        self.site_energies = site_energies
        self.li_site_indices = set(li_site_indices)

        # Occupancy only defined on Li sites; map from global structure index
        # to compact Li-site index for arrays
        self.li_index_list = sorted(list(self.li_site_indices))
        self.global_to_li = {g_idx: i for i, g_idx in enumerate(self.li_index_list)}
        self.li_to_global = {i: g_idx for i, g_idx in enumerate(self.li_index_list)}

        # Build occupancy array for Li sites
        self.occupancy = np.zeros(len(self.li_index_list), dtype=int)
        for local_i, g_idx in enumerate(self.li_index_list):
            state = initial_sites[local_i]['state'] if local_i < len(initial_sites) else 0
            self.occupancy[local_i] = state

        # Initialize particle mapping using global indices
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for local_i, g_idx in enumerate(self.li_index_list):
            if self.occupancy[local_i] == 1:
                start = structure.lattice.get_cartesian_coords(
                    structure[g_idx].frac_coords
                )
                self.site_to_particle[g_idx] = p_id
                self.particle_positions[p_id] = {
                    'start': np.array(start),
                    'current': np.array(start),
                }
                p_id += 1

        self.num_particles = len(self.particle_positions)
        self.current_time = 0.0
        self.step_count = 0

        self.kb = 8.617e-5  # eV/K
        self.E_a = params['E_a']
        self.nu = params['nu']

    def _rate(self, Ei, Ej):
        """Compute hop rate from site with energy Ei to Ej with detailed balance."""
        dE = Ej - Ei
        barrier = self.E_a + max(0.0, dE)
        return self.nu * np.exp(-barrier / (self.kb * self.params['T']))

    def run_step(self):
        events = []
        rates_cumulative = []
        total_rate = 0.0

        # Loop over occupied global Li sites
        for g_src, p_id in list(self.site_to_particle.items()):
            Ei = self.site_energies[g_src]
            local_src = self.global_to_li[g_src]
            for g_tgt, vec in self.adj_list.get(g_src, []):
                if g_tgt not in self.li_site_indices:
                    continue
                local_tgt = self.global_to_li[g_tgt]
                if self.occupancy[local_tgt] == 0:
                    Ej = self.site_energies[g_tgt]
                    k_ij = self._rate(Ei, Ej)
                    if k_ij <= 0.0:
                        continue
                    total_rate += k_ij
                    events.append((g_src, g_tgt, vec, k_ij))
                    rates_cumulative.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select and execute event
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(rates_cumulative, r)
        g_src, g_tgt, vec, k_ij = events[idx]

        p_id = self.site_to_particle.pop(g_src)
        self.particle_positions[p_id]['current'] += vec

        local_src = self.global_to_li[g_src]
        local_tgt = self.global_to_li[g_tgt]
        self.occupancy[local_src] = 0
        self.occupancy[local_tgt] = 1

        self.site_to_particle[g_tgt] = p_id

        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0.0, 0.0
        msd = np.mean(
            [
                np.sum((p['current'] - p['start']) ** 2)
                for p in self.particle_positions.values()
            ]
        )  # Å^2
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (
            n * (1.602e-19) ** 2 * D / (1.38e-23 * self.params['T'])
        )  # S/cm (Nernst-Einstein)
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {'T': 300, 'E_a': 0.30, 'nu': 1e13, 'volume': structure.volume}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params, site_energies, li_site_indices)

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
                print(
                    f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns"
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
    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns with site-energy-dependent rates",
}

result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result to: {result_path}")