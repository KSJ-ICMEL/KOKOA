"""KOKOA Simulation #6 - 2026-01-25 20:55:33"""
import os, sys, json
import numpy as np
from pymatgen.core import Structure

# Pre-loaded structure
structure = Structure.from_file("C:/Users/sjkim/KOKOA/LLZO.cif")
print(f"Structure loaded: {len(structure)} atoms")

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

# === Define site energies (24d vs 96h) ===
# Default offsets (eV). Adjust if DFT values are available.
E_24d = 0.0
E_96h = 0.15

site_energies = []
for site in structure:
    # Many CIFs store the Wyckoff label in site.properties['wyckoff']
    wyck = site.properties.get('wyckoff') or site.properties.get('label') or ''
    if '24d' in str(wyck):
        site_energies.append(E_24d)
    else:
        # Assume all other Li sites are 96h (or higher‑energy) sites
        site_energies.append(E_96h)
site_energies = np.array(site_energies)

# === Define initial sites (occupancy) ===
# Preserve the original count of Li ions but distribute them according to Boltzmann weights.
initial_sites = []
for i, site in enumerate(structure):
    # placeholder state – will be overwritten in KMCSimulator
    initial_sites.append({'state': 0, 'coords': site.frac_coords})

# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, site_energies, params):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.zeros(len(initial_sites), dtype=int)
        self.site_energies = np.array(site_energies)

        # Total number of Li ions present in the original structure
        total_li = sum(1 for s in structure if "Li" in s.species.elements[0].symbol)

        # Boltzmann probabilities for each site
        kb = 8.617e-5  # eV/K
        prob = np.exp(-self.site_energies / (kb * self.params['T']))
        prob /= prob.sum()

        # Randomly choose sites according to the probabilities (without replacement)
        chosen = np.random.choice(len(initial_sites), size=total_li, replace=False, p=prob)
        self.occupancy[chosen] = 1

        # Mapping from site index to particle id and particle trajectories
        self.site_to_particle = {}
        self.particle_positions = {}
        p_id = 0
        for idx in chosen:
            start = structure.lattice.get_cartesian_coords(initial_sites[idx]['coords'])
            self.site_to_particle[idx] = p_id
            self.particle_positions[p_id] = {'start': np.array(start), 'current': np.array(start)}
            p_id += 1

        self.li_indices = set(self.site_to_particle.keys())
        self.num_particles = len(self.li_indices)
        self.current_time = 0.0
        self.step_count = 0

        # Physical constants
        self.kb = kb  # eV/K
        self.alpha = params.get('alpha', 0.05)  # eV per occupied neighbour

        # Pre‑compute temperature‑dependent quantities for phonon‑assisted hopping
        self._update_phonon_quantities()

    def _update_phonon_quantities(self):
        """Compute temperature‑dependent attempt frequency and barrier reduction.
        Uses a simple linear scaling for ν(T) and a √T scaling for ΔE_ph(T).
        """
        T = self.params['T']
        nu0 = self.params.get('nu0', self.params.get('nu', 1e13))
        beta = self.params.get('beta', 0.0)
        self.nu_T = nu0 * (1.0 + beta * (T - 300.0) / 300.0)
        gamma = self.params.get('gamma', 0.0)
        self.delta_E_ph = gamma * np.sqrt(T / 300.0)

    def _compute_rate(self, src, tgt):
        """Calculate the hopping rate for a specific src→tgt hop.
        Includes site‑energy offsets, environment‑dependent barrier,
        phonon‑assisted reduction, and temperature‑dependent attempt frequency.
        """
        # Count occupied neighbours of the target site (excluding the moving ion)
        n_occ = 0
        for nb_idx, _ in self.adj_list.get(tgt, []):
            if self.occupancy[nb_idx] == 1:
                n_occ += 1
        # Base activation energy plus neighbour penalty minus phonon reduction
        E_eff = self.params['E_a'] + self.alpha * n_occ - self.delta_E_ph
        # Add site‑energy contribution (Miller‑Abrahams term)
        E_eff += self.site_energies[tgt] - self.site_energies[src]
        # Prevent negative barriers
        if E_eff < 0:
            E_eff = 0.0
        rate = self.nu_T * np.exp(-E_eff / (self.kb * self.params['T']))
        return rate

    def run_step(self):
        events = []          # (src, tgt, vec)
        cumulative_rates = []
        total_rate = 0.0

        # Build list of possible hops with updated rates
        for src in self.li_indices:
            for tgt, vec in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    rate = self._compute_rate(src, tgt)
                    if rate <= 0:
                        continue
                    total_rate += rate
                    events.append((src, tgt, vec))
                    cumulative_rates.append(total_rate)

        if total_rate == 0.0:
            return False  # Deadlock

        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1

        # Select event based on cumulative rates
        r = np.random.uniform(0, total_rate)
        idx = np.searchsorted(cumulative_rates, r)
        src, tgt, vec = events[idx]

        # Execute the selected hop
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
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])  # Å^2
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # S/cm
        return msd, sigma

# === Simulation parameters (including site‑energy values) ===
sim_params = {
    'T': 300,               # K
    'E_a': 0.5,             # base migration barrier (eV) – keep previous value
    'alpha': 0.05,          # neighbour penalty (eV)
    'nu0': 1e13,            # attempt frequency (Hz)
    'beta': 0.0,            # linear T‑dependence of ν
    'gamma': 0.0,           # phonon‑assisted reduction prefactor (eV)
    'volume': structure.volume,
    # site‑energy offsets are already encoded in the site_energies array
}

# === 4. Run the kMC simulation ===
sim = KMCSimulator(structure, adj_list, initial_sites, site_energies, sim_params)

# Convergence loop (same as original)
while True:
    if not sim.run_step():
        break
    if sim.step_count % 1000 == 0:
        msd, sigma = sim.calculate_properties()
        if sigma > 1e-6:  # arbitrary convergence criterion
            break

# === 5. Output results ===
msd, sigma = sim.calculate_properties()
print(f"Final conductivity: {sigma:.3e} S/cm")

# Save results (unchanged from original script)
result_path = os.path.join(os.getcwd(), "kMC_result.txt")
with open(result_path, "w") as f:
    f.write(f"Conductivity (S/cm): {sigma}\n")
