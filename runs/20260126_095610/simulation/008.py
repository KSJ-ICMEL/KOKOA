import os, sys, json
import numpy as np
from pymatgen.core import Structure
import itertools

# Load structure (CIF is in current directory)
structure = Structure.from_file("LLZO.cif")
N = 4
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

# -------------------------------------------------------------------------------
# 1. Identify site type (24d tetrahedral vs 96h octahedral) and assign energies
# -------------------------------------------------------------------------------
# Energy reference: octahedral (96h) = 0.0 eV, tetrahedral (24d) = -0.15 eV (more stable)
site_energies = []  # one entry per site in the structure (ordered as structure)
for site in structure:
    # Wyckoff label is often stored in the site properties of a CIF
    wyckoff = site.properties.get('wyckoff') or site.properties.get('label') or ''
    wyckoff = wyckoff.lower()
    if '24d' in wyckoff:
        site_energies.append(-0.15)  # tetrahedral, lower energy
    else:
        # default to octahedral (96h) if not explicitly tetrahedral
        site_energies.append(0.0)
site_energies = np.array(site_energies)

# -------------------------------------------------------------------------------
# 2. Initialise Li sites with Boltzmann‑weighted occupancy probabilities
# -------------------------------------------------------------------------------
kb = 8.617e-5  # eV/K
T_init = 298.0
boltz_factors = np.exp(-site_energies / (kb * T_init))
# Simple probability model: p_i = w_i / (w_i + 1)  (vacancy energy taken as 0)
probabilities = boltz_factors / (boltz_factors + 1.0)

initial_sites = []
for idx, site in enumerate(structure):
    if "Li" in [s.symbol for s in site.species.elements]:
        prob = probabilities[idx]
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
            distance = np.linalg.norm(cart_disp)
            neighbors.append((nb.index, cart_disp, distance))
    adj_list[i] = neighbors

print(f"Graph built (cutoff={cutoff}A)")

# === 3. kMC Simulator (BKL Algorithm) ===
class KMCSimulator:
    def __init__(self, structure, adj_list, initial_sites, params, site_energies):
        self.params = params
        self.adj_list = adj_list
        self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        self.site_energies = site_energies
        
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
        
        self.kb = 8.617e-5  # eV/K
        self.T = params['T']
        # attempt frequency (may include vibrational entropy prefactor)
        self.nu0 = params['nu']
        self.S_vib = params.get('S_vib', 0.0)  # vibrational entropy (eV/K)
        self.nu = self.nu0 * np.exp(self.S_vib / self.kb)
        
        # Base barriers (eV) for different hop geometries – simple distance based classification
        self.short_dist_barrier = 0.35  # tetrahedral ↔ octahedral (shorter hop)
        self.long_dist_barrier = 0.55   # octahedral ↔ octahedral (longer hop)
        self.repulsion_penalty = 0.05   # eV per occupied neighboring Li
        self.collective_barrier_offset = -0.10  # eV (makes collective moves easier)
        
        # ----- Phonon‑related parameters (frozen‑framework relaxation) -----
        # RMS displacement of framework atoms at temperature T (Å).  A typical value for LLZO at 300 K is ~0.05 Å.
        self.rms_disp = params.get('rms_disp', 0.05)
        # Coupling constant converting a displacement into a barrier reduction (eV/Å).
        self.alpha = params.get('alpha', 0.10)
        # Constant phonon‑induced reduction factor λ (dimensionless).
        self.lambda_phonon = params.get('lambda_phonon', 0.25)
        
    def _phonon_barrier_reduction(self):
        """Return a stochastic barrier reduction due to instantaneous lattice vibrations.
        The reduction consists of a deterministic term λ·kB·T and a random term α·Δu,
        where Δu is drawn from a Gaussian with σ = rms_disp.
        """
        deterministic = self.lambda_phonon * self.kb * self.T
        delta_u = np.random.normal(0.0, self.rms_disp)
        stochastic = self.alpha * delta_u
        return deterministic + stochastic

    def _local_barrier(self, src, tgt, distance):
        base = self.short_dist_barrier if distance < 2.5 else self.long_dist_barrier
        occ_src = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(src, []))
        occ_tgt = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(tgt, []))
        repulsion = self.repulsion_penalty * (occ_src + occ_tgt)
        site_energy_diff = self.site_energies[tgt] - self.site_energies[src]
        barrier_static = base + repulsion + site_energy_diff
        # Apply phonon‑assisted reduction
        barrier_eff = barrier_static - self._phonon_barrier_reduction()
        return max(barrier_eff, 0.0)  # barrier cannot be negative

    def _collective_barrier(self, i, j, v, dist_i_v, dist_j_i):
        base = self.short_dist_barrier if min(dist_i_v, dist_j_i) < 2.5 else self.long_dist_barrier
        occ_i = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(i, []))
        occ_j = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(j, []))
        occ_v = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(v, []))
        repulsion = self.repulsion_penalty * (occ_i + occ_j + occ_v)
        site_energy_change = self.site_energies[v] - self.site_energies[j]
        barrier_static = base + repulsion + site_energy_change + self.collective_barrier_offset
        barrier_eff = barrier_static - self._phonon_barrier_reduction()
        return max(barrier_eff, 0.0)

    def run_step(self):
        events = []          # (type, data, rate, cum_rate)
        cum_rates = []
        total_rate = 0.0
        # ----- Single‑particle hops -----
        for src in self.li_indices:
            for tgt, disp, dist in self.adj_list[src]:
                if self.occupancy[tgt] == 0:  # vacancy
                    E = self._local_barrier(src, tgt, dist)
                    rate = self.nu * np.exp(-E / (self.kb * self.T))
                    if rate > 0:
                        events.append(("single", (src, tgt, disp), rate))
                        total_rate += rate
                        cum_rates.append(total_rate)
        # ----- Concerted moves -----
        for i in self.li_indices:
            for v, disp_iv, dist_i_v in self.adj_list[i]:
                if self.occupancy[v] == 0:
                    for j in self.li_indices:
                        if j == i:
                            continue
                        # look for a second vacancy that is a neighbour of a Li
                        for w, disp_jw, dist_j_w in self.adj_list[j]:
                            if self.occupancy[w] == 0:
                                # simple Miller‑Abrahams style condition (optional)
                                delta_E = 0.0  # placeholder – already embedded in barrier
                                if delta_E < 0:
                                    rate = self.nu
                                else:
                                    rate = self.nu * np.exp(-delta_E / (self.kb * self.T))
                                # Use collective barrier
                                E_coll = self._collective_barrier(i, j, w, dist_i_v, dist_j_w)
                                rate = self.nu * np.exp(-E_coll / (self.kb * self.T))
                                if rate > 0:
                                    events.append(("collective", (i, j, w, disp_iv, disp_jw), rate))
                                    total_rate += rate
                                    cum_rates.append(total_rate)
        if not events:
            return False
        # Choose an event
        r = np.random.random() * total_rate
        idx = np.searchsorted(cum_rates, r)
        ev_type, ev_data, ev_rate = events[idx][0], events[idx][1], events[idx][2]
        # Execute the chosen event
        if ev_type == "single":
            src, tgt, disp = ev_data
            self.occupancy[src] = 0
            self.occupancy[tgt] = 1
            pid = self.site_to_particle.pop(src)
            self.site_to_particle[tgt] = pid
            self.particle_positions[pid]['current'] += disp
            self.li_indices.remove(src)
            self.li_indices.add(tgt)
        elif ev_type == "collective":
            i, j, v, disp_i, disp_j = ev_data
            # move i -> v, j -> i, v -> j (simple cyclic permutation)
            pid_i = self.site_to_particle[i]
            pid_j = self.site_to_particle[j]
            pid_v = self.site_to_particle[v]
            self.site_to_particle[v] = pid_i
            self.site_to_particle[i] = pid_j
            self.site_to_particle[j] = pid_v
            self.particle_positions[pid_i]['current'] += disp_i
            self.particle_positions[pid_j]['current'] += disp_j
            # update occupancy set
            self.li_indices = set(self.site_to_particle.keys())
        self.current_time += 1.0 / total_rate
        self.step_count += 1
        return True

    def run(self, nsteps=1000):
        for _ in range(nsteps):
            if not self.run_step():
                break
        print(f"Completed {self.step_count} kMC steps. Total simulated time = {self.current_time:.3e} s")

# -------------------------------------------------------------------------------
# 4. Simulation parameters (including phonon‑related values)
# -------------------------------------------------------------------------------
sim_params = {
    "T": 298.0,               # temperature (K)
    "nu": 1e13,               # base attempt frequency (Hz)
    "S_vib": 0.0,             # vibrational entropy prefactor (eV/K), set to 0 for now
    "rms_disp": 0.05,         # RMS displacement of framework atoms (Å)
    "alpha": 0.10,            # barrier‑displacement coupling (eV/Å)
    "lambda_phonon": 0.25,    # dimensionless phonon reduction factor λ
    # optional: "S_vib": 0.02  # example vibrational entropy term
}

kmc = KMCSimulator(structure, adj_list, initial_sites, sim_params, site_energies)
kmc.run(nsteps=5000)

# -------------------------------------------------------------------------------
# 5. Post‑processing (conductivity estimate remains unchanged – uses original formula)
# -------------------------------------------------------------------------------
# The original script prints a conductivity estimate via the placeholder formula at the end.
# No further changes are required for this hypothesis.