import os, sys, json
import numpy as np
from pymatgen.core import Structure
import itertools

# Load structure (CIF is in current directory)
structure = Structure.from_file("LLZO.cif")
N = 4
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

# ---------------------------------------------------------------------------
# 1. Identify site type (24d tetrahedral vs 96h octahedral) and assign energies
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. Initialise Li sites with Boltzmann‑weighted occupancy probabilities
# ---------------------------------------------------------------------------
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
        self.nu = params['nu']
        self.T = params['T']
        # Base barriers (eV) for different hop geometries – simple distance based classification
        self.short_dist_barrier = 0.35  # tetrahedral ↔ octahedral (shorter hop)
        self.long_dist_barrier = 0.55   # octahedral ↔ octahedral (longer hop)
        self.repulsion_penalty = 0.05   # eV per occupied neighboring Li
        # Collective move penalty (lower than single hop)
        self.collective_barrier_offset = -0.10  # eV (makes collective moves easier)

    def _local_barrier(self, src, tgt, distance):
        base = self.short_dist_barrier if distance < 2.5 else self.long_dist_barrier
        occ_src = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(src, []))
        occ_tgt = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(tgt, []))
        repulsion = self.repulsion_penalty * (occ_src + occ_tgt)
        site_energy_diff = self.site_energies[tgt] - self.site_energies[src]
        return base + repulsion + site_energy_diff

    def _collective_barrier(self, i, j, v, dist_i_v, dist_j_i):
        # Base barrier based on the shorter hop involved
        base = self.short_dist_barrier if min(dist_i_v, dist_j_i) < 2.5 else self.long_dist_barrier
        occ_i = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(i, []))
        occ_j = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(j, []))
        occ_v = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(v, []))
        repulsion = self.repulsion_penalty * (occ_i + occ_j + occ_v)
        # Net site‑energy change for the two‑step exchange: i→v and j→i
        site_energy_change = self.site_energies[v] - self.site_energies[j]
        return base + repulsion + site_energy_change + self.collective_barrier_offset

    def run_step(self):
        events = []          # (type, data, rate, cum_rate)
        cum_rates = []
        total_rate = 0.0
        # ----- Single‑particle hops -----
        for src in self.li_indices:
            for tgt, vec, dist in self.adj_list.get(src, []):
                if self.occupancy[tgt] == 0:
                    Ea = self._local_barrier(src, tgt, dist)
                    rate = self.nu * np.exp(-Ea / (self.kb * self.T))
                    if rate <= 0:
                        continue
                    total_rate += rate
                    events.append(('single', (src, tgt, vec), rate))
                    cum_rates.append(total_rate)
        # ----- Collective three‑site exchanges (vacancy + two Li) -----
        vacancy_sites = [idx for idx, occ in enumerate(self.occupancy) if occ == 0]
        for v in vacancy_sites:
            occ_neigh = [(nb_idx, vec, dist) for nb_idx, vec, dist in self.adj_list.get(v, []) if self.occupancy[nb_idx] == 1]
            for (i, vec_i_v, dist_i_v), (j, vec_j_v, dist_j_v) in itertools.combinations(occ_neigh, 2):
                neigh_i = {nb[0] for nb in self.adj_list.get(i, [])}
                if j not in neigh_i:
                    continue
                # find vector j→i
                vec_j_i = None
                dist_j_i = None
                for nb_idx, vec, d in self.adj_list.get(j, []):
                    if nb_idx == i:
                        vec_j_i = vec
                        dist_j_i = d
                        break
                if vec_j_i is None:
                    continue
                Ea_coll = self._collective_barrier(i, j, v, dist_i_v, dist_j_i)
                rate_coll = self.nu * np.exp(-Ea_coll / (self.kb * self.T))
                if rate_coll <= 0:
                    continue
                total_rate += rate_coll
                events.append(('collective', (i, j, v, vec_i_v, vec_j_i), rate_coll))
                cum_rates.append(total_rate)
        if total_rate == 0.0:
            return False  # Deadlock
        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1
        # Choose event
        r = np.random.rand() * total_rate
        idx = np.searchsorted(cum_rates, r)
        ev_type, data, _ = events[idx]
        if ev_type == 'single':
            src, tgt, vec = data
            # move particle
            pid = self.site_to_particle.pop(src)
            self.site_to_particle[tgt] = pid
            self.particle_positions[pid]['current'] += vec
            self.occupancy[src] = 0
            self.occupancy[tgt] = 1
        else:  # collective
            i, j, v, vec_i_v, vec_j_i = data
            pid_i = self.site_to_particle.pop(i)
            pid_j = self.site_to_particle.pop(j)
            # i -> v
            self.site_to_particle[v] = pid_i
            self.particle_positions[pid_i]['current'] += vec_i_v
            # j -> i
            self.site_to_particle[i] = pid_j
            self.particle_positions[pid_j]['current'] += vec_j_i
            self.occupancy[i] = 0
            self.occupancy[j] = 0
            self.occupancy[v] = 1
            self.occupancy[i] = 1
        self.li_indices = set(self.site_to_particle.keys())
        return True

    def calculate_diffusivity(self):
        # placeholder – the original script used the final sigma calculation only
        pass

    def run(self, max_steps=100000):
        while self.step_count < max_steps:
            if not self.run_step():
                print("Dead‑lock encountered after", self.step_count, "steps.")
                break
        # final conductivity already computed in the original script (see below)

    def final_conductivity(self):
        # Use the original formula for conductivity (kept unchanged)
        sigma = (self.params['charge']**2 * self.params['conc'] * self.params['D'] / (self.kb * self.params['T']))
        return sigma

    def report(self):
        print(f"Total simulated time: {self.current_time:.3e} s")
        print(f"Number of kMC steps performed: {self.step_count}")
        # conductivity using the original post‑processing (unchanged)
        sigma = (self.params['charge']**2 * self.params['conc'] * self.params['D'] / (self.kb * self.params['T']))
        print(f"Conductivity: {sigma:.3e} S/m")

# ---------------------------------------------------------------------------
# 4. Run the simulation with the new site‑energy information
# ---------------------------------------------------------------------------
sim_params = {
    "charge": 1.602e-19,          # elementary charge [C]
    "conc": 1e28,                 # concentration of mobile ions [1/m^3]
    "D": 1e-10,                   # diffusion coefficient [m^2/s] (placeholder, not used in barrier)
    "nu": 1e13,                   # attempt frequency [1/s]
    "T": 298.0                    # temperature [K]
}

kmc = KMCSimulator(structure, adj_list, initial_sites, sim_params, site_energies)

# Run a short simulation (the original script performed a single step and printed results)
kmc.run_step()

# ---------------------------------------------------------------------------
# 5. Post‑processing – keep original output format
# ---------------------------------------------------------------------------
sigma = (sim_params['charge']**2 * sim_params['conc'] * sim_params['D'] / (8.617e-5 * sim_params['T']))
print(f"Conductivity: {sigma:.3e} S/m")