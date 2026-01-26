import os, sys, json
import numpy as np
from pymatgen.core import Structure
import itertools

# Load structure (CIF is in current directory)
structure = Structure.from_file("LLZO.cif")
N = 4
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
            distance = np.linalg.norm(cart_disp)
            neighbors.append((nb.index, cart_disp, distance))
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
        return base + repulsion

    def _collective_barrier(self, i, j, v, dist_i_v, dist_j_i):
        # Use the shorter of the two hops as a reference
        base = self.short_dist_barrier if min(dist_i_v, dist_j_i) < 2.5 else self.long_dist_barrier
        # Add repulsion from neighbours of the three sites
        occ_i = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(i, []))
        occ_j = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(j, []))
        occ_v = sum(self.occupancy[nb[0]] for nb in self.adj_list.get(v, []))
        repulsion = self.repulsion_penalty * (occ_i + occ_j + occ_v)
        return base + repulsion + self.collective_barrier_offset

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
        # Find all vacancies
        vacancy_sites = [idx for idx, occ in enumerate(self.occupancy) if occ == 0]
        for v in vacancy_sites:
            # occupied neighbours of the vacancy
            occ_neigh = [(nb_idx, vec, dist) for nb_idx, vec, dist in self.adj_list.get(v, []) if self.occupancy[nb_idx] == 1]
            # consider all unordered pairs of occupied neighbours
            for (i, vec_i_v, dist_i_v), (j, vec_j_v, dist_j_v) in itertools.combinations(occ_neigh, 2):
                # check that i and j are also neighbours (forming a triangle)
                neigh_i = {nb[0] for nb in self.adj_list.get(i, [])}
                if j not in neigh_i:
                    continue
                # vector from j to i (needed for second hop)
                vec_j_i = None
                dist_j_i = None
                for nb_idx, vec, d in self.adj_list.get(j, []):
                    if nb_idx == i:
                        vec_j_i = vec
                        dist_j_i = d
                        break
                if vec_j_i is None:
                    continue
                # compute barrier for the collective move
                Ea_coll = self._collective_barrier(i, j, v, dist_i_v, dist_j_i)
                rate_coll = self.nu * np.exp(-Ea_coll / (self.kb * self.T))
                if rate_coll <= 0:
                    continue
                total_rate += rate_coll
                # store data needed for execution: (i, j, v, vec_i_v, vec_j_i)
                events.append(('collective', (i, j, v, vec_i_v, vec_j_i), rate_coll))
                cum_rates.append(total_rate)
        
        if total_rate == 0.0:
            return False  # Deadlock
        # BKL time advance
        self.current_time += -np.log(np.random.rand()) / total_rate
        self.step_count += 1
        # Select event
        r = np.random.rand() * total_rate
        idx = np.searchsorted(cum_rates, r)
        ev_type, data, _ = events[idx]
        if ev_type == 'single':
            src, tgt, vec = data
            # Execute hop
            p_id = self.site_to_particle.pop(src)
            self.particle_positions[p_id]['current'] += vec
            self.occupancy[src], self.occupancy[tgt] = 0, 1
            self.site_to_particle[tgt] = p_id
        else:  # collective
            i, j, v, vec_i_v, vec_j_i = data
            # particle IDs
            p_i = self.site_to_particle.pop(i)
            p_j = self.site_to_particle.pop(j)
            # Update positions
            self.particle_positions[p_i]['current'] += vec_i_v      # i -> vacancy
            self.particle_positions[p_j]['current'] += vec_j_i      # j -> i
            # Update occupancy
            self.occupancy[i] = 1   # now occupied by particle from j
            self.occupancy[j] = 0   # becomes vacancy
            self.occupancy[v] = 1   # now occupied by particle from i
            # Update site‑to‑particle map
            self.site_to_particle[i] = p_j
            self.site_to_particle[v] = p_i
        # Refresh the set of occupied Li sites
        self.li_indices = set(self.site_to_particle.keys())
        return True

    def calculate_properties(self):
        if self.current_time == 0:
            return 0, 0
        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])
        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])
        return msd, sigma

# === 4. Run Simulation ===
sim_params = {'T': 298, 'nu': 1e13, 'volume': structure.volume}
sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

target_time = 1e-6  # stop after ~1 µs (adjustable)
while sim.current_time < target_time:
    if not sim.run_step():
        print('Deadlock reached – terminating simulation')
        break

# Report final conductivity
msd, sigma = sim.calculate_properties()
print(f"Final conductivity: {sigma:.3e} S/m after {sim.current_time:.3e} s and {sim.step_count} steps")