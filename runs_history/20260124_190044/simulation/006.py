"""
KOKOA Simulation #6
Generated: 2026-01-24 19:18:10
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/홍성은/KOKOA"
_CIF_PATH = "C:/Users/홍성은/KOKOA/LLZO_with_vacancy.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/홍성은/KOKOA/runs/20260124_190044')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import os

    # === 1. Structure Loading and Symmetry Analysis ===
    unit_structure = Structure.from_file(_CIF_PATH)
    n_unit = len(unit_structure)

    # Identify Wyckoff sites for the Li-sublattice
    temp_struct = unit_structure.copy()
    for i in range(len(temp_struct)):
        if temp_struct[i].species_string == "He":
            temp_struct.replace(i, "Li")

    try:
        sga = SpacegroupAnalyzer(temp_struct, symprec=0.1)
        ds = sga.get_symmetry_dataset()
        wyckoffs_unit = ds['wyckoffs']
    except:
        wyckoffs_unit = ['h'] * n_unit

    # Expand to supercell
    structure = unit_structure.copy()
    structure.make_supercell([3, 3, 3])
    vol_ang = structure.volume
    num_total_atoms = len(structure)

    # Identify Li/Vacancy sites in the supercell
    li_indices = [i for i, site in enumerate(structure) if site.species_string in ["Li", "He"]]
    num_sites = len(li_indices)
    site_map = {old_idx: new_idx for new_idx, old_idx in enumerate(li_indices)}
    inv_site_map = {new_idx: old_idx for new_idx, old_idx in enumerate(li_indices)}

    # Map Wyckoff labels to supercell sites
    wyckoffs_super = [wyckoffs_unit[i % n_unit] for i in range(num_total_atoms)]

    # === 2. Energy Model Parameters ===
    # Site energies (24d is more stable than 96h)
    site_energies_static = np.array([0.0 if 'd' in wyckoffs_super[inv_site_map[i]] else 0.15 for i in range(num_sites)])

    # Coulomb interaction matrix (1/r repulsion)
    # V_ij = 14.4 / (epsilon * r_ij) in eV. Using epsilon_r = 50 for LLZO.
    epsilon_r = 50.0
    V_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        old_i = inv_site_map[i]
        # Use a 10.0 A cutoff for Coulomb interactions
        neighbors = structure.get_neighbors(structure[old_i], r=10.0)
        for nb in neighbors:
            if nb.index in site_map:
                j = site_map[nb.index]
                V_matrix[i, j] = 14.4 / (epsilon_r * nb.nn_distance)

    # === 3. Build Adjacency Graph with Path Types ===
    adj_list = {}
    for i in range(num_sites):
        old_i = inv_site_map[i]
        adj_list[i] = []
        # Use a 4.0 A cutoff for hopping
        neighbors = structure.get_neighbors(structure[old_i], r=4.0)
        for nb in neighbors:
            if nb.index in site_map:
                j = site_map[nb.index]
                disp = nb.coords - structure[old_i].coords
            
                # Determine path type for E_base
                w_i = wyckoffs_super[old_i]
                w_j = wyckoffs_super[nb.index]
                if ('d' in w_i and 'h' in w_j) or ('h' in w_i and 'd' in w_j):
                    path_type = 'dh'
                elif ('h' in w_i and 'h' in w_j):
                    path_type = 'hh'
                else:
                    path_type = 'other'
                adj_list[i].append((j, disp, path_type))

    # === 4. kMC Simulator (BKL Algorithm with BEP Barriers) ===
    class KMCSimulator:
        def __init__(self, adj_list, occupancy, site_energies_static, V_matrix):
            self.adj_list = adj_list
            self.occupancy = occupancy.astype(int)
            self.site_energies_static = site_energies_static
            self.V_matrix = V_matrix
        
            self.particle_to_site = {p_id: i for p_id, i in enumerate(np.where(self.occupancy == 1)[0])}
            self.num_particles = len(self.particle_to_site)
            self.displacements = np.zeros((self.num_particles, 3))
            self.current_time = 0.0
            self.steps = 0
        
            self.kB = 8.617333e-05  # eV/K
            self.T = 300.0
            self.nu = 1e12  # Attempt frequency (Hz)

        def get_rates(self):
            events = []
            rates = []
        
            # Precompute current interaction energy at each site
            V_occ = self.V_matrix @ self.occupancy
        
            for p_id, i in self.particle_to_site.items():
                for j, disp, path_type in self.adj_list[i]:
                    if self.occupancy[j] == 0:
                        # Energy change dE = E_final - E_initial
                        # E_initial = E_static_i + sum_{k!=i} V_ik * occ_k
                        # E_final = E_static_j + sum_{k!=i} V_jk * occ_k
                        dE = (self.site_energies_static[j] + V_occ[j] - self.V_matrix[j, i]) - \
                             (self.site_energies_static[i] + V_occ[i])
                    
                        # BEP Relation: E_a = E_base + max(0, dE)
                        if path_type == 'dh':
                            E_base = 0.18
                        elif path_type == 'hh':
                            E_base = 0.22
                        else:
                            E_base = 0.25
                    
                        E_a = E_base + max(0.0, dE)
                        rate = self.nu * np.exp(-E_a / (self.kB * self.T))
                    
                        events.append((p_id, i, j, disp))
                        rates.append(rate)
            return events, np.array(rates)

        def step():
            events, rates = self.get_rates()
            total_rate = np.sum(rates)
            if total_rate == 0:
                return False
        
            # Time increment
            dt = -np.log(np.random.random()) / total_rate
            self.current_time += dt
        
            # Select event
            r = np.random.random() * total_rate
            cum_rates = np.cumsum(rates)
            idx = np.searchsorted(cum_rates, r)
            p_id, i, j, disp = events[idx]
        
            # Update state
            self.occupancy[i] = 0
            self.occupancy[j] = 1
            self.particle_to_site[p_id] = j
            self.displacements[p_id] += disp
            self.steps += 1
            return True

        def run(self, target_time):
            while self.current_time < target_time:
                success = self.step()
                if not success: break
                if self.steps % 2000 == 0:
                    print(f"Step {self.steps} | Time: {self.current_time:.3e} s")
                if self.steps > 100000: break # Safety break

    # === 5. Execution and Analysis ===
    initial_occ = np.array([1 if structure[inv_site_map[i]].species_string == "Li" else 0 for i in range(num_sites)])
    sim = KMCSimulator(adj_list, initial_occ, site_energies_static, V_matrix)

    target_time = 5e-09
    sim.run(target_time)

    # Conductivity Calculation
    # sigma = (e^2 / (V * kB * T)) * (MSD / 6t)
    # Conversion factor for A^3, s, A^2 to S/cm is approx 6.198e-10
    msd = np.sum(np.sum(sim.displacements**2, axis=1))
    if sim.current_time > 0:
        diffusivity = msd / (6.0 * sim.current_time * sim.num_particles) # A^2/s
        # Using the derived factor for sigma
        conductivity = (6.198e-10 / vol_ang) * (msd / (6.0 * sim.current_time))
    else:
        conductivity = 0.0

    print(f"Conductivity: {conductivity} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
