"""
KOKOA Simulation #2
Generated: 2026-01-24 19:08:05
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
    # Load the unit cell first to identify Wyckoff sites
    unit_structure = Structure.from_file(_CIF_PATH)
    n_unit = len(unit_structure)

    # Use a temporary structure with all Li to find symmetry of the Li-sublattice
    temp_struct = unit_structure.copy()
    for i in range(len(temp_struct)):
        if temp_struct[i].species_string == "He":
            temp_struct.replace(i, "Li")

    try:
        sga = SpacegroupAnalyzer(temp_struct, symprec=0.1)
        wyckoffs_unit = sga.get_symmetry_dataset()['wyckoffs']
    except:
        # Fallback if symmetry analysis fails
        wyckoffs_unit = ['h'] * n_unit

    # Expand to supercell
    structure = unit_structure.copy()
    structure.make_supercell([3, 3, 3])
    vol_ang = structure.volume

    # === 2. Initialize Li and Vacancy sites with Energies ===
    initial_sites = []
    site_to_idx = {}  # Map supercell structure index to initial_sites index

    # Site energies based on literature (24d is more stable than 96h)
    # Wyckoff 'd' -> 24d, 'h' -> 96h
    for i, site in enumerate(structure):
        symbol = site.species_string
        if symbol in ["Li", "He"]:
            wyckoff = wyckoffs_unit[i % n_unit]
            energy = 0.2  # Default for 96h (octahedral)
            if 'd' in wyckoff:
                energy = 0.0  # 24d (tetrahedral) is more stable
        
            state = 1 if symbol == "Li" else 0
            initial_sites.append({"coords": site.frac_coords, "state": state, "energy": energy})
            site_to_idx[i] = len(initial_sites) - 1

    # === 3. Build Adjacency Graph ===
    cutoff = 4.0
    neighbors_data = structure.get_all_neighbors(r=cutoff)
    adj_list = {}

    for i, site in enumerate(structure):
        if i not in site_to_idx:
            continue
        src_idx = site_to_idx[i]
        neighbors = []
        for nb in neighbors_data[i]:
            if nb.index in site_to_idx:
                tgt_idx = site_to_idx[nb.index]
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                neighbors.append((tgt_idx, cart_disp))
        adj_list[src_idx] = neighbors

    # === 4. kMC Simulator (BKL Algorithm) ===
    class KMCSimulator:
        def __init__(self, adj_list, initial_sites, params):
            self.params = params
            self.adj_list = adj_list
            self.sites = initial_sites
            self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        
            # Track particles
            self.site_to_particle = {}
            self.particle_to_site = {}
            p_id = 0
            for idx, s in enumerate(initial_sites):
                if s['state'] == 1:
                    self.site_to_particle[idx] = p_id
                    self.particle_to_site[p_id] = idx
                    p_id += 1
        
            self.num_particles = p_id
            self.displacements = np.zeros((self.num_particles, 3))
            self.current_time = 0.0
            self.steps = 0

        def run(self):
            while self.current_time < self.params['target_time']:
                events = []
                total_rate = 0.0
            
                # 1. Calculate rates for all possible hops
                for p_id in range(self.num_particles):
                    src_idx = self.particle_to_site[p_id]
                    for tgt_idx, disp in self.adj_list[src_idx]:
                        if self.occupancy[tgt_idx] == 0:
                            # Metropolis-adjusted barrier
                            dE = self.sites[tgt_idx]['energy'] - self.sites[src_idx]['energy']
                            barrier = self.params['E_a'] + max(0, dE)
                            rate = self.params['nu'] * np.exp(-barrier / (self.params['kB'] * self.params['T']))
                            events.append((p_id, src_idx, tgt_idx, disp, rate))
                            total_rate += rate
            
                if total_rate == 0:
                    break
            
                # 2. Select event and update time
                dt = -np.log(np.random.random()) / total_rate
                self.current_time += dt
            
                r = np.random.random() * total_rate
                cumulative_rate = 0.0
                for p_id, src_idx, tgt_idx, disp, rate in events:
                    cumulative_rate += rate
                    if cumulative_rate >= r:
                        # 3. Execute move
                        self.occupancy[src_idx] = 0
                        self.occupancy[tgt_idx] = 1
                        self.particle_to_site[p_id] = tgt_idx
                        self.site_to_particle[tgt_idx] = p_id
                        del self.site_to_particle[src_idx]
                        self.displacements[p_id] += disp
                        break
            
                self.steps += 1
                if self.steps % 2000 == 0:
                    print(f"Step {self.steps}: Time = {self.current_time:.2e} s")

    # === 5. Execution and Conductivity Calculation ===
    params = {
        'E_a': 0.3,        # Migration barrier (eV)
        'nu': 1e12,        # Attempt frequency (Hz)
        'kB': 8.617e-5,    # Boltzmann constant (eV/K)
        'T': 300,          # Temperature (K)
        'target_time': 5e-09
    }

    sim = KMCSimulator(adj_list, initial_sites, params)
    sim.run()

    # Constants for conductivity
    e = 1.602176634e-19
    kB_J = 1.380649e-23
    T = 300

    msd = np.sum(sim.displacements**2)
    # Conductivity formula: sigma = (e^2 * MSD) / (6 * V * kB * T * t)
    # Convert Angstrom^2 to cm^2 (1e-16) and Angstrom^3 to cm^3 (1e-24)
    if sim.current_time > 0:
        val = (msd * 1e-16 * (e**2)) / (6 * (vol_ang * 1e-24) * kB_J * T * sim.current_time)
    else:
        val = 0.0

    print(f"Conductivity: {val} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
