"""
KOKOA Simulation #22
Generated: 2026-01-25 18:15:55
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260125_181417')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    '''kMC Simulation for Li-ion Conductivity in Solid Electrolyte with Environment‑Dependent Barriers'''
    import numpy as np
    from pymatgen.core import Structure
    import os

    # === 1. Structure Loading ===
    # Determine CIF path: first try same directory as this script, then its parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cif_path = os.path.join(script_dir, "LLZO.cif")
    if not os.path.isfile(cif_path):
        # fallback to parent directory (e.g., runs/<timestamp>/LLZO.cif)
        parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
        cif_path = os.path.join(parent_dir, "LLZO.cif")
        if not os.path.isfile(cif_path):
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)

    N = 4  # Supercell expansion
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
        
            # Physical constants / parameters
            self.kb = 8.617e-5  # eV/K
            self.nu = params['nu']
            self.E_base = params['E_base']      # base activation energy (eV)
            self.E_neighbor = params['E_neighbor']  # penalty per occupied neighbor (eV)
            self.T = params['T']

        def _occupied_neighbors(self, site_idx):
            """Return number of occupied Li neighbours of a given site (excluding the site itself)."""
            count = 0
            for nb_idx, _ in self.adj_list.get(site_idx, []):
                if nb_idx in self.li_indices and self.occupancy[nb_idx] == 1:
                    count += 1
            return count

        # ... (rest of KMCSimulator implementation) ...

    # Example usage (parameters would be defined elsewhere)
    # params = {'nu': 1e13, 'E_base': 0.5, 'E_neighbor': 0.1, 'T': 300}
    # simulator = KMCSimulator(structure, adj_list, initial_sites, params)
    # simulator.run(steps=10000)
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
