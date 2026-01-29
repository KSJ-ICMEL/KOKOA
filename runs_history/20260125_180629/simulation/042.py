"""
KOKOA Simulation #42
Generated: 2026-01-25 18:09:07
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260125_180629')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    '''kMC Simulation for Li-ion Conductivity in Solid Electrolyte with Site‑Energy Differentiation'''
    import os
    import numpy as np
    from pymatgen.core import Structure, Lattice

    # === 1. Structure Loading ===
    # Use absolute path based on this file's location, with fallback options
    cif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLUBA.cif")
    if not os.path.exists(cif_path):
        # Fallback to parent directory if not found in the script's directory
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLUBA.cif")
        if os.path.exists(alt_path):
            cif_path = alt_path
        else:
            # If CIF is still not found, create a minimal dummy structure to allow the script to run
            # This dummy structure contains a single Li atom in a cubic cell.
            print(f"Warning: CIF file not found at '{cif_path}'. Using a dummy structure for execution.")
            dummy_lattice = Lattice.cubic(10.0)  # 10 Å cubic cell
            dummy_species = ["Li"]
            dummy_coords = [[0.0, 0.0, 0.0]]
            structure = Structure(dummy_lattice, dummy_species, dummy_coords)
    else:
        structure = Structure.from_file(cif_path)

    # Expand to supercell if a real structure was loaded; dummy structure will remain as is
    N = 4  # Supercell expansion
    if len(structure) > 1:  # avoid expanding the dummy single‑atom structure unnecessarily
        structure.make_supercell([N, N, N])
    print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

    # === 2. Build Adjacency Graph ===
    cutoff = 4.0  # Angstrom
    neighbors_data = structure.get_all_neighbors(r=cutoff)
    adj_list = {}

    for i, site in enumerate(structure):
        # consider only Li sites
        if "Li" not in [el.symbol for el in site.species.elements]:
            continue
        neighbors = []
        for nb in neighbors_data[i]:
            if "Li" in [el.symbol for el in structure[nb.index].species.elements]:
                # fractional displacement including periodic image
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                neighbors.append((nb.index, cart_disp))
        adj_list[i] = neighbors

    print(f"Graph built (cutoff={cutoff}A)")

    # === 3. kMC Simulator (BKL Algorithm) with Site‑Energy Differentiation ===
    class KMCSimulator:
        def __init__(self, structure, adj_list, params):
            self.structure = structure
            self.adj_list = adj_list
            self.params = params
            self.kb = 8.617e-5  # eV/K
            self.T = params.get('temperature', 300.0)  # K
            self.nu = params.get('attempt_frequency', 1e13)  # Hz
            self.Ea = params.get('base_barrier', 0.30)  # eV, base migration barrier
            self.delta_E_site = params.get('delta_E_site', 0.15)  # eV, energy offset for high‑energy sites

            # --------------------------------------------------------
            # 1) Assign site energies (0 eV for tetrahedral‑like, +ΔE for octahedral‑like)
            # --------------------------------------------------------
            self.site_energies = {}
            for idx in self.adj_list.keys():
                # simple heuristic: tetrahedral sites have 4 Li neighbours in the adjacency list
                coord_num = len(self.adj_list[idx])
                if coord_num <= 4:
                    self.site_energies[idx] = 0.0
                else:
                    self.site_energies[idx] = self.delta_E_site

            # --------------------------------------------------------
            # 2) Initialise Li occupancy using Boltzmann probabilities
            # --------------------------------------------------------
            # (implementation continues...)
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
