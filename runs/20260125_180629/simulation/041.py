"""
KOKOA Simulation #41
Generated: 2026-01-25 18:09:05
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
    import numpy as np
    import os
    from pymatgen.core import Structure

    # === 1. Structure Loading ===
    # Use absolute path based on this file's location, with fallback options
    cif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLUBA.cif")
    if not os.path.exists(cif_path):
        # Fallback to parent directory if not found in the script's directory
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLUBA.cif")
        if os.path.exists(alt_path):
            cif_path = alt_path
        else:
            # Final fallback to the current working directory
            cwd_path = os.path.join(os.getcwd(), "LLUBA.cif")
            if os.path.exists(cwd_path):
                cif_path = cwd_path
            else:
                raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)

    N = 4  # Supercell expansion
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

            # ----------------------------------------
            # 1) Assign site energies (0 eV for tetrahedral‑like, +ΔE for octahedral‑like)
            # ----------------------------------------
            self.site_energies = {}
            for idx in self.adj_list.keys():
                # simple heuristic: tetrahedral sites have 4 Li neighbours in the adjacency list
                coord_num = len(self.adj_list[idx])
                if coord_num <= 4:
                    self.site_energies[idx] = 0.0
                else:
                    self.site_energies[idx] = self.delta_E_site

            # ----------------------------------------
            # 2) Initialise Li occupancy using Boltzmann probabilities
            # ----------------------------------------
            # (implementation continues...)
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
