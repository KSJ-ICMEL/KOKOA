"""
KOKOA Simulation #2
Generated: 2026-01-24 18:37:35
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/홍성은/KOKOA"
_CIF_PATH = "C:/Users/홍성은/KOKOA/LLZO_with_vacancy.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/홍성은/KOKOA/runs/20260124_183016')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import os
    import scipy.constants as const

    # === 1. Configuration and Constants ===
    # _CIF_PATH is injected at runtime
    cif_path = _CIF_PATH
    target_time = 5e-09
    nu = 1e13  # Attempt frequency (Hz)
    Ea = 0.3   # Base migration barrier (eV)
    T = 300    # Temperature (K)
    kB_eV = 8.617333262145e-5
    kB_J = const.k
    e_charge = const.e

    # === 2. Structure Loading and Site Labeling ===
    structure = Structure.from_file(cif_path)

    # Identify Wyckoff sites in the unit cell before expansion
    # We temporarily treat all Li/He sites as Li for symmetry analysis
    temp_struct = structure.copy()
    for site in temp_struct:
        if site.species_string == "He":
            site.species = "Li"

    sga = SpacegroupAnalyzer(temp_struct, symprec=0.1)
    sym_struct = sga.get_symmetrized_structure()
    unit_cell_labels = [sym_struct.get_wyckoff_label(i) for i in range(len(structure))]

    # Expand to 3x3x3 supercell
    structure.make_supercell([3, 3, 3])
    num_unit_cells = 3 * 3 * 3
    supercell_labels = unit_cell_labels * num_unit_cells

    # Assign Site Energies
    site_energies = []
    for label in supercell_labels:
        if "24d" in label:
            site_energies.append(0.0)
        elif "96h" in label:
            site_energies.append(0.12)
        else:
            site_energies.append(0.2)  # Fallback for other sites

    # === 3. Initialize kMC Lattice ===
    occupancy = np.zeros(len(structure), dtype=int)
    site_to_particle = {}
    particle_to_site = {}
    particle_displacements = {}
    p_id_counter = 0

    for i, site in enumerate(structure):
        if site.species_string == "Li":
            occupancy[i] = 1
            site_to_particle[i] = p_id_counter
            particle_to_site[p_id_counter] = i
            particle_displacements[p_id_counter] = np.zeros(3)
            p_id_counter += 1
        elif site.species_string == "He":
            occupancy[i] = 0

    # === 4. Build Adjacency Graph ===
    cutoff = 3.2  # Angstroms
    adj_list = [[] for _ in range(len(structure))]
    neighbors_data = structure.get_all_neighbors(r=cutoff)

    for i in range(len(structure)):
        if structure[i].species_string not in ["Li", "He"]:
            continue
        for nb in neighbors_data[i]:
            if structure[nb.index].species_string in ["Li", "He"]:
                # Calculate cartesian displacement including PBC
                frac_diff = structure[nb.index].frac_coords - structure[i].frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                adj_list[i].append({
                    'to': nb.index,
                    'dist': cart_disp,
                    'energy_j': site_energies[nb.index]
                })

    # === 5. kMC Simulation (BKL Algorithm) ===
    current_time = 0.0
    steps = 0

    while current_time < target_time:
        rates = []
        hops = []
    
        # Calculate all possible rates
        for p_id, site_i in particle_to_site.items():
            Ei = site_energies[site_i]
            for neighbor in adj_list[site_i]:
                site_j = neighbor['to']
                if occupancy[site_j] == 0:
                    Ej = neighbor['energy_j']
                    # Transition State Theory rate with site energy difference
                    barrier = Ea + max(0, Ej - Ei)
                    rate = nu * np.exp(-barrier / (kB_eV * T))
                    rates.append(rate)
                    hops.append((p_id, site_i, site_j, neighbor['dist']))
    
        total_rate = sum(rates)
        if total_rate == 0:
            break
        
        # Time step
        dt = -np.log(np.random.random()) / total_rate
        current_time += dt
    
        # Select hop
        r = np.random.random() * total_rate
        idx = np.searchsorted(np.cumsum(rates), r)
        p_id, s_i, s_j, dist = hops[idx]
    
        # Update state
        occupancy[s_i] = 0
        occupancy[s_j] = 1
        del site_to_particle[s_i]
        site_to_particle[s_j] = p_id
        particle_to_site[p_id] = s_j
        particle_displacements[p_id] += dist
    
        steps += 1
        if steps % 2000 == 0:
            print(f"Step {steps}, Time {current_time:.2e} s")

    # === 6. Conductivity Calculation ===
    vol_cm3 = structure.volume * 1e-24
    total_sq_disp_cm2 = sum(np.sum(d**2) for d in particle_displacements.values()) * 1e-16

    if current_time > 0:
        # Einstein-Smoluchowski derived conductivity
        val = (total_sq_disp_cm2 * e_charge**2) / (6 * vol_cm3 * kB_J * T * current_time)
    else:
        val = 0.0

    print(f"Conductivity: {val} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
