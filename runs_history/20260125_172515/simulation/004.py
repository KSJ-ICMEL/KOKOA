"""
KOKOA Simulation #4
Generated: 2026-01-25 17:27:20
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260125_172515')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import os
    import numpy as np
    from pymatgen.core import Structure
    import matplotlib.pyplot as plt

    # ------------------- User‑provided inputs -------------------
    # Path to the CIF file (injected at runtime)
    _CIF_PATH = _CIF_PATH  # placeholder, will be replaced by the execution environment

    TARGET_TIME = 5e-09  # seconds
    SUPER_CELL = [3, 3, 3]
    CUTOFF = 4.0  # Å, neighbour search radius for possible hops
    TEMPERATURE = 300.0  # K
    KB_EV = 8.617e-5  # eV/K
    KB_J = KB_EV * 1.602176634e-19  # J/K
    NU = 1e13  # attempt frequency, s^-1
    E0 = 0.40  # eV, base migration barrier
    ALPHA = 0.05  # eV per occupied neighbour

    # Site‑energy values (eV)
    E_LOW = 0.0      # 24d tetrahedral sites
    E_HIGH = 0.15    # 96h octahedral sites

    # ------------------- 1. Load structure -------------------
    if not os.path.isfile(_CIF_PATH):
        raise FileNotFoundError(f"CIF file not found: {_CIF_PATH}")

    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(SUPER_CELL)
    print(f"Supercell built: {SUPER_CELL}, total atoms: {len(structure)}")

    # ------------------- 2. Identify Li sites -------------------
    li_site_indices = []
    for i, site in enumerate(structure):
        if "Li" in [el.symbol for el in site.species.elements]:
            li_site_indices.append(i)
    li_site_indices = np.array(li_site_indices)
    num_li_sites = len(li_site_indices)
    print(f"Number of Li sites: {num_li_sites}")

    # ------------------- 3. Assign site energies -------------------
    # Try to use Wyckoff label if present; otherwise fall back to coordination‑based heuristic
    site_energies = np.zeros(num_li_sites)
    for idx, site_idx in enumerate(li_site_indices):
        site = structure[site_idx]
        wyck = site.properties.get('wyckoff') if hasattr(site, 'properties') else None
        if wyck and '24d' in wyck:
            site_energies[idx] = E_LOW
        elif wyck and '96h' in wyck:
            site_energies[idx] = E_HIGH
        else:
            # heuristic: count Li neighbours within 2.5 Å (tetrahedral ≈4 neighbours)
            neigh = structure.get_neighbors(site, 2.5)
            li_neigh = sum(1 for n in neigh if "Li" in [el.symbol for el in n.species.elements])
            site_energies[idx] = E_LOW if li_neigh <= 4 else E_HIGH

    # ------------------- 4. Build adjacency list (possible hops) -------------------
    adj_list = {i: [] for i in li_site_indices}
    # also store neighbour list for barrier term (undirected)
    neighbour_map = {i: [] for i in li_site_indices}

    all_neighbors = structure.get_all_neighbors(r=CUTOFF)
    for i, site in enumerate(structure):
        if i not in li_site_indices:
            continue
        for nb in all_neighbors[i]:
            j = nb.index
            if j not in li_site_indices:
                continue
            # displacement vector (cartesian) taking periodic image into account
            frac_diff = structure[j].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adj_list[i].append((j, cart_disp))
            neighbour_map[i].append(j)

    print("Adjacency list constructed.")

    # ------------------- 5. Initialise occupancy (Boltzmann‑weighted) -------------------
    # Desired overall Li occupancy ~50 % (i.e. half of the Li sites are vacant)
    desired_occupied = num_li_sites // 2
    # Boltzmann weights for each site type
    boltz_weights = np.exp(-site_energies / (KB_EV * TEMPERATURE))
    probabilities = boltz_weights / boltz_weights.sum()
    initial_occupied_indices = np.random.choice(num_li_sites, size=desired_occupied, replace=False, p=probabilities)
    occupancy = np.zeros(num_li_sites, dtype=int)
    occupancy[initial_occupied_indices] = 1

    # Mapping from global site index to position in occupancy array
    site_to_occpos = {site_idx: pos for pos, site_idx in enumerate(li_site_indices)}
    # Mapping from occupied site -> particle id and vice‑versa
    particle_id_counter = 0
    site_to_particle = {}
    particle_to_site = {}
    particle_positions = {}
    for pos, occ in enumerate(occupancy):
        if occ:
            global_idx = li_site_indices[pos]
            pid = particle_id_counter
            particle_id_counter += 1
            site_to_particle[global_idx] = pid
            particle_to_site[pid] = global_idx
            # store initial Cartesian position
            particle_positions[pid] = structure.lattice.get_cartesian_coords(structure[global_idx].frac_coords)

    # Pre‑compute occupied neighbour counts for barrier term
    occupied_neighbour_counts = np.zeros(num_li_sites, dtype=int)
    for pos, site_idx in enumerate(li_site_indices):
        count = 0
        for nb in neighbour_map[site_idx]:
            nb_pos = site_to_occpos[nb]
            count += occupancy[nb_pos]
        occupied_neighbour_counts[pos] = count

    # ------------------- 6. Kinetic Monte Carlo loop -------------------
    time = 0.0
    step = 0
    while time < TARGET_TIME:
        events = []  # each entry: (origin_global, target_global, rate, disp, particle_id)
        total_rate = 0.0
        for pos, occ in enumerate(occupancy):
            if not occ:
                continue
            origin = li_site_indices[pos]
            pid = site_to_particle[origin]
            # number of occupied neighbours for barrier term
            n_occ_nb = occupied_neighbour_counts[pos]
            for target, disp in adj_list[origin]:
                target_pos = site_to_occpos[target]
                if occupancy[target_pos]:
                    continue  # target already occupied
                # migration barrier
                delta_E = E0 + ALPHA * n_occ_nb + (site_energies[target_pos] - site_energies[pos])
                rate = NU * np.exp(-delta_E / (KB_EV * TEMPERATURE))
                if rate > 0:
                    events.append((origin, target, rate, disp, pid))
                    total_rate += rate
        if total_rate == 0.0:
            print("No more possible events; terminating early.")
            break
        # Choose event
        r = np.random.rand() * total_rate
        cumulative = 0.0
        for origin, target, rate, disp, pid in events:
            cumulative += rate
            if r <= cumulative:
                chosen = (origin, target, disp, pid)
                break
        # Advance time
        dt = -np.log(np.random.rand()) / total_rate
        time += dt
        step += 1
        # Execute hop
        origin, target, disp, pid = chosen
        origin_pos = site_to_occpos[origin]
        target_pos = site_to_occpos[target]
        # update occupancy
        occupancy[origin_pos] = 0
        occupancy[target_pos] = 1
        # update particle‑site maps
        del site_to_particle[origin]
        site_to_particle[target] = pid
        particle_to_site[pid] = target
        # update particle Cartesian position (apply displacement with periodic wrap)
        particle_positions[pid] = particle_positions[pid] + disp
        # Update occupied neighbour counts for origin, target and their neighbours
        affected_sites = set([origin, target])
        affected_sites.update(neighbour_map[origin])
        affected_sites.update(neighbour_map[target])
        for s in affected_sites:
            s_pos = site_to_occpos[s]
            count = 0
            for nb in neighbour_map[s]:
                nb_pos = site_to_occpos[nb]
                count += occupancy[nb_pos]
            occupied_neighbour_counts[s_pos] = count
        # Progress report
        if step % 2000 == 0:
            print(f"Step {step}, time {time:.3e} s, total_rate {total_rate:.3e} s⁻¹")

    print(f"KMC finished after {step} steps, total simulated time {time:.3e} s")

    # ------------------- 7. Compute conductivity -------------------
    # Compute mean‑squared displacement (MSD)
    final_positions = np.array(list(particle_positions.values()))
    initial_positions = np.array([structure.lattice.get_cartesian_coords(structure[particle_to_site[pid]].frac_coords) for pid in particle_positions.keys()])
    displacements = final_positions - initial_positions
    msd = np.mean(np.sum(displacements**2, axis=1))
    # Diffusion coefficient (Einstein relation)
    D = msd / (6.0 * time)  # Å²/s
    # Convert D to cm²/s (1 Å = 1e-8 cm)
    D_cm2_s = D * 1e-16
    # Li concentration (number per cm³)
    volume_cm3 = structure.lattice.volume * 1e-24  # Å³ → cm³
    num_li = occupancy.sum()
    conc = num_li / volume_cm3  # cm⁻³
    # Conductivity σ = D·c·q²/(kB·T)
    q = 1.602176634e-19  # C
    sigma = D_cm2_s * conc * q**2 / (KB_J * TEMPERATURE)  # S/cm
    print(f"Conductivity: {sigma:.3e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
