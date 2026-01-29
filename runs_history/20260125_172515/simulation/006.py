"""
KOKOA Simulation #6
Generated: 2026-01-25 17:28:39
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
    from scipy.constants import physical_constants, epsilon_0, pi

    # ------------------- User‑provided inputs -------------------
    # Path to the CIF file (injected at runtime)
    _CIF_PATH = _CIF_PATH  # placeholder, will be replaced by the execution environment

    target_time = 5e-9  # seconds
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

    # Coulombic interaction parameters
    Q_E = physical_constants['elementary charge'][0]  # C
    EPS_R = 10.0          # relative dielectric constant (approx.)
    LAMBDA = 5.0          # Å, screening length

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
    site_energies = np.zeros(num_li_sites)
    for idx, site_idx in enumerate(li_site_indices):
        site = structure[site_idx]
        wyck = site.properties.get('wyckoff') if hasattr(site, 'properties') else None
        if wyck and '24d' in wyck:
            site_energies[idx] = E_LOW
        elif wyck and '96h' in wyck:
            site_energies[idx] = E_HIGH
        else:
            # fallback heuristic based on Li coordination
            neigh = structure.get_neighbors(site, 2.5)
            li_neigh = sum(1 for n in neigh if "Li" in [el.symbol for el in n.species.elements])
            site_energies[idx] = E_LOW if li_neigh <= 4 else E_HIGH

    # ------------------- 4. Build adjacency list (possible hops) -------------------
    adj_list = {i: [] for i in li_site_indices}
    # neighbour_map for ALPHA term (undirected)
    neighbour_map = {i: [] for i in li_site_indices}

    all_neighbors = structure.get_all_neighbors(r=CUTOFF)
    for i, site in enumerate(structure):
        if i not in li_site_indices:
            continue
        for nb in all_neighbors[i]:
            j = nb.index
            if j not in li_site_indices:
                continue
            # displacement vector (cartesian) with periodic image
            frac_diff = structure[j].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adj_list[i].append((j, cart_disp))
            neighbour_map[i].append(j)

    print("Adjacency list constructed.")

    # ------------------- 5. Pre‑compute screened Coulomb matrix -------------------
    # Convert positions to Cartesian (Å) for distance calculations
    positions = np.array([structure[i].coords for i in li_site_indices])  # Å
    # Pairwise distance matrix (Å) using periodic images
    # We'll use pymatgen's get_distance with images via get_all_neighbors for each site
    N = num_li_sites
    coulomb_matrix = np.zeros((N, N))
    for a in range(N):
        for b in range(a + 1, N):
            # shortest image distance
            dist = structure.get_distance(li_site_indices[a], li_site_indices[b])  # Å
            if dist < 1e-8:
                continue
            # screened Coulomb (eV)
            e_ij = (Q_E**2) / (4 * pi * epsilon_0 * EPS_R * (dist * 1e-10))  # Joules
            e_ij *= np.exp(-dist / LAMBDA)  # screening
            e_ij_ev = e_ij / Q_E  # convert J to eV (1 e = 1.602e-19 C)
            coulomb_matrix[a, b] = e_ij_ev
            coulomb_matrix[b, a] = e_ij_ev

    # ------------------- 6. Initialise occupancy -------------------
    # In LLZO the Li sublattice is partially occupied. We'll start with the nominal
    # composition Li7.0La3Zr2O12 → 56 Li sites per primitive cell, 7/56 occupied.
    # For the supercell we occupy the same fraction randomly.
    frac_occupied = 7.0 / 56.0
    num_occupied = int(round(frac_occupied * N))
    occupancy = np.zeros(N, dtype=bool)
    occupied_indices = np.random.choice(N, size=num_occupied, replace=False)
    occupancy[occupied_indices] = True

    # Store initial positions of each ion (for MSD)
    initial_positions = positions[occupancy].copy()
    current_positions = initial_positions.copy()

    # Mapping from site index (global) to local Li‑site index
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(li_site_indices)}

    # ------------------- 7. Helper functions -------------------
    def compute_hop_rate(i_global, j_global, occ_vec):
        """Calculate the hopping rate for a move i -> j.
        i_global, j_global are indices in the full structure (as in li_site_indices).
        occ_vec is the current occupancy boolean array (length N).
        """
        i_loc = global_to_local[i_global]
        j_loc = global_to_local[j_global]
        # Base barrier + site‑energy difference
        delta_site = site_energies[j_loc] - site_energies[i_loc]
        # Neighbour term (ALPHA per occupied neighbour of destination)
        neigh_occ = sum(occ_vec[global_to_local[n]] for n in neighbour_map[j_global] if n in global_to_local)
        neighbor_term = ALPHA * neigh_occ
        # Coulombic contribution
        # Remove interactions of i with all other occupied sites, add interactions of j
        coulomb_delta = 0.0
        for k_loc, occ in enumerate(occ_vec):
            if not occ or k_loc == i_loc or k_loc == j_loc:
                continue
            coulomb_delta += coulomb_matrix[j_loc, k_loc] - coulomb_matrix[i_loc, k_loc]
        # Total barrier (eV)
        barrier = E0 + delta_site + neighbor_term + coulomb_delta
        if barrier < 0:
            barrier = 0.0
        rate = NU * np.exp(-barrier / (KB_EV * TEMPERATURE))
        return rate

    # Pre‑compute rates for all possible hops (i->j) where i is occupied and j is vacant
    hop_list = []  # each element: (i_global, j_global, disp_vector)
    for i_global in li_site_indices:
        i_loc = global_to_local[i_global]
        if not occupancy[i_loc]:
            continue
        for j_global, disp in adj_list[i_global]:
            j_loc = global_to_local[j_global]
            if occupancy[j_loc]:
                continue
            hop_list.append((i_global, j_global, disp))

    # Compute initial rates
    rates = np.array([compute_hop_rate(i, j, occupancy) for i, j, _ in hop_list])

    # ------------------- 8. Kinetic Monte Carlo loop -------------------
    time = 0.0
    step = 0
    msd = 0.0
    while time < target_time:
        total_rate = rates.sum()
        if total_rate == 0:
            print("No more possible hops. Stopping simulation.")
            break
        # Choose event
        r1 = np.random.rand()
        cum_rates = np.cumsum(rates)
        event_idx = np.searchsorted(cum_rates, r1 * total_rate)
        i_global, j_global, disp = hop_list[event_idx]
        i_loc = global_to_local[i_global]
        j_loc = global_to_local[j_global]
        # Update time
        r2 = np.random.rand()
        dt = -np.log(r2) / total_rate
        time += dt
        # Update occupancy
        occupancy[i_loc] = False
        occupancy[j_loc] = True
        # Update positions for MSD (track the ion that moved)
        # Find which ion in current_positions corresponds to i_loc
        ion_idx = np.where((initial_positions == positions[i_loc]).all(axis=1))[0]
        if ion_idx.size == 0:
            # This can happen if the ion moved previously; locate by matching current position
            ion_idx = np.where((current_positions == positions[i_loc]).all(axis=1))[0]
        if ion_idx.size > 0:
            idx = ion_idx[0]
            # Apply displacement (including periodic wrap)
            current_positions[idx] += disp
        # Re‑build hop list and rates locally (only hops involving i or j change)
        # Remove hops that are no longer valid and add new ones
        new_hop_list = []
        new_rates = []
        for k, (src, dst, dvec) in enumerate(hop_list):
            src_occ = occupancy[global_to_local[src]]
            dst_occ = occupancy[global_to_local[dst]]
            if src_occ and not dst_occ:
                new_hop_list.append((src, dst, dvec))
                new_rates.append(compute_hop_rate(src, dst, occupancy))
            # else discard
        # Add hops that become possible because i became vacant and j became occupied
        # i is now vacant -> any occupied neighbour can hop into i
        for nbr in neighbour_map[i_global]:
            if nbr not in global_to_local:
                continue
            nbr_loc = global_to_local[nbr]
            if occupancy[nbr_loc]:
                # hop nbr -> i
                disp_vec = -np.array([v for s, v in adj_list[nbr] if s == i_global][0])
                new_hop_list.append((nbr, i_global, disp_vec))
                new_rates.append(compute_hop_rate(nbr, i_global, occupancy))
        # j is now occupied -> hops from j to its vacant neighbours become possible
        for nbr in neighbour_map[j_global]:
            if nbr not in global_to_local:
                continue
            nbr_loc = global_to_local[nbr]
            if not occupancy[nbr_loc]:
                # hop j -> nbr
                disp_vec = np.array([v for s, v in adj_list[j_global] if s == nbr][0])
                new_hop_list.append((j_global, nbr, disp_vec))
                new_rates.append(compute_hop_rate(j_global, nbr, occupancy))
        hop_list = new_hop_list
        rates = np.array(new_rates)
        step += 1
        if step % 2000 == 0:
            print(f"Step {step}, time {time:.3e} s, total_rate {total_rate:.3e} s⁻¹")

    # ------------------- 9. Post‑processing -------------------
    # Compute mean‑squared displacement (Å²)
    if current_positions.shape[0] > 0:
        displacements = current_positions - initial_positions
        msd = np.mean(np.sum(displacements**2, axis=1))
    else:
        msd = 0.0
    # Diffusion coefficient (cm²/s)
    D_cm2_s = msd * 1e-16 / (6 * time)  # Å² -> cm² (1 Å = 1e-8 cm)
    # Number density of mobile Li (cm⁻³)
    volume_cm3 = structure.lattice.volume * 1e-24  # Å³ -> cm³
    n_density = occupancy.sum() / volume_cm3
    # Conductivity (S/cm) using Nernst‑Einstein relation
    sigma = n_density * (Q_E**2) * D_cm2_s / (KB_J * TEMPERATURE)
    print(f"Conductivity: {sigma:.3e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
