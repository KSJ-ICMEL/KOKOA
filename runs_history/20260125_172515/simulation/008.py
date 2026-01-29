"""
KOKOA Simulation #8
Generated: 2026-01-25 17:29:53
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
    from scipy.constants import physical_constants, k, pi
    import matplotlib.pyplot as plt

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

    # Coulombic interaction parameters (used only for optional barrier modulation)
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
            # fallback based on local Li coordination (simple heuristic)
            neigh = structure.get_neighbors(site, 2.5)
            li_neigh = sum(1 for n in neigh if "Li" in [el.symbol for el in n.species.elements])
            site_energies[idx] = E_LOW if li_neigh <= 4 else E_HIGH

    # ------------------- 4. Build adjacency list with displacement vectors -------------------
    adj_list = {i: [] for i in li_site_indices}
    # Pre‑compute all neighbors within cutoff
    all_neighbors = structure.get_all_neighbors(r=CUTOFF)
    for i, site in enumerate(structure):
        if i not in li_site_indices:
            continue
        for nb in all_neighbors[i]:
            j = nb.index
            if j not in li_site_indices:
                continue
            # fractional displacement (including periodic image)
            frac_disp = structure[j].frac_coords - site.frac_coords + nb.image
            # wrap into [-0.5, 0.5) to get the shortest image
            frac_disp = (frac_disp + 0.5) % 1.0 - 0.5
            adj_list[i].append((j, frac_disp))

    # ------------------- 5. Initialise occupancy -------------------
    # Assume all Li sites are initially occupied (one Li per site)
    occupied = {i: True for i in li_site_indices}
    # For simplicity we keep a list of Li ions indexed by a unique id
    li_ids = list(li_site_indices)  # each id corresponds to a site index initially
    # Position of each Li ion (fractional coordinates)
    li_positions_frac = {lid: structure[lid].frac_coords.copy() for lid in li_ids}

    # ------------------- 6. Helper functions -------------------
    def count_occupied_neighbors(site_idx):
        """Count occupied Li neighbours of a given site within CUTOFF."""
        count = 0
        for nb in all_neighbors[site_idx]:
            if nb.index in li_site_indices and occupied.get(nb.index, False):
                count += 1
        return count

    def barrier_energy(origin, dest):
        """Calculate migration barrier for a hop from origin to dest.
        Includes base barrier, site‑energy difference, and ALPHA term for occupied neighbours.
        """
        # base barrier
        E_bar = E0
        # site‑energy difference (dest - origin)
        idx_o = np.where(li_site_indices == origin)[0][0]
        idx_d = np.where(li_site_indices == dest)[0][0]
        E_bar += site_energies[idx_d] - site_energies[idx_o]
        # occupied neighbour contribution (origin site neighbours before hop)
        occ_nb = count_occupied_neighbors(origin)
        E_bar += ALPHA * occ_nb
        return E_bar

    def rate_from_barrier(E_bar):
        return NU * np.exp(-E_bar / (KB_EV * TEMPERATURE))

    # ------------------- 7. Simulation loop (Gillespie) -------------------
    current_time = 0.0
    step = 0
    # Track tracer displacements (cartesian) for each Li ion
    tracer_disp_cart = {lid: np.zeros(3) for lid in li_ids}
    # Track total charge‑center displacement (cartesian)
    charge_center_disp = np.zeros(3)
    # Pre‑compute lattice for fast conversion
    lattice = structure.lattice

    while current_time < target_time:
        # Build list of possible hops and their rates
        hops = []  # each entry: (origin, dest, rate, frac_disp)
        for origin in li_site_indices:
            if not occupied[origin]:
                continue
            for dest, frac_disp in adj_list[origin]:
                if occupied[dest]:
                    continue  # destination must be vacant
                E_bar = barrier_energy(origin, dest)
                k_hop = rate_from_barrier(E_bar)
                if k_hop > 0:
                    hops.append((origin, dest, k_hop, frac_disp))
        if not hops:
            print("No available hops – simulation stopped early.")
            break
        # Gillespie selection
        rates = np.array([h[2] for h in hops])
        R_total = rates.sum()
        r1 = np.random.random()
        dt = -np.log(r1) / R_total
        current_time += dt
        # Choose which hop occurs
        r2 = np.random.random() * R_total
        cumulative = 0.0
        for origin, dest, k_hop, frac_disp in hops:
            cumulative += k_hop
            if r2 <= cumulative:
                chosen_origin, chosen_dest, chosen_frac = origin, dest, frac_disp
                break
        # Perform the hop
        # Identify which Li ion is at chosen_origin
        moving_li = None
        for lid, pos in li_positions_frac.items():
            # due to periodic wrapping, compare fractional coordinates modulo 1
            if np.allclose((pos - structure[chosen_origin].frac_coords) % 1.0, 0.0, atol=1e-6):
                moving_li = lid
                break
        if moving_li is None:
            # fallback: assume one‑to‑one mapping (origin index == li id)
            moving_li = chosen_origin
        # Update occupancy
        occupied[chosen_origin] = False
        occupied[chosen_dest] = True
        # Update position of the moving ion
        old_frac = li_positions_frac[moving_li]
        new_frac = (old_frac + chosen_frac) % 1.0
        li_positions_frac[moving_li] = new_frac
        # Cartesian displacement for this hop
        cart_disp = lattice.get_cartesian_coords(chosen_frac)
        # Update tracer displacement
        tracer_disp_cart[moving_li] += cart_disp
        # Update charge‑center displacement (sum of all Li displacements divided by N)
        charge_center_disp += cart_disp / len(li_ids)
        # Progress output
        step += 1
        if step % 2000 == 0:
            print(f"Step {step}, time {current_time:.3e} s, total hops {len(hops)}")

    # ------------------- 8. Post‑processing -------------------
    # Compute mean‑square displacements
    tracer_sq = np.mean([np.dot(v, v) for v in tracer_disp_cart.values()])
    charge_sq = np.dot(charge_center_disp, charge_center_disp)

    # Diffusivities (3D)
    D_tracer = tracer_sq / (6.0 * current_time)
    D_sigma = charge_sq / (6.0 * current_time)
    Haven = D_sigma / D_tracer if D_tracer > 0 else 0.0

    # Number density of Li (m⁻³)
    volume_A3 = lattice.volume  # Å³
    volume_m3 = volume_A3 * 1e-30
    N_li = len(li_ids)
    n = N_li / volume_m3

    # Conductivity (S/m) using Haven ratio
    sigma_S_per_m = n * Q_E**2 * Haven * D_tracer / (KB_J * TEMPERATURE)
    # Convert to S/cm
    sigma_S_per_cm = sigma_S_per_m / 100.0

    print(f"Haven ratio: {Haven:.3f}")
    print(f"Conductivity: {sigma_S_per_cm:.3e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
