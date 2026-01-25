"""
KOKOA Simulation #10
Generated: 2026-01-25 17:31:23
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
    from scipy.constants import k as k_B_J, physical_constants, Avogadro
    from pymatgen.core import Structure
    from pymatgen.core.periodic_table import Element
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

    # Bottleneck parameters (tuned from DFT/NEB data)
    R_CUT = 0.80  # Å, radius below which barrier starts to increase
    K_FACTOR = 0.50  # eV/Å, slope of barrier increase with decreasing radius

    # ------------------- 1. Load structure -------------------
    if not os.path.isfile(_CIF_PATH):
        raise FileNotFoundError(f"CIF file not found: {_CIF_PATH}")

    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(SUPER_CELL)
    print(f"Supercell built: {SUPER_CELL}, total atoms: {len(structure)}")

    # ------------------- 2. Identify Li sites -------------------
    li_site_indices = [i for i, site in enumerate(structure) if "Li" in [el.symbol for el in site.species]]
    li_site_indices = np.array(li_site_indices, dtype=int)
    num_li_sites = len(li_site_indices)
    print(f"Number of Li sites: {num_li_sites}")

    # ------------------- 3. Create vacancy configuration -------------------
    # Introduce a small vacancy concentration (≈5%)
    np.random.seed(42)
    vacancy_fraction = 0.05
    num_vacancies = int(vacancy_fraction * num_li_sites)
    vacancy_sites = np.random.choice(li_site_indices, size=num_vacancies, replace=False)
    occupied = np.ones(num_li_sites, dtype=bool)
    # Map global site index -> position in li_site_indices array
    global_to_local = {g: i for i, g in enumerate(li_site_indices)}
    for g in vacancy_sites:
        occupied[global_to_local[g]] = False

    # ------------------- 4. Helper functions -------------------

    def get_cartesian(idx):
        """Return Cartesian coordinates of a site given its global index."""
        return structure[idx].coords

    def bottleneck_radius(i_global, j_global, n_samples=10):
        """Estimate the minimum opening radius along the straight line i→j.
        Returns the smallest distance from the line to any non‑Li atom minus that atom's covalent radius.
        """
        pos_i = get_cartesian(i_global)
        pos_j = get_cartesian(j_global)
        # Vector along the hop
        vec = pos_j - pos_i
        min_dist = np.inf
        # Loop over sample points
        for t in np.linspace(0, 1, n_samples):
            point = pos_i + t * vec
            # Check all framework atoms (non‑Li)
            for site in structure:
                if "Li" in [el.symbol for el in site.species]:
                    continue
                # Periodic images are handled by the supercell; simple distance is sufficient
                d = np.linalg.norm(point - site.coords)
                # Covalent radius of the atom type (fallback to 1.0 Å if unknown)
                elem = site.species_string.split()[0]
                try:
                    r_cov = Element(elem).covalent_radius
                except Exception:
                    r_cov = 1.0
                clearance = d - r_cov
                if clearance < min_dist:
                    min_dist = clearance
        return max(min_dist, 0.0)  # never negative

    def barrier_correction(r_min):
        """Map bottleneck radius to an additional barrier (eV)."""
        if r_min >= R_CUT:
            return 0.0
        return K_FACTOR * (R_CUT - r_min)

    # ------------------- 5. Build adjacency list with geometry‑dependent rates -------------------
    # Pre‑compute neighbor information within the cutoff
    all_neighbors = structure.get_all_neighbors(r=CUTOFF)
    adjacency = {i: [] for i in li_site_indices}
    rate_lookup = {}
    for i_global in li_site_indices:
        for neighbor in all_neighbors[i_global]:
            j_global = neighbor.site_index
            if j_global not in li_site_indices:
                continue
            # Avoid double counting (i→j and j→i will be added separately when i loops)
            # Compute bottleneck radius
            r_min = bottleneck_radius(i_global, j_global)
            # Discard impossible hops
            if r_min < 0.2:  # Å, arbitrary physical lower bound
                continue
            delta_E = barrier_correction(r_min)
            total_E = E0 + delta_E  # eV
            rate = NU * np.exp(-total_E / (KB_EV * TEMPERATURE))
            # Store hop information
            adjacency[i_global].append((j_global, rate, r_min, total_E))
            rate_lookup[(i_global, j_global)] = rate

    print("Adjacency list built with geometry‑dependent barriers.")

    # ------------------- 6. Kinetic Monte Carlo simulation -------------------
    # Prepare data structures for MSD calculation
    # Track the Cartesian position of each Li ion (only occupied sites initially)
    ion_positions = {}
    for local_idx, occupied_flag in enumerate(occupied):
        if occupied_flag:
            g_idx = li_site_indices[local_idx]
            ion_positions[g_idx] = get_cartesian(g_idx).copy()

    # Helper to pick an event weighted by rates
    def pick_event(possible_events, total_rate):
        r = np.random.random() * total_rate
        cumulative = 0.0
        for (i, j, rate) in possible_events:
            cumulative += rate
            if r < cumulative:
                return i, j, rate
        # Fallback (should not happen)
        return possible_events[-1]

    current_time = 0.0
    step = 0
    msd_list = []
    time_list = []

    while current_time < target_time:
        possible_events = []
        total_rate = 0.0
        # Build list of all allowed hops (occupied -> vacant neighbor)
        for i_global in list(ion_positions.keys()):
            for (j_global, rate, _, _) in adjacency[i_global]:
                if j_global in ion_positions:
                    continue  # destination already occupied
                possible_events.append((i_global, j_global, rate))
                total_rate += rate
        if total_rate == 0.0:
            print("No more possible hops – simulation halted.")
            break
        # Time increment
        dt = -np.log(np.random.random()) / total_rate
        current_time += dt
        # Choose event
        i_sel, j_sel, _ = pick_event(possible_events, total_rate)
        # Execute hop: move ion from i to j
        ion_positions[j_sel] = ion_positions.pop(i_sel)
        # Update position to exact Cartesian of new site (no diffusion within site)
        ion_positions[j_sel] = get_cartesian(j_sel).copy()
        step += 1
        if step % 2000 == 0:
            print(f"Step {step}, time {current_time:.3e} s")
        # Record MSD every 5000 steps
        if step % 5000 == 0:
            # Compute mean‑square displacement of all ions relative to their initial positions
            displacements = []
            for g_idx, pos in ion_positions.items():
                init_pos = get_cartesian(g_idx)  # initial site of this ion (same as current site index)
                disp = np.linalg.norm(pos - init_pos) ** 2
                displacements.append(disp)
            msd = np.mean(displacements) if displacements else 0.0
            msd_list.append(msd)
            time_list.append(current_time)

    # ------------------- 7. Post‑processing -------------------
    if len(time_list) < 2:
        raise RuntimeError("Insufficient data to compute diffusion coefficient.")
    # Linear fit of MSD vs time (first and last point for simplicity)
    D = (msd_list[-1] - msd_list[0]) / (6.0 * (time_list[-1] - time_list[0]))  # cm^2/s (positions are Å, convert)
    # Convert Å^2 to cm^2 (1 Å = 1e-8 cm)
    D *= 1e-16
    # Number density of mobile Li (per cm^3)
    volume = structure.lattice.volume * (1e-24)  # Å^3 -> cm^3
    n_li = len(ion_positions) / volume
    q = physical_constants['elementary charge'][0]
    # Nernst‑Einstein conductivity
    sigma = n_li * q**2 * D / (KB_J * TEMPERATURE)  # S/m
    sigma_cm = sigma * 100  # S/cm
    print(f"Conductivity: {sigma_cm:.3e} S/cm")

    # Optional: plot MSD vs time
    plt.figure()
    plt.plot(time_list, msd_list, 'o-')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (Å$^2$)')
    plt.title('Mean‑Square Displacement')
    plt.tight_layout()
    plt.show()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
