"""
KOKOA Simulation #2
Generated: 2026-01-25 17:25:52
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
    import numpy as np
    import os
    from pymatgen.core import Structure
    import matplotlib.pyplot as plt

    # ------------------- User‑provided inputs -------------------
    # Path to the CIF file (injected at runtime)
    _CIF_PATH = _CIF_PATH  # placeholder, will be replaced by the execution environment

    # Simulation parameters
    TARGET_TIME = 5e-09  # seconds
    SUPER_CELL = [3, 3, 3]
    CUTOFF = 4.0  # Å, neighbour search radius for possible hops
    TEMPERATURE = 300.0  # K
    KB = 8.617e-5  # eV/K
    NU = 1e13  # attempt frequency, s^-1 (typical phonon frequency)
    E0 = 0.40  # eV, base migration barrier
    ALPHA = 0.05  # eV per occupied neighbour

    # ------------------- 1. Load structure -------------------
    if not os.path.isfile(_CIF_PATH):
        raise FileNotFoundError(f"CIF file not found: {_CIF_PATH}")

    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(SUPER_CELL)
    print(f"Supercell built: {SUPER_CELL}, total atoms: {len(structure)}")

    # Identify Li sites (indices) and store their fractional coordinates
    li_indices = []
    li_frac_coords = []
    for i, site in enumerate(structure):
        if "Li" in [el.symbol for el in site.species.elements]:
            li_indices.append(i)
            li_frac_coords.append(site.frac_coords)
    li_indices = np.array(li_indices)
    li_frac_coords = np.array(li_frac_coords)

    num_li_sites = len(li_indices)
    print(f"Number of Li sites: {num_li_sites}")

    # ------------------- 2. Build adjacency (possible hops) -------------------
    # For each Li site we find other Li sites within the cutoff distance.
    adj_list = {idx: [] for idx in li_indices}
    # Pre‑compute neighbor information for barrier evaluation (all Li neighbours within cutoff)
    neighbour_map = {idx: [] for idx in li_indices}

    all_neighbors = structure.get_all_neighbors(r=CUTOFF)
    for i, site in enumerate(structure):
        if i not in li_indices:
            continue
        for nb in all_neighbors[i]:
            j = nb.index
            if j not in li_indices:
                continue
            # displacement vector (cartesian) from i to j taking periodic images into account
            frac_diff = structure[j].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            # store as possible hop (i -> j)
            adj_list[i].append((j, cart_disp))
            # also store neighbour for barrier model (undirected)
            neighbour_map[i].append(j)

    print("Adjacency list constructed.")

    # ------------------- 3. Initialise occupancy -------------------
    # Randomly occupy ~50% of Li sites to create vacancies for hopping
    np.random.seed(42)
    occupancy = np.zeros(num_li_sites, dtype=int)
    initial_occupied = np.random.choice(num_li_sites, size=num_li_sites // 2, replace=False)
    occupancy[initial_occupied] = 1
    # Map from global site index to position in the occupancy array
    siteidx_to_occpos = {site: pos for pos, site in enumerate(li_indices)}

    # Initialise particle positions (only for occupied sites)
    particle_positions = {}
    particle_id = 0
    for pos, occ in enumerate(occupancy):
        if occ == 1:
            site_global = li_indices[pos]
            cart = structure.lattice.get_cartesian_coords(structure[site_global].frac_coords)
            particle_positions[particle_id] = {
                "site": site_global,
                "start": cart.copy(),
                "current": cart.copy()
            }
            particle_id += 1
    num_particles = particle_id
    print(f"Initialized {num_particles} Li particles (occupied sites).")

    # ------------------- 4. Helper functions -------------------
    def count_occupied_neighbours(site_idx, occ_array):
        """Return number of occupied Li neighbours (excluding the site itself)."""
        neigh = neighbour_map[site_idx]
        count = 0
        for nb in neigh:
            pos = siteidx_to_occpos[nb]
            count += occ_array[pos]
        return count

    def barrier_for_hop(src, tgt, occ_array):
        """Empirical barrier: E0 + α * (occupied neighbours of src + tgt)."""
        n_src = count_occupied_neighbours(src, occ_array)
        n_tgt = count_occupied_neighbours(tgt, occ_array)
        return E0 + ALPHA * (n_src + n_tgt)

    def rate_for_barrier(Ea):
        return NU * np.exp(-Ea / (KB * TEMPERATURE))

    # ------------------- 5. kMC loop -------------------
    current_time = 0.0
    step = 0
    # For MSD calculation we keep a list of squared displacements per particle
    msd_accumulator = np.zeros(num_particles)

    while current_time < TARGET_TIME:
        events = []          # (src_global, tgt_global, disp_vector, rate)
        cumulative_rates = []
        total_rate = 0.0
        # Scan all possible hops
        for src in li_indices:
            src_pos = siteidx_to_occpos[src]
            if occupancy[src_pos] == 0:
                continue  # source empty
            for tgt, disp in adj_list[src]:
                tgt_pos = siteidx_to_occpos[tgt]
                if occupancy[tgt_pos] == 1:
                    continue  # target already occupied
                Ea = barrier_for_hop(src, tgt, occupancy)
                r = rate_for_barrier(Ea)
                if r <= 0:
                    continue
                total_rate += r
                events.append((src, tgt, disp, r))
                cumulative_rates.append(total_rate)
        if total_rate == 0.0:
            print("No more possible events – simulation stops early.")
            break
        # Choose event
        rnd = np.random.rand() * total_rate
        idx = np.searchsorted(cumulative_rates, rnd)
        src, tgt, disp, _ = events[idx]
        # Update occupancy
        src_pos = siteidx_to_occpos[src]
        tgt_pos = siteidx_to_occpos[tgt]
        occupancy[src_pos] = 0
        occupancy[tgt_pos] = 1
        # Update particle bookkeeping (find which particle was at src)
        moved_particle = None
        for pid, info in particle_positions.items():
            if info["site"] == src:
                moved_particle = pid
                break
        if moved_particle is None:
            raise RuntimeError("Inconsistent particle mapping.")
        # Update particle record
        particle_positions[moved_particle]["site"] = tgt
        particle_positions[moved_particle]["current"] += disp
        # Update MSD accumulator for this particle
        dr = particle_positions[moved_particle]["current"] - particle_positions[moved_particle]["start"]
        msd_accumulator[moved_particle] = np.dot(dr, dr)
        # Advance time
        dt = -np.log(np.random.rand()) / total_rate
        current_time += dt
        step += 1
        if step % 2000 == 0:
            print(f"Step {step}, time {current_time:.3e} s, total_rate {total_rate:.3e} s⁻¹")

    # ------------------- 6. Post‑processing -------------------
    # Mean squared displacement (average over particles)
    if num_particles > 0:
        msd = np.mean(msd_accumulator)
    else:
        msd = 0.0
    D = msd / (6.0 * current_time) if current_time > 0 else 0.0
    # Number density of Li (occupied sites) per cm³
    volume_cm3 = structure.lattice.volume * 1e-24  # Å³ → cm³
    n_li = np.sum(occupancy) / volume_cm3
    e_charge = 1.602176634e-19  # C
    sigma = n_li * (e_charge ** 2) * D / (KB * 1.602176634e-19 * TEMPERATURE)  # using kB in J/K
    # Convert to S/cm (already in SI, divide by 100 for cm)
    sigma_sc_cm = sigma / 100.0
    print(f"Conductivity: {sigma_sc_cm:.6e} S/cm")

    # Optional: plot MSD vs time (if many steps recorded) – omitted for brevity
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
