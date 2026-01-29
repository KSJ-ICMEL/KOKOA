"""
KOKOA Simulation #2
Generated: 2026-01-25 17:20:38
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260125_171933')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from pymatgen.core import Structure
    from scipy.constants import Boltzmann, elementary_charge

    # -----------------------------------------------------------------------------
    # 0. User‑provided path to the LLZO CIF file (injected at runtime)
    # -----------------------------------------------------------------------------
    _CIF_PATH = _CIF_PATH  # placeholder that will be replaced by the execution environment

    # -----------------------------------------------------------------------------
    # 1. Load structure and build supercell
    # -----------------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    # Expand to a 3x3x3 supercell as required
    structure.make_supercell([3, 3, 3])
    print(f"Supercell built: 3x3x3, total atoms = {len(structure)}")

    # -----------------------------------------------------------------------------
    # 2. Identify Li sites and initialise occupancy
    # -----------------------------------------------------------------------------
    li_site_indices = []          # indices of sites that can host Li
    initial_occupancy = []        # 1 if occupied, 0 otherwise
    for i, site in enumerate(structure):
        if any(el.symbol == "Li" for el in site.species.elements):
            li_site_indices.append(i)
            # Use the site occupancy from the CIF if present, otherwise 0.5 probability
            occ = site.species.get("Li", 0.5)
            state = 1 if np.random.rand() < occ else 0
            initial_occupancy.append(state)

    li_site_indices = np.array(li_site_indices, dtype=int)
    occupancy = np.array(initial_occupancy, dtype=int)
    num_li = occupancy.sum()
    print(f"Li sites identified: {len(li_site_indices)}, initially occupied: {num_li}")

    # -----------------------------------------------------------------------------
    # 3. Build neighbour list for possible hops (Li‑Li within a cutoff)
    # -----------------------------------------------------------------------------
    cutoff = 3.5  # Å – typical Li‑Li hopping distance in LLZO
    # For each Li site store a list of (target_index, hop_vector, distance)
    neighbor_events = []  # list of dicts for each possible hop
    for src_idx in li_site_indices:
        src_site = structure[src_idx]
        # Get neighbours within cutoff (including periodic images)
        neighbours = structure.get_neighbors(src_site, r=cutoff)
        for nb in neighbours:
            tgt_idx = nb.index
            if tgt_idx not in li_site_indices:
                continue  # only consider Li‑Li hops
            # Avoid double counting (store only src < tgt)
            if src_idx >= tgt_idx:
                continue
            # Hop vector in Cartesian coordinates (taking periodic image into account)
            frac_diff = structure[tgt_idx].frac_coords - src_site.frac_coords + nb.image
            cart_vec = structure.lattice.get_cartesian_coords(frac_diff)
            distance = np.linalg.norm(cart_vec)
            neighbor_events.append({
                "src": src_idx,
                "tgt": tgt_idx,
                "vec": cart_vec,
                "dist": distance
            })
    print(f"Total possible Li‑Li hop events: {len(neighbor_events)}")

    # -----------------------------------------------------------------------------
    # 4. Helper functions to classify site type and compute local occupancy
    # -----------------------------------------------------------------------------
    # Simple heuristic: tetrahedral sites have 4 O neighbours within ~2.3 Å, octahedral have 6.
    def site_type(site_index):
        site = structure[site_index]
        O_neighbors = structure.get_neighbors(site, r=2.5)
        O_count = sum(1 for nb in O_neighbors if any(el.symbol == "O" for el in nb.species.elements))
        return "tet" if O_count <= 4 else "oct"

    # Pre‑compute site types for all Li sites
    site_types = {idx: site_type(idx) for idx in li_site_indices}

    # -----------------------------------------------------------------------------
    # 5. Parameters for the environment‑dependent barrier model
    # -----------------------------------------------------------------------------
    # Base barriers (eV) for the two site types – taken from DFT literature
    E0 = {"tet": 0.25, "oct": 0.35}
    alpha = 0.05   # eV per occupied neighbour in the first shell
    beta = 0.10    # eV per Å of hop distance beyond a reference (2.5 Å)
    reference_dist = 2.5  # Å
    nu_attempt = 1e13  # Hz – attempt frequency (same for all hops)
    T = 300.0  # K
    kB = Boltzmann / elementary_charge  # eV/K

    # -----------------------------------------------------------------------------
    # 6. Functions to compute the rate of a single hop given the current occupancy
    # -----------------------------------------------------------------------------
    def occupied_neighbors(site_idx, exclude_idx=None):
        """Count occupied Li neighbours of *site_idx* within the cutoff, optionally
        ignoring *exclude_idx* (used when evaluating a specific hop)."""
        site = structure[site_idx]
        neigh = structure.get_neighbors(site, r=cutoff)
        count = 0
        for nb in neigh:
            nb_idx = nb.index
            if nb_idx == exclude_idx:
                continue
            if nb_idx in li_site_indices:
                li_pos = np.where(li_site_indices == nb_idx)[0][0]
                if occupancy[li_pos] == 1:
                    count += 1
        return count

    def hop_rate(event):
        src = event["src"]
        tgt = event["tgt"]
        # Only allow hop if source occupied and target empty
        src_occ = occupancy[np.where(li_site_indices == src)[0][0]]
        tgt_occ = occupancy[np.where(li_site_indices == tgt)[0][0]]
        if src_occ == 0 or tgt_occ == 1:
            return 0.0
        # Base barrier from source site type
        base_E = E0[site_types[src]]
        # Occupancy contribution (neighbors of source, excluding target)
        N_occ = occupied_neighbors(src, exclude_idx=tgt)
        # Distance contribution
        dist = event["dist"]
        dist_factor = max(0.0, dist - reference_dist) * beta
        # Total activation energy
        E_a = base_E + alpha * N_occ + dist_factor
        rate = nu_attempt * np.exp(-E_a / (kB * T))
        return rate

    # -----------------------------------------------------------------------------
    # 7. kMC simulation loop (BKL / residence‑time algorithm)
    # -----------------------------------------------------------------------------
    target_time = 5e-09  # seconds
    current_time = 0.0
    step = 0
    # Track particle positions for MSD calculation
    particle_positions = {}
    for li_idx, occ in zip(li_site_indices, occupancy):
        if occ == 1:
            cart = structure.lattice.get_cartesian_coords(structure[li_idx].frac_coords)
            particle_positions[li_idx] = cart.copy()

    # Store MSD history for optional plotting
    msd_history = []
    time_history = []

    while current_time < target_time:
        # Build list of rates for all admissible hops
        rates = []
        cumulative = []
        total_rate = 0.0
        for ev in neighbor_events:
            r = hop_rate(ev)
            if r > 0:
                total_rate += r
                rates.append(r)
                cumulative.append(total_rate)
        if total_rate == 0.0:
            print("No further hops possible – simulation stopped early.")
            break
        # Choose a random event weighted by rates
        rnd = np.random.rand() * total_rate
        ev_index = np.searchsorted(cumulative, rnd)
        chosen_event = neighbor_events[ev_index]
        src = chosen_event["src"]
        tgt = chosen_event["tgt"]
        # Perform the hop: update occupancy
        src_pos = np.where(li_site_indices == src)[0][0]
        tgt_pos = np.where(li_site_indices == tgt)[0][0]
        occupancy[src_pos] = 0
        occupancy[tgt_pos] = 1
        # Update particle position for the moving ion
        particle_positions[tgt] = particle_positions.pop(src)
        particle_positions[tgt] += chosen_event["vec"]  # move by hop vector
        # Advance time
        dt = -np.log(np.random.rand()) / total_rate
        current_time += dt
        step += 1
        if step % 2000 == 0:
            print(f"Step {step}, simulated time = {current_time:.2e} s")
        # Record MSD every 5000 steps (optional)
        if step % 5000 == 0:
            # Compute mean‑squared displacement of all Li ions relative to their start
            displacements = []
            for li_idx in particle_positions:
                start = structure.lattice.get_cartesian_coords(structure[li_idx].frac_coords)
                disp = particle_positions[li_idx] - start
                displacements.append(np.dot(disp, disp))
            msd = np.mean(displacements) if displacements else 0.0
            msd_history.append(msd)
            time_history.append(current_time)

    # -----------------------------------------------------------------------------
    # 8. Post‑processing: diffusion coefficient and conductivity
    # -----------------------------------------------------------------------------
    if current_time == 0:
        print("Simulation time is zero – cannot compute conductivity.")
    else:
        # Final MSD using the last recorded positions
        displacements = []
        for li_idx in particle_positions:
            start = structure.lattice.get_cartesian_coords(structure[li_idx].frac_coords)
            disp = particle_positions[li_idx] - start
            displacements.append(np.dot(disp, disp))
        final_msd = np.mean(displacements) if displacements else 0.0
        D = final_msd / (6.0 * current_time)  # m^2/s (cartesian coords are in Å, convert)
        D_m2 = D * 1e-20  # Å^2 → m^2
        # Li number density (Li per m^3)
        volume_ang3 = structure.lattice.volume  # Å^3 for the supercell
        volume_m3 = volume_ang3 * 1e-30
        n_li = occupancy.sum()
        n_density = n_li / volume_m3
        sigma = n_density * (elementary_charge ** 2) * D_m2 / (kB * elementary_charge * T)  # S/m
        sigma_S_cm = sigma * 1e-2  # convert to S/cm
        print(f"Conductivity: {sigma_S_cm:.6e} S/cm")

    # Optional: plot MSD vs time if matplotlib is available
    if msd_history:
        plt.figure()
        plt.plot(time_history, msd_history, marker='o')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Squared Displacement (Å$^2$)')
        plt.title('MSD during kMC')
        plt.tight_layout()
        plt.show()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
