"""
KOKOA Simulation #2
Generated: 2026-01-22 17:43:07
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260122_174154')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from pymatgen.core import Structure
    import sys
    import math

    # -----------------------------------------------------------------------------
    # 0. User‑provided inputs (injected at runtime)
    # -----------------------------------------------------------------------------
    # Path to the CIF file containing the crystal structure (e.g., LLZO)
    # The testing harness will set this variable before executing the script.
    _CIF_PATH = globals().get('_CIF_PATH')
    if _CIF_PATH is None:
        raise RuntimeError('Variable _CIF_PATH must be defined at runtime')

    # -----------------------------------------------------------------------------
    # 1. Load structure and build supercell
    # -----------------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    # 3×3×3 supercell as required by the task
    structure.make_supercell([3, 3, 3])
    print(f"Supercell built: 3x3x3, total atoms = {len(structure)}")

    # -----------------------------------------------------------------------------
    # 2. Identify Li sites and initialise occupancy
    # -----------------------------------------------------------------------------
    li_indices = [i for i, site in enumerate(structure) if "Li" in site.species_string]
    num_li_sites = len(li_indices)
    # Random half‑filled Li sub‑lattice (probability 0.5 for each site)
    np.random.seed(42)
    occupancy = np.zeros(len(structure), dtype=int)
    for idx in li_indices:
        occupancy[idx] = 1 if np.random.rand() < 0.5 else 0

    # Mapping from occupied site -> particle id and vice‑versa
    site_to_particle = {}
    particle_positions = {}
    particle_id = 0
    for idx in li_indices:
        if occupancy[idx] == 1:
            site_to_particle[idx] = particle_id
            cart = structure.lattice.get_cartesian_coords(structure[idx].frac_coords)
            particle_positions[particle_id] = np.array(cart)
            particle_id += 1
    num_particles = particle_id
    print(f"Initialized {num_particles} Li ions (out of {num_li_sites} Li sites)")

    # -----------------------------------------------------------------------------
    # 3. Build adjacency list for Li‑Li hops (cut‑off = 4 Å)
    # -----------------------------------------------------------------------------
    cutoff = 4.0  # Å
    adjacency = {idx: [] for idx in li_indices}
    # Pre‑compute neighbor information once (index, displacement vector in Å, image)
    for src in li_indices:
        site = structure[src]
        neighbors = structure.get_neighbors(site, cutoff)
        for nb in neighbors:
            tgt = nb.index
            if tgt not in li_indices:
                continue
            # displacement vector from src to tgt in Cartesian Å
            frac_diff = structure[tgt].frac_coords - site.frac_coords + nb.image
            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
            adjacency[src].append((tgt, cart_disp))

    # -----------------------------------------------------------------------------
    # 4. Helper to compute hop‑specific rate
    # -----------------------------------------------------------------------------
    kb_eV = 8.617e-5  # eV/K
    kb_J = 1.380649e-23  # J/K
    T = 300.0  # K

    # Maximum possible neighbour count (used for normalisation)
    max_neighbors = max(len(v) for v in adjacency.values()) if adjacency else 1

    def compute_rate(src, tgt, occ):
        """Return (rate [s⁻¹], Ea [eV], ν [Hz]) for a hop src→tgt.
        The barrier and prefactor depend on the number of occupied Li neighbours
        around the origin site (excluding the target site)."""
        # Count occupied Li neighbours of src (excluding tgt)
        occupied_nb = 0
        for nb_idx, _ in adjacency[src]:
            if nb_idx == tgt:
                continue
            if occ[nb_idx] == 1:
                occupied_nb += 1
        occ_factor = occupied_nb / max_neighbors  # 0 … 1
        # Simple linear model for barrier and attempt frequency
        Ea = 0.30 + 0.20 * occ_factor          # eV, range 0.30–0.50 eV
        nu = 1e13 * (1.0 - 0.30 * occ_factor)   # Hz, modest reduction with crowding
        rate = nu * math.exp(-Ea / (kb_eV * T))
        return rate, Ea, nu

    # -----------------------------------------------------------------------------
    # 5. Kinetic Monte Carlo loop (BKL / residence‑time algorithm)
    # -----------------------------------------------------------------------------
    target_time = 5e-9  # seconds
    current_time = 0.0
    step = 0
    # Store MSD data for conductivity calculation
    initial_positions = {pid: pos.copy() for pid, pos in particle_positions.items()}

    while current_time < target_time:
        events = []          # (src, tgt, disp, rate)
        cumulative = []      # cumulative sum of rates
        total_rate = 0.0
        # Enumerate all possible hops
        for src in li_indices:
            if occupancy[src] == 0:
                continue
            for tgt, disp in adjacency[src]:
                if occupancy[tgt] != 0:
                    continue
                rate, _, _ = compute_rate(src, tgt, occupancy)
                if rate <= 0:
                    continue
                total_rate += rate
                events.append((src, tgt, disp, rate))
                cumulative.append(total_rate)
        if total_rate == 0.0:
            print("No further hops possible – simulation stops early.")
            break
        # Choose event
        r = np.random.rand() * total_rate
        idx = np.searchsorted(cumulative, r)
        src, tgt, disp, _ = events[idx]
        # Execute hop: update occupancy
        occupancy[src] = 0
        occupancy[tgt] = 1
        # Update particle mapping and positions
        pid = site_to_particle.pop(src)
        site_to_particle[tgt] = pid
        particle_positions[pid] += disp  # Cartesian displacement
        # Advance time
        dt = -math.log(np.random.rand()) / total_rate
        current_time += dt
        step += 1
        if step % 2000 == 0:
            print(f"Step {step}, time = {current_time:.3e} s, total_rate = {total_rate:.3e} s⁻¹")

    # -----------------------------------------------------------------------------
    # 6. Post‑processing: compute conductivity
    # -----------------------------------------------------------------------------
    # Mean‑squared displacement (average over all particles)
    msd = 0.0
    for pid, pos in particle_positions.items():
        dr = pos - initial_positions[pid]
        msd += np.dot(dr, dr)
    msd /= num_particles  # Å²
    # Convert to m²
    msd_m2 = msd * (1e-10)**2
    # Diffusion coefficient D = <r²> / (6 t)
    D = msd_m2 / (6.0 * current_time)
    # Number density of Li (m⁻³)
    volume_A3 = structure.lattice.volume  # Å³
    volume_cm3 = volume_A3 * 1e-24
    n_density = num_particles / volume_cm3  # cm⁻³
    # Conductivity σ = n·e²·D / (kB·T)
    e = 1.602176634e-19  # C
    sigma = n_density * (e**2) * D / (kb_J * T)  # S·cm⁻¹ (since n is per cm³)
    print(f"Conductivity: {sigma:.6e} S/cm")

    # Optional: plot MSD vs time (not required for the task but useful for debugging)
    # plt.figure()
    # plt.loglog(current_time, msd_m2, 'o')
    # plt.xlabel('Time (s)')
    # plt.ylabel('MSD (m²)')
    # plt.show()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
