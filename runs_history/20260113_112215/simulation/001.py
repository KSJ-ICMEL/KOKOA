"""
KOKOA Simulation #1
Generated: 2026-01-13 11:22:57
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260113_112215')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure

    # ----------------------------------------------------------------------
    # 1. Load structure and build supercell
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell([3, 3, 3])          # fixed 3×3×3 supercell

    # ----------------------------------------------------------------------
    # 2. Identify Li sites and initialise occupancy
    # ----------------------------------------------------------------------
    li_indices = [i for i, site in enumerate(structure)
                  if any(el.symbol == "Li" for el in site.species)]

    n_li = len(li_indices)

    # Create a small fraction of vacancies (e.g., 5%)
    rng = np.random.default_rng()
    vacancy_mask = rng.random(n_li) < 0.05
    occupancy = np.ones(n_li, dtype=int)
    occupancy[vacancy_mask] = 0

    # Mapping: site_index (0..n_li-1) -> ion_id (or -1 if vacant)
    site_to_ion = np.full(n_li, -1, dtype=int)
    ion_to_site = {}
    ion_counter = 0
    for idx, occ in enumerate(occupancy):
        if occ:
            site_to_ion[idx] = ion_counter
            ion_to_site[ion_counter] = idx
            ion_counter += 1

    n_ions = ion_counter

    # Store initial Cartesian positions of each ion (Å)
    initial_positions = np.zeros((n_ions, 3))
    for ion_id, site_idx in ion_to_site.items():
        frac = structure.frac_coords[li_indices[site_idx]]
        initial_positions[ion_id] = structure.lattice.get_cartesian_coords(frac)

    # ----------------------------------------------------------------------
    # 3. Build adjacency list for Li‑Li hops
    # ----------------------------------------------------------------------
    cutoff = 4.0  # Å
    adjacency = {}  # src (0..n_li-1) -> list of (tgt, cartesian displacement)

    for src_local, src_idx in enumerate(li_indices):
        site = structure[src_idx]
        neigh = structure.get_neighbors(site, cutoff)
        hops = []
        for nb in neigh:
            tgt_idx = nb.index
            # consider only Li neighbours
            if not any(el.symbol == "Li" for el in structure[tgt_idx].species):
                continue
            tgt_local = li_indices.index(tgt_idx)  # convert to 0..n_li-1
            # fractional displacement including periodic image
            frac_disp = (structure.frac_coords[tgt_idx] -
                         site.frac_coords + nb.image)
            cart_disp = structure.lattice.get_cartesian_coords(frac_disp)
            hops.append((tgt_local, cart_disp))
        adjacency[src_local] = hops

    # ----------------------------------------------------------------------
    # 4. KMCSimulator definition
    # ----------------------------------------------------------------------
    kb_eV = 8.617333262e-5          # eV/K
    kb_J = 1.380649e-23            # J/K
    e_charge = 1.602176634e-19     # C
    target_time = 5e-9             # seconds (fixed)

    class KMCSimulator:
        def __init__(self, structure, li_indices, adjacency,
                     occupancy, site_to_ion, ion_to_site,
                     params):
            self.structure = structure
            self.li_indices = li_indices
            self.adj = adjacency
            self.occupancy = occupancy.copy()          # 0/1 per Li site
            self.site_to_ion = site_to_ion.copy()      # site -> ion id
            self.ion_to_site = ion_to_site.copy()      # ion id -> site
            self.params = params
            self.time = 0.0
            self.step = 0

            # store current Cartesian positions of ions (Å)
            n_ions = len(self.ion_to_site)
            self.positions = np.zeros((n_ions, 3))
            for ion_id, site_idx in self.ion_to_site.items():
                frac = self.structure.frac_coords[self.li_indices[site_idx]]
                self.positions[ion_id] = self.structure.lattice.get_cartesian_coords(frac)

            # initial positions for MSD
            self.initial_positions = self.positions.copy()

        def vacancy_factor(self, src):
            """1 - (occupied neighbours / total neighbours) for a source site."""
            neigh = self.adj.get(src, [])
            if not neigh:
                return 1.0
            occupied = sum(self.occupancy[tgt] for tgt, _ in neigh)
            return 1.0 - occupied / len(neigh)

        def run_step(self):
            """Perform a single BKL (Gillespie) kMC step."""
            rates = []
            events = []   # (src, tgt, disp, rate)
            total_rate = 0.0

            nu = self.params['nu']
            Ea = self.params['Ea']
            T = self.params['T']
            prefactor = nu * np.exp(-Ea / (kb_eV * T))

            for src in range(len(self.li_indices)):
                if self.occupancy[src] == 0:
                    continue
                for tgt, disp in self.adj.get(src, []):
                    if self.occupancy[tgt] == 1:
                        continue
                    factor = self.vacancy_factor(src)
                    rate = prefactor * factor
                    total_rate += rate
                    events.append((src, tgt, disp, total_rate))

            if total_rate == 0.0:
                # No possible moves – stop simulation
                self.time = target_time
                return

            # Choose event
            r = np.random.random() * total_rate
            for src, tgt, disp, cum_rate in events:
                if r < cum_rate:
                    chosen = (src, tgt, disp)
                    break

            src, tgt, disp = chosen

            # Update occupancy and mappings
            ion_id = self.site_to_ion[src]
            self.occupancy[src] = 0
            self.occupancy[tgt] = 1
            self.site_to_ion[src] = -1
            self.site_to_ion[tgt] = ion_id
            self.ion_to_site[ion_id] = tgt

            # Update ion position
            self.positions[ion_id] += disp   # Å

            # Advance time
            self.time += -np.log(np.random.random()) / total_rate
            self.step += 1

        def calculate_properties(self):
            """Return MSD (Å^2), diffusion coefficient (cm^2/s), conductivity (S/cm)."""
            dt = self.time
            if dt == 0:
                return 0.0, 0.0, 0.0

            # Mean‑square displacement (Å^2)
            displacements = self.positions - self.initial_positions
            msd = np.mean(np.sum(displacements**2, axis=1))

            # Convert to cm^2
            msd_cm2 = msd * 1e-16

            # Diffusion coefficient D = MSD / (6 t)
            D = msd_cm2 / (6.0 * dt)   # cm^2/s

            # Number density of mobile Li (cm^-3)
            vol_cm3 = self.structure.lattice.volume * 1e-24
            n_occ = np.sum(self.occupancy)
            n = n_occ / vol_cm3

            # Conductivity σ = n q^2 D / (kB T)
            sigma = n * e_charge**2 * D / (kb_J * self.params['T'])   # S/cm
            return msd, D, sigma

    # ----------------------------------------------------------------------
    # 5. Simulation parameters
    # ----------------------------------------------------------------------
    params = {
        'nu': 1e13,          # s^-1, attempt frequency
        'Ea': 0.30,          # eV, activation energy
        'T' : 300.0,         # K
    }

    sim = KMCSimulator(structure, li_indices, adjacency,
                       occupancy, site_to_ion, ion_to_site, params)

    # ----------------------------------------------------------------------
    # 6. Main kMC loop with progress reporting
    # ----------------------------------------------------------------------
    while sim.time < target_time:
        sim.run_step()
        if sim.step % 2000 == 0:
            msd, D, sigma = sim.calculate_properties()
            time_ns = sim.time * 1e9
            sigma_mS_cm = sigma * 1e3   # mS/cm
            print(f"Step {sim.step:6d}, Time {time_ns:8.3f} ns, "
                  f"MSD {msd:10.3f} Å², sigma {sigma_mS_cm:8.3f} mS/cm")

    # ----------------------------------------------------------------------
    # 7. Final conductivity output
    # ----------------------------------------------------------------------
    _, _, final_sigma = sim.calculate_properties()
    print(f"Conductivity: {final_sigma:.6e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
