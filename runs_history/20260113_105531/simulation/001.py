"""
KOKOA Simulation #1
Generated: 2026-01-13 10:56:12
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260113_105531')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    import scipy.constants as const
    from pymatgen.core import Structure

    # ----------------------------------------------------------------------
    # Configuration (fixed by the task)
    # ----------------------------------------------------------------------
    target_time = 5e-9                     # seconds
    supercell_dim = [3, 3, 3]              # 3×3×3 expansion
    temperature = 300.0                   # K
    hop_rate = 1e13                        # s⁻¹ (attempt frequency, uniform)
    cutoff = 4.0                           # Å, neighbor search radius
    print_interval = 2000                 # steps

    # ----------------------------------------------------------------------
    # Load structure
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(supercell_dim)

    # ----------------------------------------------------------------------
    # Identify Li sites
    # ----------------------------------------------------------------------
    li_site_indices = [i for i, site in enumerate(structure)
                       if any(el.symbol == "Li" for el in site.species)]

    num_sites = len(li_site_indices)

    # ----------------------------------------------------------------------
    # Initialise occupancy (50 % random occupancy for a non‑trivial conductivity)
    # ----------------------------------------------------------------------
    np.random.seed(42)
    occupied = np.random.rand(num_sites) < 0.5          # boolean array
    num_ions = occupied.sum()

    # Mapping: site index → ion id (or -1 if vacant)
    site_to_ion = -np.ones(num_sites, dtype=int)
    ion_initial_pos = np.zeros((num_ions, 3))   # Å
    ion_current_pos = np.zeros((num_ions, 3))   # Å

    ion_counter = 0
    for idx, occ in enumerate(occupied):
        if occ:
            site_to_ion[idx] = ion_counter
            cart = structure.sites[li_site_indices[idx]].coords   # Å
            ion_initial_pos[ion_counter] = cart
            ion_current_pos[ion_counter] = cart
            ion_counter += 1

    # ----------------------------------------------------------------------
    # Build neighbour list for each Li site (indices relative to li_site_indices)
    # ----------------------------------------------------------------------
    neighbors = [[] for _ in range(num_sites)]

    for i, site_idx in enumerate(li_site_indices):
        site = structure.sites[site_idx]
        neigh = structure.get_neighbors(site, r=cutoff, include_index=True)
        for nb in neigh:
            nb_idx = nb.index
            # consider only Li neighbours
            if nb_idx in li_site_indices:
                j = li_site_indices.index(nb_idx)
                if i != j:
                    # store neighbour index and displacement vector (Å)
                    disp = nb.distance * nb.unit_vector   # Å
                    neighbors[i].append((j, disp))

    # ----------------------------------------------------------------------
    # KMCSimulator class
    # ----------------------------------------------------------------------
    class KMCSimulator:
        def __init__(self, structure, li_indices, neighbors, site_to_ion,
                     ion_initial_pos, ion_current_pos, temperature):
            self.structure = structure
            self.li_indices = li_indices
            self.neighbors = neighbors
            self.site_to_ion = site_to_ion
            self.ion_initial_pos = ion_initial_pos
            self.ion_current_pos = ion_current_pos
            self.temperature = temperature

            self.num_sites = len(li_indices)
            self.num_ions = ion_initial_pos.shape[0]

            self.time = 0.0
            self.step = 0

            # volume in m³ (structure.lattice.volume is Å³)
            self.volume_m3 = structure.lattice.volume * 1e-30

        def _possible_hops(self):
            """Return list of (from_site, to_site, ion_id) for all allowed hops."""
            hops = []
            for i in range(self.num_sites):
                ion_id = self.site_to_ion[i]
                if ion_id == -1:
                    continue                     # site empty
                for j, _disp in self.neighbors[i]:
                    if self.site_to_ion[j] == -1:   # destination empty
                        hops.append((i, j, ion_id))
            return hops

        def run_step(self):
            hops = self._possible_hops()
            if not hops:
                # No possible moves – simulation stops
                self.time = target_time
                return

            total_rate = hop_rate * len(hops)   # s⁻¹
            # Choose a hop uniformly (all rates equal)
            chosen = hops[np.random.randint(len(hops))]
            i, j, ion_id = chosen

            # Update occupancy mapping
            self.site_to_ion[i] = -1
            self.site_to_ion[j] = ion_id

            # Update ion position
            new_cart = self.structure.sites[self.li_indices[j]].coords   # Å
            self.ion_current_pos[ion_id] = new_cart

            # Advance time
            r = np.random.rand()
            dt = -np.log(r) / total_rate
            self.time += dt
            self.step += 1

        def calculate_properties(self):
            """Return MSD (Å²), diffusion coefficient D (m²/s), conductivity σ (S/cm)."""
            # Mean squared displacement (Å²)
            displacements = self.ion_current_pos - self.ion_initial_pos
            msd = np.mean(np.sum(displacements**2, axis=1))

            # Diffusion coefficient D = MSD / (6 t)  (m²/s)
            D_m2_s = (msd * 1e-20) / (6.0 * self.time)   # Å² → m²

            # Number density n = N_occupied / V   (m⁻³)
            n = self.num_ions / self.volume_m3

            # Nernst‑Einstein conductivity σ = (q² n D) / (kB T)   (S/m)
            sigma_S_m = (const.e**2 * n * D_m2_s) / (const.k * self.temperature)

            # Convert to S/cm
            sigma_S_cm = sigma_S_m * 1e-2
            return msd, D_m2_s, sigma_S_cm

        def run(self):
            while self.time < target_time:
                self.run_step()
                if self.step % print_interval == 0:
                    msd, D, sigma = self.calculate_properties()
                    print(f"Step: {self.step:6d} | "
                          f"Time: {self.time*1e9:8.3f} ns | "
                          f"MSD: {msd:10.3f} Å² | "
                          f"σ: {sigma*1e3:8.3f} mS/cm")
            # Final report
            _, _, sigma_final = self.calculate_properties()
            print(f"Conductivity: {sigma_final:.3e} S/cm")


    # ----------------------------------------------------------------------
    # Execute simulation
    # ----------------------------------------------------------------------
    sim = KMCSimulator(structure,
                       li_site_indices,
                       neighbors,
                       site_to_ion,
                       ion_initial_pos,
                       ion_current_pos,
                       temperature)

    sim.run()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
