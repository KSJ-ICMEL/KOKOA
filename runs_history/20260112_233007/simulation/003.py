"""
KOKOA Simulation #3
Generated: 2026-01-12 23:31:44
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_233007')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure
    from scipy.constants import k as kB_J, e as e_charge, Avogadro

    # ----------------------------------------------------------------------
    # Constants (SI unless noted)
    # ----------------------------------------------------------------------
    kB_eV = 8.617333262e-5          # eV/K
    T = 300.0                       # K
    nu = 1e13                       # 1/s, attempt frequency
    base_Ea = 0.30                  # eV, base barrier
    alpha = 0.02                    # eV per neighbor (tunable)
    target_time = 1e-8              # s, fixed simulation time
    cutoff = 4.0                    # Å, neighbor search radius
    dim = 3                         # dimensionality for MSD

    # ----------------------------------------------------------------------
    # Load structure
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    # make a modest supercell to have enough sites for KMC
    structure.make_supercell([2, 2, 2])

    class KMCSimulator:
        def __init__(self, struct, cutoff, base_Ea, alpha, nu, T):
            self.struct = struct
            self.cutoff = cutoff
            self.base_Ea = base_Ea
            self.alpha = alpha
            self.nu = nu
            self.T = T
            self.time = 0.0

            # ------------------------------------------------------------------
            # Identify Li sites and build neighbor list (indices of Li sites)
            # ------------------------------------------------------------------
            self.li_site_indices = self._get_li_site_indices()
            self.N_ions = len(self.li_site_indices)          # assume one Li per Li site
            self._build_neighbor_dict()

            # ------------------------------------------------------------------
            # Occupancy: True if site occupied by a Li ion
            # ------------------------------------------------------------------
            self.occupied = np.ones(self.N_ions, dtype=bool)  # all Li sites start occupied
            # map ion id -> site index (initially ion i sits on site i)
            self.ion_site = np.arange(self.N_ions, dtype=int)

            # ------------------------------------------------------------------
            # Store initial Cartesian positions for MSD
            # ------------------------------------------------------------------
            self.initial_positions = np.array(
                [self.struct.cart_coords[idx] for idx in self.li_site_indices]
            )
            self.current_positions = self.initial_positions.copy()

            # Pre‑compute volume (Å³) for conductivity later
            self.volume = self.struct.lattice.volume  # Å³

        def _get_li_site_indices(self):
            """Return list of indices in the structure that correspond to Li sites."""
            li_idxs = []
            for i, site in enumerate(self.struct):
                if any(el.symbol == "Li" for el in site.species.elements):
                    li_idxs.append(i)
            return li_idxs

        def _build_neighbor_dict(self):
            """For each Li site, store list of neighboring Li site indices within cutoff."""
            self.neighbor_dict = {i: [] for i in range(self.N_ions)}
            # Use pymatgen neighbor finder on the full structure
            for i, site_idx in enumerate(self.li_site_indices):
                site = self.struct[site_idx]
                neighbors = self.struct.get_neighbors(site, self.cutoff)
                for nb in neighbors:
                    # consider only Li neighbors
                    if any(el.symbol == "Li" for el in nb.species.elements):
                        # find the index of this neighbor in li_site_indices
                        nb_global_idx = nb.index
                        try:
                            nb_local_idx = self.li_site_indices.index(nb_global_idx)
                            if nb_local_idx != i:
                                self.neighbor_dict[i].append(nb_local_idx)
                        except ValueError:
                            continue  # neighbor not a Li site in our list

        def _occupied_neighbor_count(self, site_local_idx):
            """Count occupied Li neighbors of a given site."""
            neigh = self.neighbor_dict[site_local_idx]
            return np.sum(self.occupied[neigh])

        def _average_occupancy(self):
            """Mean number of occupied neighbors per Li site."""
            counts = np.array([self._occupied_neighbor_count(i) for i in range(self.N_ions)])
            return counts.mean()

        def _list_possible_hops(self):
            """Return list of (ion_id, from_idx, to_idx) for all allowed hops."""
            hops = []
            for ion_id, from_idx in enumerate(self.ion_site):
                for to_idx in self.neighbor_dict[from_idx]:
                    if not self.occupied[to_idx]:          # destination must be vacant
                        hops.append((ion_id, from_idx, to_idx))
            return hops

        def run(self):
            """Perform KMC until target_time is reached."""
            while self.time < target_time:
                hops = self._list_possible_hops()
                if not hops:
                    break  # no possible moves

                avg_occ = self._average_occupancy()

                # Compute rates for each hop
                rates = np.empty(len(hops))
                for i, (_, _, to_idx) in enumerate(hops):
                    n_B = self._occupied_neighbor_count(to_idx)   # occupied neighbors of destination
                    Ea_eff = self.base_Ea + self.alpha * (n_B - avg_occ)
                    rates[i] = self.nu * np.exp(-Ea_eff / (kB_eV * self.T))

                total_rate = rates.sum()
                if total_rate <= 0:
                    break

                # Choose hop
                r = np.random.rand() * total_rate
                cumulative = np.cumsum(rates)
                hop_idx = np.searchsorted(cumulative, r)
                ion_id, from_idx, to_idx = hops[hop_idx]

                # Update occupancy and ion mapping
                self.occupied[from_idx] = False
                self.occupied[to_idx] = True
                self.ion_site[ion_id] = to_idx

                # Update position of this ion
                self.current_positions[ion_id] = self.struct.cart_coords[self.li_site_indices[to_idx]]

                # Advance time
                self.time += -np.log(np.random.rand()) / total_rate

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient D, and ionic conductivity."""
            # Mean squared displacement (Å²)
            displacements = self.current_positions - self.initial_positions
            msd = np.mean(np.sum(displacements**2, axis=1))

            # Diffusion coefficient D (cm²/s)
            D_cm2_s = msd / (2 * dim * self.time) * 1e-16   # Å² → cm²

            # Number density of Li ions (cm⁻³)
            n_ions = self.N_ions / (self.volume * 1e-24)    # Å³ → cm³

            # Conductivity σ = n·e²·D / (kB·T)  (S/cm)
            sigma = n_ions * e_charge**2 * D_cm2_s / (kB_J * self.T)

            print(f"Conductivity: {sigma:.3e} S/cm")
            return {"MSD (Å²)": msd, "D (cm²/s)": D_cm2_s, "sigma (S/cm)": sigma}

    # ----------------------------------------------------------------------
    # Run the simulation
    # ----------------------------------------------------------------------
    kmc = KMCSimulator(structure, cutoff, base_Ea, alpha, nu, T)
    kmc.run()
    kmc.calculate_properties()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
