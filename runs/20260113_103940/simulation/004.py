"""
KOKOA Simulation #4
Generated: 2026-01-13 10:41:31
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260113_103940')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from scipy.constants import k as kB, e as e_charge, N_A
    from scipy.spatial import cKDTree
    from pymatgen.core import Structure
    import matplotlib.pyplot as plt

    # ------------------- Configuration -------------------
    _cutoff = 4.0          # Å, neighbor search radius for hops and density
    _E0 = 0.30             # eV, base activation energy
    _alpha = 0.05          # eV/Å, distance penalty
    _gamma = 0.01          # eV, neighbor‑count penalty
    _beta = 0.01           # eV per ion, screening term
    _nu0 = 1e13            # Hz, attempt frequency
    _T = 300.0             # K, temperature
    _target_time = 5e-9    # s, fixed simulation length
    # -----------------------------------------------------

    # ------------------- Load Structure -----------------
    structure = Structure.from_file(_CIF_PATH)          # _CIF_PATH is pre‑injected
    # enlarge to reduce finite‑size effects (optional)
    structure.make_supercell([2, 2, 2])
    _vol_cm3 = structure.lattice.volume * 1e-24        # Å³ → cm³
    # -----------------------------------------------------

    class KMCSimulator:
        def __init__(self, struct):
            self.struct = struct

            # indices of Li sites in the structure
            self.li_site_indices = [i for i, site in enumerate(struct) if "Li" in site.species_string]
            self.n_sites = len(self.li_site_indices)

            # Cartesian coordinates of Li sites (Å)
            self.site_coords = np.array([struct[i].coords for i in self.li_site_indices])

            # Build KD‑tree for neighbor searches
            self.kdtree = cKDTree(self.site_coords)

            # adjacency list: possible hop destinations within _cutoff (excluding self)
            self.adj = {}
            for i, coord in enumerate(self.site_coords):
                neigh = self.kdtree.query_ball_point(coord, _cutoff)
                neigh = [j for j in neigh if j != i]   # exclude self
                self.adj[i] = neigh

            # Occupancy: which ion (0…n_ions‑1) occupies each site, -1 if empty
            # Initially each Li site is occupied by a distinct ion
            self.occupancy = np.arange(self.n_sites, dtype=int)

            # Map ion → current site index
            self.ion_site = np.arange(self.n_sites, dtype=int)

            self.n_ions = self.n_sites

            # Record initial positions for MSD
            self.initial_coords = self.site_coords[self.ion_site].copy()

            # Simulation clock
            self.time = 0.0

            # Store MSD vs time for optional plotting
            self.time_record = [0.0]
            self.msd_record = [0.0]

        def _local_density(self, dest_idx):
            """Number of Li ions within _cutoff of destination site (including the ion that will occupy it)."""
            neigh = self.kdtree.query_ball_point(self.site_coords[dest_idx], _cutoff)
            # count occupied sites among neighbours
            occ = self.occupancy[neigh]
            return np.sum(occ >= 0)

        def _neighbor_count(self, dest_idx):
            """Number of occupied neighbour sites around destination (excluding the moving ion)."""
            neigh = self.adj[dest_idx]
            occ = self.occupancy[neigh]
            return np.sum(occ >= 0)

        def _compute_rates(self):
            """Return list of (ion, origin, destination, rate) for all allowed hops."""
            rates = []
            for ion in range(self.n_ions):
                origin = self.ion_site[ion]
                for dest in self.adj[origin]:
                    if self.occupancy[dest] != -1:   # destination already occupied
                        continue
                    # geometric quantities
                    dist = np.linalg.norm(self.site_coords[origin] - self.site_coords[dest])
                    neighbor_cnt = self._neighbor_count(dest)
                    local_den = self._local_density(dest)

                    # activation energy (eV)
                    E_act = (_E0 +
                             _alpha * dist +
                             _gamma * neighbor_cnt -
                             _beta * local_den)

                    # rate (Hz)
                    rate = _nu0 * np.exp(-E_act / (kB * _T / e_charge))  # convert kB*T to eV
                    rates.append((ion, origin, dest, rate))
            return rates

        def run_step(self):
            """Perform a single KMC event."""
            rates = self._compute_rates()
            if not rates:
                # No possible moves; stop simulation
                self.time = _target_time
                return

            # Build cumulative distribution
            rate_vals = np.array([r[3] for r in rates])
            total_rate = rate_vals.sum()
            cum_rates = np.cumsum(rate_vals)

            # Choose event
            r = np.random.random() * total_rate
            idx = np.searchsorted(cum_rates, r)
            ion, origin, dest, _ = rates[idx]

            # Execute hop
            self.occupancy[origin] = -1
            self.occupancy[dest] = ion
            self.ion_site[ion] = dest

            # Advance time (Gillespie)
            dt = -np.log(np.random.random()) / total_rate
            self.time += dt

            # Record MSD
            disp = self.site_coords[self.ion_site] - self.initial_coords
            msd = np.mean(np.sum(disp**2, axis=1))
            self.time_record.append(self.time)
            self.msd_record.append(msd)

        def run(self):
            """Run KMC until target simulation time is reached."""
            while self.time < _target_time:
                self.run_step()

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient, and ionic conductivity."""
            # Use final MSD and total simulation time
            final_disp = self.site_coords[self.ion_site] - self.initial_coords
            msd = np.mean(np.sum(final_disp**2, axis=1))          # Å²
            msd_cm2 = msd * 1e-16                                 # Å² → cm²
            D = msd_cm2 / (6.0 * self.time)                       # cm²/s

            # Number density of Li (ions per cm³)
            n_density = self.n_ions / _vol_cm3                     # cm⁻³

            # Conductivity (S/cm) via Nernst‑Einstein
            sigma = (n_density * e_charge**2 * D) / (kB * _T)      # S/cm

            print(f"Conductivity: {sigma:.3e} S/cm")
            return sigma

    # ------------------- Execute Simulation -------------------
    sim = KMCSimulator(structure)
    sim.run()
    sim.calculate_properties()
    # Optional: plot MSD vs time
    plt.figure()
    plt.plot(sim.time_record, sim.msd_record)
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (Å$^2$)')
    plt.title('Mean Squared Displacement')
    plt.show()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
