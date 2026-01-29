"""
KOKOA Simulation #6
Generated: 2026-01-13 10:42:27
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

    # ------------------- Configuration -------------------
    _target_time = 5e-9          # s, fixed simulation length
    _T = 300.0                  # K, temperature
    _nu0 = 1e13                 # Hz, attempt frequency
    _E0 = 0.30                  # eV, base activation energy
    _alpha = 0.05               # eV/Å, distance penalty
    _gamma = 0.01               # eV, neighbor‑count penalty (unused in this simple model)
    _rate_floor = 1e-20         # Hz, minimal non‑zero rate
    _cutoff = 4.0               # Å, neighbor search radius for hops
    # -----------------------------------------------------

    # ------------------- Load Structure -----------------
    structure = Structure.from_file(_CIF_PATH)          # _CIF_PATH is pre‑injected
    structure.make_supercell([2, 2, 2])                  # enlarge to reduce finite‑size effects
    _vol_cm3 = structure.lattice.volume * 1e-24         # Å³ → cm³
    # -----------------------------------------------------

    class KMCSimulator:
        def __init__(self, struct):
            self.struct = struct
            # All site coordinates (Å)
            self.coords = np.array([site.coords for site in struct])
            self.n_sites = len(self.coords)

            # Identify mobile Li sites (occupied at start)
            self.li_sites = [i for i, site in enumerate(struct) if "Li" in site.species_string]
            self.n_ions = len(self.li_sites)

            # Occupancy mask: True = occupied by Li
            self.occupied = np.zeros(self.n_sites, dtype=bool)
            self.occupied[self.li_sites] = True

            # Build neighbor list for each site using KDTree
            tree = cKDTree(self.coords)
            self.neighbor_lists = []
            for i in range(self.n_sites):
                idx = tree.query_ball_point(self.coords[i], _cutoff)
                idx.remove(i)                     # exclude self
                self.neighbor_lists.append(np.array(idx, dtype=int))

            # Record initial positions for MSD
            self.initial_positions = self.coords[self.li_sites].copy()
            # Store trajectory of positions (optional, here we just keep final)
            self.final_positions = self.coords[self.li_sites].copy()

            self.time = 0.0

        def _compute_hop_rates(self):
            """Return arrays: from_idx, to_idx, rates for all allowed hops."""
            from_list = []
            to_list = []
            rate_list = []

            for i in self.li_sites:                     # occupied sites
                neigh = self.neighbor_lists[i]
                vacant = neigh[~self.occupied[neigh]]    # only vacant destinations
                if vacant.size == 0:
                    continue
                dists = np.linalg.norm(self.coords[vacant] - self.coords[i], axis=1)
                # Simple barrier model
                barriers = _E0 + _alpha * dists
                raw_rates = _nu0 * np.exp(-barriers / (kB * _T / e_charge))  # convert kB*T to eV
                rates = np.maximum(raw_rates, _rate_floor)

                from_list.append(np.full(vacant.shape, i, dtype=int))
                to_list.append(vacant)
                rate_list.append(rates)

            if not from_list:
                return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

            from_arr = np.concatenate(from_list)
            to_arr = np.concatenate(to_list)
            rates_arr = np.concatenate(rate_list)
            return from_arr, to_arr, rates_arr

        def run_step(self):
            """Perform a single KMC step."""
            from_idx, to_idx, rates = self._compute_hop_rates()
            total_rate = rates.sum()
            if total_rate == 0.0:
                # No possible moves; terminate simulation
                self.time = _target_time
                return

            # Choose hop
            r = np.random.rand() * total_rate
            cumulative = np.cumsum(rates)
            hop_index = np.searchsorted(cumulative, r)
            i_from = from_idx[hop_index]
            i_to = to_idx[hop_index]

            # Update occupancy
            self.occupied[i_from] = False
            self.occupied[i_to] = True

            # Update list of Li sites (occupied indices)
            self.li_sites.remove(i_from)
            self.li_sites.append(i_to)

            # Advance time
            dt = -np.log(np.random.rand()) / total_rate
            self.time += dt

        def run(self):
            """Run KMC until target time is reached."""
            while self.time < _target_time:
                self.run_step()
            # Store final positions of Li ions
            self.final_positions = self.coords[self.li_sites].copy()

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient, and ionic conductivity."""
            displacements = self.final_positions - self.initial_positions
            msd = np.mean(np.sum(displacements**2, axis=1))   # Å²
            msd_cm2 = msd * 1e-16                              # Å² → cm²

            D = msd_cm2 / (6 * self.time)                     # cm²/s

            # Number density of charge carriers (ions per cm³)
            n = self.n_ions / _vol_cm3                         # cm⁻³

            sigma = n * e_charge**2 * D / (kB * _T)            # S/cm

            print(f"Conductivity: {sigma:.3e} S/cm")
            return {"MSD (Å²)": msd, "D (cm²/s)": D, "sigma (S/cm)": sigma}

    # ------------------- Execute Simulation -----------------
    sim = KMCSimulator(structure)
    sim.run()
    sim.calculate_properties()
    # ---------------------------------------------------------
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
