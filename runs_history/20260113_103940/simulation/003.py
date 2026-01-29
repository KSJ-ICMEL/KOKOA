"""
KOKOA Simulation #3
Generated: 2026-01-13 10:41:07
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
    _cutoff = 4.0          # Å, neighbor search radius
    _E0 = 0.30             # eV, base activation energy
    _alpha = 0.05          # eV/Å, distance penalty
    _gamma = 0.01          # eV, neighbor‑count penalty
    _nu0 = 1e13            # Hz, attempt frequency
    _T = 300.0             # K, temperature
    _target_time = 5e-9    # s, fixed simulation length
    # -----------------------------------------------------

    # ------------------- Load Structure -----------------
    structure = Structure.from_file(_CIF_PATH)          # _CIF_PATH is pre‑injected
    structure.make_supercell([2, 2, 2])                 # enlarge to reduce finite size effects
    _vol_cm3 = structure.lattice.volume * 1e-24        # Å³ → cm³
    # -----------------------------------------------------

    class KMCSimulator:
        def __init__(self, struct):
            self.struct = struct
            # identify Li sites
            self.li_site_indices = [i for i, site in enumerate(struct) if "Li" in site.species_string]
            self.n_ions = len(self.li_site_indices)

            # site positions (Cartesian Å)
            self.site_coords = np.array([struct[i].coords for i in self.li_site_indices])

            # adjacency list: possible hops within cutoff
            self.adj = {i: [] for i in range(self.n_ions)}
            for i, coord in enumerate(self.site_coords):
                dists = np.linalg.norm(self.site_coords - coord, axis=1)
                neigh = np.where((dists > 1e-3) & (dists <= _cutoff))[0]
                for j in neigh:
                    self.adj[i].append((j, dists[j]))

            # initial occupancy: one ion per site
            self.ion_site = np.arange(self.n_ions)          # ion i sits on site i
            self.positions = self.site_coords[self.ion_site].copy()  # Å

            self.time = 0.0                                 # s
            self._eV_to_J = e_charge                        # 1 eV = e_charge J

        def _neighbor_counts(self):
            """Return array of neighbor counts (excluding self) for each occupied site."""
            tree = cKDTree(self.positions)
            counts = np.empty(self.n_ions, dtype=int)
            for i, pos in enumerate(self.positions):
                idx = tree.query_ball_point(pos, _cutoff)
                counts[i] = len(idx) - 1   # exclude the ion itself
            return counts

        def run_step(self):
            """Perform a single kMC event using the Gillespie algorithm."""
            # neighbor counts per ion (based on current positions)
            n_counts = self._neighbor_counts()

            # build list of possible events and their rates
            rates = []
            events = []   # (ion_index, target_site)
            for ion_idx, site_idx in enumerate(self.ion_site):
                neigh_list = self.adj[site_idx]
                if not neigh_list:
                    continue
                n_neigh = n_counts[ion_idx]          # local Li crowding at origin site
                for tgt, dist in neigh_list:
                    # activation energy with distance and neighbor penalty
                    Ea_eV = _E0 + _alpha * dist + _gamma * n_neigh
                    k = _nu0 * np.exp(-Ea_eV * self._eV_to_J / (kB * _T))
                    rates.append(k)
                    events.append((ion_idx, tgt))

            if not rates:
                # no possible moves
                self.time = _target_time
                return

            rates = np.array(rates)
            total_rate = rates.sum()
            # time increment
            r = np.random.random()
            dt = -np.log(r) / total_rate
            self.time += dt

            # select event
            r2 = np.random.random() * total_rate
            cum = np.cumsum(rates)
            idx = np.searchsorted(cum, r2)
            ion_idx, tgt_site = events[idx]

            # update occupancy and positions
            self.ion_site[ion_idx] = tgt_site
            self.positions[ion_idx] = self.site_coords[tgt_site]

        def run(self):
            """Run the kMC simulation until the target time is reached."""
            while self.time < _target_time:
                self.run_step()

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient, and ionic conductivity."""
            # displacement from initial positions
            disp = self.positions - self.site_coords[np.arange(self.n_ions)]
            msd_A2 = np.mean(np.sum(disp**2, axis=1))          # Å²
            msd_cm2 = msd_A2 * 1e-16                           # cm²

            D_cm2_s = msd_cm2 / (6.0 * self.time)              # Einstein relation

            # number density of Li (cm⁻³)
            n_density = self.n_ions / _vol_cm3

            # Nernst‑Einstein conductivity
            sigma = D_cm2_s * n_density * e_charge**2 / (kB * _T)   # S·cm⁻¹

            print(f"Conductivity: {sigma:.3e} S/cm")
            return sigma

    # ------------------- Execute Simulation -----------------
    sim = KMCSimulator(structure)
    sim.run()
    sim.calculate_properties()
    # ---------------------------------------------------------
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
