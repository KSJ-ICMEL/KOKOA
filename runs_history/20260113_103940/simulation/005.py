"""
KOKOA Simulation #5
Generated: 2026-01-13 10:42:06
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
    from scipy.constants import k as kB, e as e_charge, epsilon_0, N_A
    from scipy.spatial import cKDTree
    from pymatgen.core import Structure

    # ------------------- Configuration -------------------
    _target_time = 5e-9          # s, fixed simulation length
    _T = 300.0                  # K, temperature
    _nu0 = 1e13                 # Hz, attempt frequency
    _E0 = 0.30                  # eV, base activation energy
    _alpha = 0.05               # eV/Å, distance penalty
    _gamma = 0.01               # eV, neighbor‑count penalty
    _epsilon_r = 10.0           # relative dielectric constant
    _lambda = 2.0               # Å, screening length
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
            # Identify mobile Li sites
            self.li_indices = [i for i, site in enumerate(struct) if "Li" in site.species_string]
            self.n_ions = len(self.li_indices)

            # Site coordinates (Å) for all sites
            self.all_coords = np.array([site.coords for site in struct])
            self.n_sites = len(self.all_coords)

            # Occupancy: -1 = empty, else ion index
            self.occupancy = -np.ones(self.n_sites, dtype=int)
            for ion_idx, site_idx in enumerate(self.li_indices):
                self.occupancy[site_idx] = ion_idx

            # Initial positions of ions (Å)
            self.positions = self.all_coords[self.li_indices].copy()
            self.initial_positions = self.positions.copy()

            # KD‑tree for site‑site neighbor lookup
            self.site_tree = cKDTree(self.all_coords)
            self.neighbor_list = self._build_neighbor_list()

            # Simulation bookkeeping
            self.time = 0.0
            self.displacements_sq = np.zeros(self.n_ions)   # cumulative squared displacement

        def _build_neighbor_list(self):
            """Pre‑compute neighbor sites within _cutoff for each site."""
            neighbor_list = []
            for i in range(self.n_sites):
                idx = self.site_tree.query_ball_point(self.all_coords[i], r=_cutoff)
                # remove self
                idx = [j for j in idx if j != i]
                dists = np.linalg.norm(self.all_coords[idx] - self.all_coords[i], axis=1)
                neighbor_list.append(list(zip(idx, dists)))
            return neighbor_list

        def _coulomb_contribution(self, from_pos, to_pos, ion_charge):
            """Screened Coulomb energy (eV) from all other Li ions at the destination site."""
            # positions of other ions (Å)
            other_pos = np.delete(self.positions, ion_charge, axis=0)  # exclude moving ion
            if other_pos.size == 0:
                return 0.0
            r_ij = np.linalg.norm(other_pos - to_pos, axis=1)  # Å
            # avoid division by zero
            r_ij = np.where(r_ij < 1e-12, 1e-12, r_ij)
            # Coulomb energy in Joule, then convert to eV
            E_J = (e_charge * e_charge) / (4 * np.pi * epsilon_0 * _epsilon_r * r_ij) \
                  * np.exp(-r_ij / _lambda)
            E_eV = E_J / e_charge
            # All Li are +1, so repulsive (add)
            return np.sum(E_eV)

        def _neighbor_count(self, site_idx, exclude_ion=None):
            """Number of Li ions within _cutoff of a given site (excluding a specific ion)."""
            center = self.all_coords[site_idx]
            idx = self.site_tree.query_ball_point(center, r=_cutoff)
            count = 0
            for j in idx:
                occ = self.occupancy[j]
                if occ != -1 and occ != exclude_ion:
                    count += 1
            return count

        def run_step(self):
            """Perform a single kMC step."""
            possible_hops = []
            rates = []

            # Loop over all ions
            for ion_idx, site_idx in enumerate(self.li_indices):
                # current site of this ion
                cur_site = site_idx
                # examine neighbor sites
                for nbr_idx, dr in self.neighbor_list[cur_site]:
                    if self.occupancy[nbr_idx] != -1:   # occupied
                        continue
                    # base activation energy (eV)
                    E_act = _E0 + _alpha * dr
                    # neighbor‑count penalty at destination
                    nb = self._neighbor_count(nbr_idx, exclude_ion=ion_idx)
                    E_act += _gamma * nb
                    # Coulomb contribution from other ions at destination
                    to_pos = self.all_coords[nbr_idx]
                    E_act += self._coulomb_contribution(self.all_coords[cur_site], to_pos, ion_idx)

                    # hopping rate (Hz)
                    rate = _nu0 * np.exp(-E_act / (kB * _T))
                    possible_hops.append((ion_idx, cur_site, nbr_idx, dr))
                    rates.append(rate)

            if not rates:
                # No possible moves (should not happen)
                self.time = _target_time
                return

            rates = np.array(rates)
            total_rate = rates.sum()
            # Choose hop
            r = np.random.rand() * total_rate
            cumulative = np.cumsum(rates)
            hop_idx = np.searchsorted(cumulative, r)
            ion_idx, from_site, to_site, dr = possible_hops[hop_idx]

            # Update occupancy
            self.occupancy[from_site] = -1
            self.occupancy[to_site] = ion_idx
            self.li_indices[ion_idx] = to_site

            # Update ion position
            self.positions[ion_idx] = self.all_coords[to_site]

            # Advance time
            dt = -np.log(np.random.rand()) / total_rate
            self.time += dt

        def run(self):
            """Run KMC until target simulation time is reached."""
            while self.time < _target_time:
                self.run_step()

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient, and ionic conductivity."""
            # Mean‑square displacement (Å²)
            disp = self.positions - self.initial_positions
            msd = np.mean(np.sum(disp**2, axis=1))

            # Diffusion coefficient D (cm²/s)
            D_cm2_s = msd / (6.0 * self.time) * 1e-16   # Å² → cm²

            # Number density of mobile ions (cm⁻³)
            n = self.n_ions / _vol_cm3

            # Conductivity σ = n·e²·D/(kB·T) (S/cm)
            sigma = n * e_charge**2 * D_cm2_s / (kB * _T)

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
