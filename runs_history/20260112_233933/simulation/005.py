"""
KOKOA Simulation #5
Generated: 2026-01-12 23:41:44
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_233933')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from scipy.constants import elementary_charge as e, Boltzmann as k_B, Avogadro
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import VoronoiNN

    # ----------------------------------------------------------------------
    # 1. Load crystal structure (CIF path is provided in the variable _CIF_PATH)
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)

    # ----------------------------------------------------------------------
    # 2. kMC simulator
    # ----------------------------------------------------------------------
    class KMCSimulator:
        def __init__(
            self,
            structure: Structure,
            mobile_species: str = "Li",
            temperature: float = 300.0,          # K
            attempt_freq: float = 1e12,          # Hz
            activation_energy: float = 0.35,     # eV (global baseline)
            cutoff: float = 4.0,                 # Å, neighbour search radius
            alpha: float = 0.02,                 # eV per coordination deviation
            target_time: float = 1e-8,           # s (fixed)
        ):
            self.structure = structure
            self.mobile_species = mobile_species
            self.T = temperature
            self.nu0 = attempt_freq
            self.Ea_global = activation_energy * e          # J
            self.cutoff = cutoff
            self.alpha = alpha * e                           # J per CN deviation
            self.target_time = target_time

            # ------------------------------------------------------------------
            # Identify mobile-ion sites and initialise positions
            # ------------------------------------------------------------------
            self.mobile_indices = [
                i for i, site in enumerate(structure.sites)
                if mobile_species in site.species_string
            ]
            if not self.mobile_indices:
                raise ValueError(f"No sites with species '{mobile_species}' found.")
            self.N = len(self.mobile_indices)

            # current site of each ion (index into structure.sites)
            self.ion_sites = np.array(self.mobile_indices, copy=True)

            # store initial Cartesian coordinates for MSD
            self.initial_coords = np.array(
                [structure.sites[i].coords for i in self.ion_sites]
            )

            # ------------------------------------------------------------------
            # Pre‑compute neighbour list for every site (within cutoff)
            # ------------------------------------------------------------------
            self.neighbor_dict = {i: [] for i in range(len(structure))}
            for i, site in enumerate(structure.sites):
                neighbors = structure.get_neighbors(site, self.cutoff, include_index=True)
                self.neighbor_dict[i] = [nbr[2] for nbr in neighbors]  # neighbour site indices

            # ------------------------------------------------------------------
            # Compute site‑specific activation energies from coordination numbers
            # ------------------------------------------------------------------
            vnn = VoronoiNN(cutoff=self.cutoff)
            cn_list = []
            for i, site in enumerate(structure.sites):
                cn = len(vnn.get_nn_info(structure, i))
                cn_list.append(cn)
            cn_arr = np.array(cn_list, dtype=float)
            cn_avg = cn_arr.mean()
            # site_Ea = global + alpha * (cn_avg - cn_i)
            self.site_Ea = self.Ea_global + self.alpha * (cn_avg - cn_arr)  # J per site

            # ------------------------------------------------------------------
            # Prepare list of possible hops (i -> j) for mobile ions only
            # ------------------------------------------------------------------
            self.hop_pairs = []  # list of (origin_site, dest_site)
            for i in self.mobile_indices:
                for j in self.neighbor_dict[i]:
                    # allow hop only to another site that can host a mobile ion
                    # (i.e., any lattice site; vacancy handling is implicit)
                    self.hop_pairs.append((i, j))

            # Pre‑compute rates for each hop pair (will be updated when ions move)
            self._update_hop_rates()

            # simulation bookkeeping
            self.time = 0.0
            self.displacements = np.zeros((self.N, 3))  # cumulative displacement vectors

        def _hop_rate(self, i, j):
            """Rate for a hop from site i to site j (average barrier)."""
            Ea_ij = 0.5 * (self.site_Ea[i] + self.site_Ea[j])
            return self.nu0 * np.exp(-Ea_ij / (k_B * self.T))

        def _update_hop_rates(self):
            """Re‑evaluate rates for all hop pairs."""
            self.hop_rates = np.array([self._hop_rate(i, j) for i, j in self.hop_pairs])
            self.cum_rates = np.cumsum(self.hop_rates)
            self.total_rate = self.cum_rates[-1]

        def run(self):
            """Perform kMC steps until target_time is reached."""
            rng = np.random.default_rng()
            while self.time < self.target_time:
                if self.total_rate == 0:
                    break  # no possible moves
                # time increment
                r = rng.random()
                dt = -np.log(r) / self.total_rate
                self.time += dt
                if self.time > self.target_time:
                    break

                # choose hop
                r2 = rng.random() * self.total_rate
                hop_idx = np.searchsorted(self.cum_rates, r2)
                origin, dest = self.hop_pairs[hop_idx]

                # find which ion currently occupies the origin site
                ion_mask = self.ion_sites == origin
                if not np.any(ion_mask):
                    # origin site empty (should be rare); skip this hop
                    continue
                ion_id = np.where(ion_mask)[0][0]

                # update ion position
                old_coord = self.structure.sites[origin].coords
                new_coord = self.structure.sites[dest].coords
                self.displacements[ion_id] += new_coord - old_coord
                self.ion_sites[ion_id] = dest

                # after move, recompute rates (simple but safe)
                self._update_hop_rates()

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient, and ionic conductivity."""
            # Mean‑squared displacement (Å^2)
            msd = np.mean(np.sum(self.displacements**2, axis=1))
            # Convert to m^2
            msd_m2 = msd * (1e-10)**2

            # Diffusion coefficient D = MSD / (2*d*t)  (d=3)
            D = msd_m2 / (6.0 * self.time)  # m^2/s

            # Number density of mobile ions (cm^-3)
            vol_A3 = self.structure.lattice.volume  # Å^3
            vol_cm3 = vol_A3 * 1e-24
            n = self.N / vol_cm3  # cm^-3

            # Ionic conductivity σ = n * q^2 * D / (k_B * T)
            sigma = n * (e**2) * D / (k_B * self.T)  # S/m
            sigma_sc_cm = sigma * 1e-2  # convert S/m to S/cm

            print(f"Conductivity: {sigma_sc_cm:.5e} S/cm")
            return {"MSD (Å^2)": msd, "D (m^2/s)": D, "sigma (S/cm)": sigma_sc_cm}

    # ----------------------------------------------------------------------
    # 3. Execute simulation
    # ----------------------------------------------------------------------
    sim = KMCSimulator(structure)
    sim.run()
    sim.calculate_properties()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
