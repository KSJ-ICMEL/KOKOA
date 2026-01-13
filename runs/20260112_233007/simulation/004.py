"""
KOKOA Simulation #4
Generated: 2026-01-12 23:32:08
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
    from scipy.constants import epsilon_0, e as e_charge, k as kB_J
    from pymatgen.core import Structure

    # ----------------------------------------------------------------------
    # Constants (SI unless noted)
    # ----------------------------------------------------------------------
    kB_eV = kB_J / e_charge          # eV/K
    T = 300.0                         # K
    nu = 1e13                         # 1/s, attempt frequency
    base_Ea = 0.30                    # eV, base barrier
    alpha = 0.02                      # eV per occupied neighbor
    beta = 0.10                       # eV·Å, Coulomb scaling factor
    epsilon_r = 10.0                  # relative dielectric constant
    target_time = 1e-8                # s, fixed simulation time
    cutoff = 4.0                      # Å, neighbor search radius
    dim = 3                           # dimensionality for MSD

    # ----------------------------------------------------------------------
    # Load structure
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell([2, 2, 2])   # modest supercell for enough sites

    class KMCSimulator:
        def __init__(self, struct, cutoff, base_Ea, alpha, beta,
                     epsilon_r, nu, T):
            self.struct = struct
            self.cutoff = cutoff
            self.base_Ea = base_Ea
            self.alpha = alpha
            self.beta = beta
            self.epsilon_r = epsilon_r
            self.nu = nu
            self.T = T
            self.time = 0.0

            # ------------------------------------------------------------------
            # Identify Li sites and build neighbor list (periodic)
            # ------------------------------------------------------------------
            li_sites = [s for s in struct.sites if s.species_string == "Li"]
            self.site_coords = np.array([s.coords for s in li_sites])   # Å
            self.num_sites = len(self.site_coords)

            # neighbor indices and distances for each site
            self.neighbor_indices = [[] for _ in range(self.num_sites)]
            self.neighbor_distances = [[] for _ in range(self.num_sites)]

            # use lattice to compute periodic distances
            for i in range(self.num_sites):
                for j in range(i + 1, self.num_sites):
                    dist = struct.lattice.get_distance_and_image(
                        self.site_coords[i], self.site_coords[j])[0]
                    if dist <= self.cutoff:
                        self.neighbor_indices[i].append(j)
                        self.neighbor_distances[i].append(dist)
                        self.neighbor_indices[j].append(i)
                        self.neighbor_distances[j].append(dist)

            # ------------------------------------------------------------------
            # Initialise occupancy (introduce a small vacancy fraction)
            # ------------------------------------------------------------------
            rng = np.random.default_rng(42)
            vacancy_fraction = 0.05
            occupied = np.ones(self.num_sites, dtype=bool)
            vacancy_indices = rng.choice(self.num_sites,
                                         size=int(vacancy_fraction * self.num_sites),
                                         replace=False)
            occupied[vacancy_indices] = False
            self.occupancy = occupied

            # map site -> ion id (-1 if vacant)
            self.ion_id_per_site = -np.ones(self.num_sites, dtype=int)
            self.ion_ids = np.where(self.occupancy)[0]
            for idx, site in enumerate(self.ion_ids):
                self.ion_id_per_site[site] = idx

            self.num_ions = len(self.ion_ids)

            # current site index for each ion
            self.ion_site = np.copy(self.ion_ids)

            # store initial positions for MSD
            self.initial_positions = self.site_coords[self.ion_site].copy()   # Å

        def _compute_rate(self, origin_idx, dest_idx):
            """Rate for hop origin -> dest (origin occupied, dest vacant)."""
            # number of occupied neighbours of origin (excluding destination)
            neigh_idxs = self.neighbor_indices[origin_idx]
            neigh_dists = self.neighbor_distances[origin_idx]
            n_occ = 0
            coulomb_term = 0.0
            for n_idx, dist in zip(neigh_idxs, neigh_dists):
                if n_idx == dest_idx:
                    continue
                if self.occupancy[n_idx]:
                    n_occ += 1
                    coulomb_term += self.beta / dist   # eV (beta in eV·Å)

            Ea = self.base_Ea + self.alpha * n_occ + coulomb_term
            if Ea < 0.0:
                Ea = 0.0
            rate = self.nu * np.exp(-Ea / (kB_eV * self.T))
            return rate

        def run_step(self):
            """Perform a single KMC event."""
            possible_hops = []
            rates = []

            # enumerate all occupied→vacant neighbour pairs
            for i in range(self.num_sites):
                if not self.occupancy[i]:
                    continue
                for j, _ in zip(self.neighbor_indices[i], self.neighbor_distances[i]):
                    if self.occupancy[j]:
                        continue
                    rate = self._compute_rate(i, j)
                    if rate > 0.0:
                        possible_hops.append((i, j))
                        rates.append(rate)

            if not rates:
                # no possible moves
                self.time = target_time
                return

            rates = np.array(rates)
            total_rate = rates.sum()
            # choose hop
            r = np.random.rand() * total_rate
            cumulative = np.cumsum(rates)
            hop_idx = np.searchsorted(cumulative, r)
            origin, dest = possible_hops[hop_idx]

            # advance time
            self.time += -np.log(np.random.rand()) / total_rate

            # execute hop
            ion_id = self.ion_id_per_site[origin]
            self.ion_id_per_site[origin] = -1
            self.ion_id_per_site[dest] = ion_id
            self.occupancy[origin] = False
            self.occupancy[dest] = True
            self.ion_site[ion_id] = dest

        def run_until_target(self):
            while self.time < target_time:
                self.run_step()

        def calculate_properties(self):
            # final positions of ions
            final_positions = self.site_coords[self.ion_site]   # Å
            displacements = final_positions - self.initial_positions
            msd = np.mean(np.sum(displacements**2, axis=1))    # Å^2

            # diffusion coefficient (Einstein relation)
            D_angs2_s = msd / (2 * dim * self.time)            # Å^2/s
            D_m2_s = D_angs2_s * 1e-20                         # m^2/s

            # carrier concentration (ions per m^3)
            volume_ang3 = self.struct.lattice.volume           # Å^3
            volume_m3 = volume_ang3 * 1e-30
            n = self.num_ions / volume_m3                      # m^-3

            # conductivity (S/m) via Nernst–Einstein
            sigma_S_m = (e_charge**2 * n * D_m2_s) / (kB_J * self.T)
            sigma_S_cm = sigma_S_m * 0.01                      # S/cm

            print(f"Conductivity: {sigma_S_cm} S/cm")
            return sigma_S_cm

    # ----------------------------------------------------------------------
    # Execute simulation
    # ----------------------------------------------------------------------
    sim = KMCSimulator(structure, cutoff, base_Ea, alpha, beta,
                       epsilon_r, nu, T)
    sim.run_until_target()
    sim.calculate_properties()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
