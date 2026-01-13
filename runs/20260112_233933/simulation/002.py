"""
KOKOA Simulation #2
Generated: 2026-01-12 23:40:39
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
    from scipy.constants import elementary_charge as e, Boltzmann as k_B
    from pymatgen.core import Structure

    # ----------------------------------------------------------------------
    # 1. Load crystal structure (CIF path is provided in the variable _CIF_PATH)
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)

    # ----------------------------------------------------------------------
    # 2. kMC simulator with site‑dependent activation‑energy scaling
    # ----------------------------------------------------------------------
    class KMCSimulator:
        def __init__(
            self,
            structure: Structure,
            mobile_species: str = "Li",
            temperature: float = 300.0,          # K
            attempt_freq: float = 1e12,          # Hz
            activation_energy: float = 0.35,     # eV (global)
            cutoff: float = 4.0,                 # Å, neighbour search radius
            min_alpha: float = 0.9,              # scaling limits
            max_alpha: float = 1.2,
            target_time: float = 1e-8,           # s (fixed)
        ):
            self.structure = structure
            self.mobile_species = mobile_species
            self.T = temperature
            self.nu0 = attempt_freq
            self.Ea_global = activation_energy * e          # J
            self.cutoff = cutoff
            self.min_alpha = min_alpha
            self.max_alpha = max_alpha
            self.target_time = target_time

            self.lattice = structure.lattice
            self.n_sites = len(structure)
            self._build_neighbourhood()
            self.coord_numbers = self._compute_coordination_numbers()
            self.max_cn = self.coord_numbers.max()

            # occupancy: True if site holds a mobile ion
            self.occupied = np.array([site.specie.symbol == mobile_species for site in structure])
            self.ion_indices = np.where(self.occupied)[0]          # site ids of ions
            self.n_ions = len(self.ion_indices)

            # store initial Cartesian positions of each ion
            self.initial_pos = self.lattice.get_cartesian_coords(self.ion_indices)
            self.current_pos = self.initial_pos.copy()

            # time bookkeeping
            self.time = 0.0

            # trajectory storage for MSD (store positions after each event)
            self.pos_history = [self.current_pos.copy()]
            self.time_history = [self.time]

        # ------------------------------------------------------------------
        # Build neighbour list within cutoff (static lattice)
        # ------------------------------------------------------------------
        def _build_neighbourhood(self):
            self.neighbours = [[] for _ in range(self.n_sites)]
            for i in range(self.n_sites):
                dists = self.lattice.get_all_distances(i, range(self.n_sites))
                neigh = [j for j, d in enumerate(dists) if 0 < d <= self.cutoff]
                self.neighbours[i] = neigh

        # ------------------------------------------------------------------
        # Coordination numbers (simple neighbour count)
        # ------------------------------------------------------------------
        def _compute_coordination_numbers(self):
            cn = np.zeros(self.n_sites, dtype=int)
            for i, neigh in enumerate(self.neighbours):
                cn[i] = len(neigh)
            return cn

        # ------------------------------------------------------------------
        # Local activation energy for a hop i → j
        # ------------------------------------------------------------------
        def _local_Ea(self, i, j):
            cn_i = self.coord_numbers[i]
            cn_j = self.coord_numbers[j]
            avg_cn = (cn_i + cn_j) / 2.0
            # linear mapping avg_cn → alpha
            alpha = (self.min_alpha +
                     (self.max_alpha - self.min_alpha) *
                     (self.max_cn - avg_cn) / self.max_cn)
            return alpha * self.Ea_global

        # ------------------------------------------------------------------
        # Rate for a specific hop i → j (Arrhenius)
        # ------------------------------------------------------------------
        def _rate(self, i, j):
            Ea_loc = self._local_Ea(i, j)
            return self.nu0 * np.exp(-Ea_loc / (k_B * self.T))

        # ------------------------------------------------------------------
        # Perform one kMC step (Gillespie)
        # ------------------------------------------------------------------
        def run_step(self):
            # collect all possible hops and their rates
            possible_hops = []
            rates = []
            for i in np.where(self.occupied)[0]:
                for j in self.neighbours[i]:
                    if not self.occupied[j]:               # destination empty
                        r = self._rate(i, j)
                        possible_hops.append((i, j))
                        rates.append(r)

            if not rates:                                 # no moves possible
                self.time = self.target_time
                return

            rates = np.array(rates)
            total_rate = rates.sum()
            # time increment
            dt = -np.log(np.random.random()) / total_rate
            self.time += dt

            # choose hop
            cum = np.cumsum(rates)
            rnd = np.random.random() * total_rate
            idx = np.searchsorted(cum, rnd)
            i, j = possible_hops[idx]

            # execute hop: update occupancy
            self.occupied[i] = False
            self.occupied[j] = True

            # update ion position tracking
            # find which ion was at site i
            ion_idx = np.where(self.ion_indices == i)[0][0]
            # move its Cartesian coordinate
            new_cart = self.lattice.get_cartesian_coords(j)
            self.current_pos[ion_idx] = new_cart
            # update stored site id for that ion
            self.ion_indices[ion_idx] = j

            # store trajectory
            self.pos_history.append(self.current_pos.copy())
            self.time_history.append(self.time)

        # ------------------------------------------------------------------
        # Run until target_time is reached
        # ------------------------------------------------------------------
        def run(self, max_steps: int = 100000):
            steps = 0
            while self.time < self.target_time and steps < max_steps:
                self.run_step()
                steps += 1

        # ------------------------------------------------------------------
        # Compute MSD, diffusion coefficient, and ionic conductivity
        # ------------------------------------------------------------------
        def calculate_properties(self):
            # final positions
            final_pos = self.current_pos
            # displacement vectors (apply minimum image convention)
            disp = final_pos - self.initial_pos
            # convert to fractional, wrap, then back to Cartesian
            frac_disp = self.lattice.get_fractional_coords(disp)
            frac_disp = frac_disp - np.rint(frac_disp)          # wrap into [-0.5,0.5]
            disp = self.lattice.get_cartesian_coords(frac_disp)

            msd = np.mean(np.sum(disp**2, axis=1))              # Å^2
            msd_m2 = msd * 1e-20                                 # convert Å^2 → m^2
            D = msd_m2 / (6 * self.time)                        # m^2/s

            # number density of mobile ions (ions per m^3)
            volume = self.lattice.volume * 1e-30                 # Å^3 → m^3
            n = self.n_ions / volume

            sigma = n * e**2 * D / (k_B * self.T)                # S/m
            sigma_cmc = sigma * 1e2                               # S/cm

            print(f"Conductivity: {sigma_cmc:.6e} S/cm")
            return {"MSD (Å^2)": msd, "D (m^2/s)": D, "sigma (S/cm)": sigma_cmc}

    # ----------------------------------------------------------------------
    # 3. Execute simulation
    # ----------------------------------------------------------------------
    sim = KMCSimulator(
        structure=structure,
        mobile_species="Li",
        temperature=300.0,
        attempt_freq=1e12,
        activation_energy=0.35,   # eV
        cutoff=4.0,
        min_alpha=0.9,
        max_alpha=1.2,
        target_time=1e-8
    )

    sim.run()
    sim.calculate_properties()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
