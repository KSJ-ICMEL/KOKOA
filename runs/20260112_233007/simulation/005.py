"""
KOKOA Simulation #5
Generated: 2026-01-12 23:32:25
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
    from scipy.constants import k as kB_J, e as e_charge, epsilon_0, Avogadro
    from pymatgen.core import Structure

    # -------------------------------------------------
    # Constants (SI unless noted)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Load structure
    # -------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    # modest supercell to provide enough mobile sites
    structure.make_supercell([2, 2, 2])

    # -------------------------------------------------
    # Helper functions
    # -------------------------------------------------
    def get_neighbors(struct, idx, radius):
        """Return indices of sites within radius of site idx (excluding idx)."""
        site = struct[idx]
        neighbors = struct.get_sites_in_sphere(site.coords, radius, include_index=False)
        return [n[2] for n in neighbors]  # n[2] is the site index

    def minimum_image(vec, lattice):
        """Apply minimum image convention for a vector in fractional coordinates."""
        frac = lattice.get_fractional_coords(vec)
        frac -= np.rint(frac)  # wrap into [-0.5,0.5)
        return lattice.get_cartesian_coords(frac)

    # -------------------------------------------------
    # KMC Simulator
    # -------------------------------------------------
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

            # Identify mobile species (first occurring species)
            self.mobile_species = struct.species[0].symbol
            self.mobile_indices = [i for i, s in enumerate(struct.species) if s.symbol == self.mobile_species]

            # Track positions (cartesian) of mobile ions
            self.positions = np.array([struct[i].coords for i in self.mobile_indices])
            self.initial_positions = self.positions.copy()

            self.time = 0.0
            self.prev_msd = 0.0
            self.prev_D = 0.0

            # Pre‑compute neighbor lists for all sites
            self.neighbor_list = {i: get_neighbors(struct, i, self.cutoff) for i in range(len(struct))}

        def _select_move(self):
            """Randomly pick a mobile ion and a possible destination site."""
            ion_idx = np.random.randint(len(self.mobile_indices))
            site_idx = self.mobile_indices[ion_idx]
            possible = self.neighbor_list[site_idx]
            if not possible:
                return None, None, None
            dest_idx = np.random.choice(possible)
            return ion_idx, site_idx, dest_idx

        def _barrier(self, site_idx, dest_idx):
            """Simple barrier model with occupation and Coulomb terms."""
            # occupation term: count occupied neighbors of destination
            occ_neighbors = sum(1 for n in self.neighbor_list[dest_idx]
                                if n in self.mobile_indices)
            Ea = self.base_Ea + self.alpha * occ_neighbors

            # Coulomb term (very simplified)
            r = np.linalg.norm(self.struct[dest_idx].coords - self.struct[site_idx].coords)
            if r > 0:
                Ea += self.beta / (self.epsilon_r * r)
            return Ea

        def run_step(self):
            """Perform a single KMC event."""
            move = self._select_move()
            if move[0] is None:
                return  # no possible move

            ion_idx, site_idx, dest_idx = move
            Ea = self._barrier(site_idx, dest_idx)          # eV
            rate = self.nu * np.exp(-Ea / (kB_eV * self.T))  # 1/s
            if rate <= 0:
                return

            # residence time
            dt = -np.log(np.random.random()) / rate
            self.time += dt

            # update structure: move ion
            self.mobile_indices[ion_idx] = dest_idx
            self.positions[ion_idx] = self.struct[dest_idx].coords

        def run_until_target(self):
            """Run KMC until the cumulative time reaches target_time."""
            while self.time < target_time:
                self.run_step()

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient, and ionic conductivity."""
            # Guard against non‑positive simulation time
            sim_time = max(self.time, 1e-15)

            # Displacements with minimum image convention
            disp = self.positions - self.initial_positions
            disp = np.array([minimum_image(d, self.struct.lattice) for d in disp])

            msd = np.mean(np.sum(disp**2, axis=1))

            # Guard against non‑finite MSD
            if not np.isfinite(msd):
                msd = self.prev_msd
            else:
                self.prev_msd = msd

            D = msd / (2 * dim * sim_time)   # m^2/s
            if not np.isfinite(D):
                D = self.prev_D
            else:
                self.prev_D = D

            # Convert D to cm^2/s
            D_cm2 = D * 1e4

            # Carrier concentration (number per m^3)
            volume_m3 = self.struct.lattice.volume * 1e-30   # Å^3 → m^3
            n = len(self.mobile_indices) / volume_m3

            # Conductivity σ = (e^2 * n * D) / (kB * T)   (S/m)
            sigma_S_m = (e_charge**2 * n * D) / (kB_J * self.T)

            # Convert to S/cm
            sigma_S_cm = sigma_S_m * 1e-2

            print(f"Conductivity: {sigma_S_cm:.5e} S/cm")
            return {"msd_m2": msd, "D_m2_s": D, "conductivity_S_cm": sigma_S_cm}

    # -------------------------------------------------
    # Execute simulation
    # -------------------------------------------------
    sim = KMCSimulator(structure, cutoff, base_Ea, alpha, beta,
                       epsilon_r, nu, T)
    sim.run_until_target()
    sim.calculate_properties()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
