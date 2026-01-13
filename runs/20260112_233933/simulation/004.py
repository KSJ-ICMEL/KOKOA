"""
KOKOA Simulation #4
Generated: 2026-01-12 23:41:21
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
    # 2. kMC simulator
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
            target_time: float = 1e-8,           # s (fixed)
        ):
            self.structure = structure
            self.mobile_species = mobile_species
            self.T = temperature
            self.nu0 = attempt_freq
            self.Ea = activation_energy * e          # J
            self.cutoff = cutoff
            self.target_time = target_time

            # ------------------------------------------------------------------
            # Identify sites initially occupied by the mobile species
            # ------------------------------------------------------------------
            self.mobile_sites = [
                i for i, site in enumerate(structure.sites)
                if site.species_string == mobile_species
            ]
            if not self.mobile_sites:
                raise ValueError(f"No sites with species '{mobile_species}' found in the structure.")

            # ------------------------------------------------------------------
            # Pre‑compute neighbour lists for every site (periodic images included)
            # ------------------------------------------------------------------
            self.neighbor_dict = {}
            for i, site in enumerate(structure.sites):
                neigh = structure.get_neighbors(site, self.cutoff)
                self.neighbor_dict[i] = [n.index for n in neigh]

            # ------------------------------------------------------------------
            # Initialise ion positions (site indices) and trajectory storage
            # ------------------------------------------------------------------
            self.ion_sites = np.array(self.mobile_sites, dtype=int)   # current site of each ion
            self.initial_frac = structure.frac_coords[self.ion_sites].copy()
            self.times = [0.0]
            self.frac_trajectories = [self.initial_frac.copy()]

            # ------------------------------------------------------------------
            # Constants for rate calculations
            # ------------------------------------------------------------------
            self.rate_per_hop = self.nu0 * np.exp(-self.Ea / (k_B * self.T))

            # Simulation state
            self.current_time = 0.0

        def run_step(self):
            """Perform a single kMC event using the Gillespie algorithm."""
            # Build list of all possible hops (ion index, destination site)
            possible_hops = []
            for ion_idx, cur_site in enumerate(self.ion_sites):
                for dest in self.neighbor_dict[cur_site]:
                    possible_hops.append((ion_idx, dest))

            if not possible_hops:
                # No hops possible – terminate simulation
                self.current_time = self.target_time
                return

            total_rate = len(possible_hops) * self.rate_per_hop
            # Time increment
            r = np.random.random()
            dt = -np.log(r) / total_rate
            self.current_time += dt

            # Choose a hop uniformly (all rates equal)
            hop_idx = np.random.randint(len(possible_hops))
            ion_idx, dest_site = possible_hops[hop_idx]

            # Execute hop
            self.ion_sites[ion_idx] = dest_site

            # Record trajectory
            self.times.append(self.current_time)
            self.frac_trajectories.append(self.structure.frac_coords[self.ion_sites].copy())

        def run(self):
            """Run the kMC until the target simulation time is reached."""
            while self.current_time < self.target_time:
                self.run_step()

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient D and ionic conductivity σ."""
            # Convert stored fractional coordinates to numpy array (steps, ions, 3)
            traj = np.array(self.frac_trajectories)          # shape (Nsteps, Nions, 3)
            # Unwrap fractional coordinates to account for periodic boundaries
            # using cumulative sum of jumps > 0.5
            diff = np.diff(traj, axis=0)
            jump = np.where(diff > 0.5, -1, np.where(diff < -0.5, 1, 0))
            unwrapped = np.cumsum(jump, axis=0) + traj[0][np.newaxis, :, :]
            # Displacement from initial positions
            disp_frac = unwrapped - unwrapped[0][np.newaxis, :, :]   # shape (Nsteps, Nions, 3)
            # Convert to Cartesian
            disp_cart = np.tensordot(disp_frac, self.structure.lattice.matrix, axes=([2], [0]))
            # Mean‑squared displacement (average over ions and time)
            msd_time = np.mean(np.sum(disp_cart**2, axis=2), axis=1)   # shape (Nsteps,)

            # Use the final MSD value at the end of the simulation
            msd_final = msd_time[-1]
            total_time = self.times[-1]

            # Diffusion coefficient (3D)
            D = msd_final / (6.0 * total_time)   # m^2/s

            # Number density of mobile ions (ions per m^3)
            volume = self.structure.lattice.volume * 1e-30   # Å^3 → m^3
            n_ions = len(self.mobile_sites) / volume

            # Ionic conductivity (S/m) → convert to S/cm
            sigma = n_ions * e**2 * D / (k_B * self.T)   # S/m
            sigma_cm = sigma * 1e-2                       # S/cm

            print(f"Conductivity: {sigma_cm:.3e} S/cm")
            return {
                "msd": msd_final,
                "D": D,
                "sigma_S_per_cm": sigma_cm,
                "total_time": total_time,
            }

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
