"""
KOKOA Simulation #3
Generated: 2026-01-12 23:41:01
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

            # fractional coordinates of all sites
            self.frac_coords = structure.frac_coords.copy()          # (N,3)
            # Cartesian lattice vectors (3x3 matrix)
            self.lattice_vectors = structure.lattice.matrix          # (3,3)

            # build neighbour list with minimum‑image convention
            self._build_neighbourhood()

            # indices of sites occupied by mobile ions
            self.mobile_sites = np.where(structure.species_string == mobile_species)[0]
            self.n_ions = len(self.mobile_sites)

            # initialise each ion on its own site
            self.ion_site = self.mobile_sites.copy()                 # current site index per ion
            # unwrapped Cartesian positions (track real displacement across PBC)
            self.init_cart = self._frac_to_cart(self.frac_coords[self.ion_site])
            self.unwrapped_cart = self.init_cart.copy()

            # simulation clock
            self.time = 0.0

        def _frac_to_cart(self, frac):
            """Convert fractional to Cartesian coordinates."""
            return frac @ self.lattice_vectors

        def _build_neighbourhood(self):
            """Neighbour list using minimum‑image convention and a distance cutoff."""
            f = self.frac_coords
            # pairwise fractional differences
            delta = f[:, None, :] - f[None, :, :]          # (N,N,3)
            delta -= np.rint(delta)                       # wrap into [-0.5,0.5]
            # Cartesian differences
            cart_diff = np.tensordot(delta, self.lattice_vectors, axes=([2], [0]))  # (N,N,3)
            dists = np.linalg.norm(cart_diff, axis=-1)    # (N,N)

            neigh = []
            for i in range(len(f)):
                mask = (dists[i] > 0.0) & (dists[i] <= self.cutoff)
                neigh.append(np.where(mask)[0])
            self.neighbourhood = neigh                     # list of arrays

        def _select_event(self):
            """Select an ion and a target neighbour using Gillespie rates."""
            # For each ion, collect possible hops and their rates
            rates = []
            choices = []   # (ion_index, target_site)
            beta = 1.0 / (k_B * self.T)

            for ion_idx, cur_site in enumerate(self.ion_site):
                neigh_sites = self.neighbourhood[cur_site]
                if len(neigh_sites) == 0:
                    continue
                # identical barrier for all hops in this simple model
                hop_rate = self.nu0 * np.exp(-self.Ea * beta)
                rates.extend([hop_rate] * len(neigh_sites))
                choices.extend([(ion_idx, tgt) for tgt in neigh_sites])

            if not rates:
                return None, None, np.inf

            rates = np.array(rates)
            total_rate = rates.sum()
            # draw random number to pick which event occurs
            r = np.random.rand() * total_rate
            cum = np.cumsum(rates)
            idx = np.searchsorted(cum, r)
            ion_idx, tgt_site = choices[idx]
            return ion_idx, tgt_site, total_rate

        def run(self):
            """Run the kMC until the target simulation time is reached."""
            while self.time < self.target_time:
                ion_idx, tgt_site, R_tot = self._select_event()
                if R_tot == np.inf:   # no possible moves
                    break
                # time increment
                dt = -np.log(np.random.rand()) / R_tot
                self.time += dt
                if self.time > self.target_time:
                    break

                # perform the hop
                cur_site = self.ion_site[ion_idx]
                # fractional displacement with minimum image
                d_frac = self.frac_coords[tgt_site] - self.frac_coords[cur_site]
                d_frac -= np.rint(d_frac)          # wrap into [-0.5,0.5]
                # Cartesian displacement
                d_cart = d_frac @ self.lattice_vectors
                # update unwrapped position
                self.unwrapped_cart[ion_idx] += d_cart
                # update site index
                self.ion_site[ion_idx] = tgt_site

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient and ionic conductivity."""
            # Mean‑squared displacement (average over ions)
            displacements = self.unwrapped_cart - self.init_cart   # (n_ions,3)
            msd = np.mean(np.sum(displacements**2, axis=1))       # Å^2
            # Convert Å^2 to m^2
            msd_m2 = msd * (1e-10)**2

            # Diffusion coefficient D = MSD / (6 * t)
            D = msd_m2 / (6.0 * self.time)   # m^2/s

            # Number density of mobile ions (ions per m^3)
            volume = self.structure.lattice.volume * (1e-10)**3   # Å^3 → m^3
            n = self.n_ions / volume

            # Conductivity via Nernst‑Einstein: σ = n·q²·D / (k_B·T)
            sigma = n * e**2 * D / (k_B * self.T)   # S/m
            sigma_cgs = sigma * 1e2                 # S/cm

            print(f"Conductivity: {sigma_cgs:.6e} S/cm")
            return {"MSD (Å^2)": msd, "D (m^2/s)": D, "sigma (S/cm)": sigma_cgs}

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
