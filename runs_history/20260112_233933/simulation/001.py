"""
KOKOA Simulation #1
Generated: 2026-01-12 23:40:12
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
            attempt_freq: float = 1e12,          # Hz (≈1 THz)
            activation_energy: float = 0.35,     # eV
            cutoff: float = 4.0,                 # Å, neighbor search radius
            target_time: float = 1e-8,           # s, fixed as required
        ):
            self.structure = structure
            self.mobile_species = mobile_species
            self.T = temperature
            self.nu0 = attempt_freq
            self.Ea = activation_energy * e          # convert eV → J
            self.cutoff = cutoff
            self.target_time = target_time

            # pre‑compute rate for a single hop (same for all hops)
            self.k_hop = self.nu0 * np.exp(-self.Ea / (k_B * self.T))

            # ------------------------------------------------------------------
            # Identify mobile sites (indices) and build neighbor list
            # ------------------------------------------------------------------
            self.mobile_sites = [
                i for i, site in enumerate(self.structure)
                if self.mobile_species in [el.symbol for el in site.species.elements]
            ]

            # neighbor list: dict {site_index: [neighbor_site_indices, ...]}
            self.neighbor_dict = {i: [] for i in self.mobile_sites}
            for i in self.mobile_sites:
                site = self.structure[i]
                neighbors = self.structure.get_neighbors(site, self.cutoff)
                for nb in neighbors:
                    j = nb.index
                    if j in self.mobile_sites and j != i:
                        self.neighbor_dict[i].append(j)

            # ------------------------------------------------------------------
            # Initialise particles: one particle per mobile site
            # ------------------------------------------------------------------
            self.particle_sites = np.array(self.mobile_sites, copy=True)  # current site index per particle
            self.initial_coords = np.array(
                [self.structure[i].coords for i in self.particle_sites]
            )  # Å

            # simulation clock
            self.time = 0.0

        # ----------------------------------------------------------------------
        # Perform a single kMC step (Gillespie algorithm)
        # ----------------------------------------------------------------------
        def run_step(self):
            # total number of possible hops from current configuration
            possible_hops = [
                (p_idx, nbr)
                for p_idx, site_idx in enumerate(self.particle_sites)
                for nbr in self.neighbor_dict[site_idx]
            ]

            if not possible_hops:
                # no hops possible → stop simulation
                self.time = self.target_time
                return

            N_events = len(possible_hops)
            R_total = N_events * self.k_hop

            # draw time increment
            rand = np.random.rand()
            dt = -np.log(rand) / R_total
            self.time += dt

            # select which event occurs
            event_idx = np.random.randint(N_events)
            particle_idx, new_site = possible_hops[event_idx]

            # update particle position
            self.particle_sites[particle_idx] = new_site

        # ----------------------------------------------------------------------
        # Run the simulation until the target time is reached
        # ----------------------------------------------------------------------
        def run(self):
            while self.time < self.target_time:
                self.run_step()

        # ----------------------------------------------------------------------
        # Compute MSD, diffusion coefficient and ionic conductivity
        # ----------------------------------------------------------------------
        def calculate_properties(self):
            # final Cartesian coordinates (Å)
            final_coords = np.array(
                [self.structure[i].coords for i in self.particle_sites]
            )

            # mean‑squared displacement (Å^2)
            displacements = final_coords - self.initial_coords
            msd_ang2 = np.mean(np.sum(displacements ** 2, axis=1))

            # convert Å^2 → cm^2 (1 Å = 1e-8 cm)
            msd_cm2 = msd_ang2 * 1e-16

            # diffusion coefficient D = <Δr^2> / (6·t)  (cm^2 s⁻¹)
            D = msd_cm2 / (6.0 * self.time)

            # mobile ion concentration c (cm⁻³)
            # volume of the cell in cm³ (1 Å³ = 1e-24 cm³)
            volume_cm3 = self.structure.lattice.volume * 1e-24
            n_ions = len(self.particle_sites)
            c = n_ions / volume_cm3

            # ionic conductivity σ = (e²·c·D) / (k_B·T)  (S m⁻¹)
            sigma_S_m = (e ** 2 * c * D) / (k_B * self.T)

            # convert to S cm⁻¹ (1 S m⁻¹ = 1e-2 S cm⁻¹)
            sigma_S_cm = sigma_S_m * 1e-2

            print(f"Conductivity: {sigma_S_cm:.5e} S/cm")
            return {"msd": msd_ang2, "D": D, "sigma": sigma_S_cm}

    # ----------------------------------------------------------------------
    # 3. Execute the simulation
    # ----------------------------------------------------------------------
    kmc = KMCSimulator(
        structure=structure,
        mobile_species="Li",
        temperature=300.0,
        attempt_freq=1e12,          # 1 THz
        activation_energy=0.35,     # eV
        cutoff=4.0,
        target_time=1e-8,           # fixed as required
    )

    kmc.run()
    kmc.calculate_properties()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
