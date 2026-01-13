"""
KOKOA Simulation #2
Generated: 2026-01-13 11:24:33
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260113_112215')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure

    # ------------------------------
    # Constants and parameters
    # ------------------------------
    e_charge = 1.602176634e-19          # C
    kb_eV = 8.617333262e-5              # eV/K
    kb_J = 1.380649e-23                 # J/K
    epsilon0 = 8.854187817e-12         # F/m

    # Simulation parameters (can be tuned)
    params = {
        "nu": 1e13,                     # attempt frequency [1/s]
        "Ea": 0.30,                     # activation energy [eV]
        "T": 300.0,                     # temperature [K]
        "epsilon_r": 10.0,              # relative dielectric constant
        "cutoff": 3.0,                  # neighbor cutoff [Å]
        "vacancy_fraction": 0.05,       # fraction of Li vacancies
        "target_time": 5e-9,            # target simulation time [s]
        "print_interval": 2000         # steps between progress prints
    }

    # ------------------------------
    # Load structure and build supercell
    # ------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell([3, 3, 3])          # fixed 3×3×3 supercell

    # ------------------------------
    # Identify Li sites
    # ------------------------------
    li_site_indices = [i for i, site in enumerate(structure)
                       if any(el.symbol == "Li" for el in site.species)]

    n_li_sites = len(li_site_indices)

    # ------------------------------
    # Build neighbor list for Li sites
    # ------------------------------
    cart_coords = structure.lattice.get_cartesian_coords(structure.frac_coords)
    li_cart = cart_coords[li_site_indices]

    neighbor_list = [[] for _ in range(n_li_sites)]
    for i in range(n_li_sites):
        for j in range(i + 1, n_li_sites):
            disp = li_cart[j] - li_cart[i]
            # minimum image convention
            disp -= structure.lattice.matrix @ np.rint(structure.lattice.inv_matrix @ disp)
            dist = np.linalg.norm(disp)
            if dist <= params["cutoff"]:
                neighbor_list[i].append(j)
                neighbor_list[j].append(i)

    # ------------------------------
    # Initialise occupancy (vacancies)
    # ------------------------------
    rng = np.random.default_rng()
    vacancies = rng.random(n_li_sites) < params["vacancy_fraction"]
    occupied = ~vacancies

    site_to_ion = np.full(n_li_sites, -1, dtype=int)   # -1 = vacant
    ion_to_site = {}
    ion_counter = 0
    for idx, occ in enumerate(occupied):
        if occ:
            site_to_ion[idx] = ion_counter
            ion_to_site[ion_counter] = idx
            ion_counter += 1

    n_ions = ion_counter

    # Store initial Cartesian positions of each ion (Å)
    initial_positions = np.zeros((n_ions, 3))
    for ion_id, site_idx in ion_to_site.items():
        frac = structure.frac_coords[li_site_indices[site_idx]]
        initial_positions[ion_id] = structure.lattice.get_cartesian_coords(frac)

    # ------------------------------
    # KMC Simulator
    # ------------------------------
    class KMCSimulator:
        def __init__(self, structure, li_indices, neighbor_list, site_to_ion,
                     ion_to_site, initial_positions, params):
            self.structure = structure
            self.li_indices = li_indices
            self.neighbor_list = neighbor_list
            self.site_to_ion = site_to_ion
            self.ion_to_site = ion_to_site
            self.positions = initial_positions.copy()   # current positions (Å)
            self.initial_positions = initial_positions.copy()
            self.params = params

            self.time = 0.0          # seconds
            self.step = 0
            self.msd_history = []

            # Pre‑compute volume in cm³ for conductivity
            self.volume_cm3 = structure.lattice.volume * 1e-24   # Å³ → cm³

        def _coulomb_penalty(self, r_ang):
            """Return ΔE in eV for a hop of length r (Å)."""
            r_m = r_ang * 1e-10
            delta_E_J = (e_charge ** 2) / (4 * np.pi * epsilon0 * self.params["epsilon_r"] * r_m)
            return delta_E_J / e_charge   # convert J → eV

        def _possible_hops(self):
            """Generate list of (ion_id, src_site, dst_site, disp_vector)."""
            hops = []
            for ion_id, src_site in self.ion_to_site.items():
                for dst_site in self.neighbor_list[src_site]:
                    if self.site_to_ion[dst_site] == -1:   # vacancy
                        # displacement with PBC
                        src_frac = self.structure.frac_coords[self.li_indices[src_site]]
                        dst_frac = self.structure.frac_coords[self.li_indices[dst_site]]
                        disp_frac = dst_frac - src_frac
                        disp_frac -= np.rint(disp_frac)   # wrap
                        disp_cart = self.structure.lattice.get_cartesian_coords(disp_frac)
                        hops.append((ion_id, src_site, dst_site, disp_cart))
            return hops

        def run(self):
            while self.time < self.params["target_time"]:
                hops = self._possible_hops()
                if not hops:
                    break   # no moves possible

                rates = []
                for ion_id, src, dst, disp in hops:
                    r = np.linalg.norm(disp)                     # Å
                    delta_E = self._coulomb_penalty(r)           # eV
                    rate = (self.params["nu"] *
                            np.exp(-(self.params["Ea"] + delta_E) /
                                   (kb_eV * self.params["T"])))
                    rates.append(rate)

                rates = np.array(rates)
                total_rate = rates.sum()
                if total_rate == 0:
                    break

                # Choose hop
                cum_rates = np.cumsum(rates)
                rrand = rng.random() * total_rate
                hop_idx = np.searchsorted(cum_rates, rrand)
                ion_id, src, dst, disp = hops[hop_idx]

                # Execute hop
                self.site_to_ion[src] = -1
                self.site_to_ion[dst] = ion_id
                self.ion_to_site[ion_id] = dst
                self.positions[ion_id] += disp   # update Cartesian position

                # Advance time (Gillespie)
                dt = -np.log(rng.random()) / total_rate
                self.time += dt
                self.step += 1

                # Record MSD
                if self.step % self.params["print_interval"] == 0:
                    self._report_progress()

        def _report_progress(self):
            elapsed_ns = self.time * 1e9
            displacements = self.positions - self.initial_positions
            msd = np.mean(np.sum(displacements ** 2, axis=1))   # Å²
            self.msd_history.append((self.time, msd))

            # Diffusion coefficient D = MSD / (6 t)
            D = msd / (6 * self.time)          # Å²/s
            D_m2 = D * 1e-20                    # convert Å² → m²

            # Number density (ions per m³)
            n_density = self.positions.shape[0] / (self.structure.lattice.volume * 1e-30)

            # Conductivity σ = n q² D / (kB T)  (S/m)
            sigma_S_m = n_density * e_charge ** 2 * D_m2 / (kb_J * self.params["T"])
            sigma_mS_cm = sigma_S_m * 1e3 / 100.0   # mS/cm

            print(f"Step {self.step:6d} | Time {elapsed_ns:8.3f} ns | "
                  f"MSD {msd:10.4f} Å² | σ {sigma_mS_cm:8.4f} mS/cm")

        def calculate_properties(self):
            # Use final MSD and total time
            total_time = self.time
            if total_time == 0:
                return 0.0, 0.0
            final_msd = np.mean(np.sum((self.positions - self.initial_positions) ** 2, axis=1))
            D = final_msd / (6 * total_time)          # Å²/s
            D_m2 = D * 1e-20                           # m²/s
            n_density = self.positions.shape[0] / (self.structure.lattice.volume * 1e-30)
            sigma_S_m = n_density * e_charge ** 2 * D_m2 / (kb_J * self.params["T"])
            sigma_S_cm = sigma_S_m / 100.0
            return D, sigma_S_cm

    # ------------------------------
    # Run simulation
    # ------------------------------
    sim = KMCSimulator(structure, li_site_indices, neighbor_list,
                       site_to_ion, ion_to_site, initial_positions, params)
    sim.run()
    D, sigma = sim.calculate_properties()
    print(f"Conductivity: {sigma} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
