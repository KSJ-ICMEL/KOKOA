"""
KOKOA Simulation #3
Generated: 2026-01-13 11:25:27
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
    angstrom_to_cm = 1e-8               # cm/Å

    params = {
        "nu": 1e13,                     # attempt frequency [1/s]
        "Ea": 0.30,                     # base activation energy [eV]
        "alpha_Ea": 0.02,               # reduction per vacant neighbor [eV]
        "T": 300.0,                     # temperature [K]
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
    coords = structure.cart_coords[li_site_indices]   # Å
    neighbor_list = [[] for _ in range(n_li_sites)]
    cutoff_sq = params["cutoff"] ** 2

    for i in range(n_li_sites):
        for j in range(i + 1, n_li_sites):
            dvec = coords[i] - coords[j]
            dist_sq = np.dot(dvec, dvec)
            if dist_sq <= cutoff_sq:
                neighbor_list[i].append(j)
                neighbor_list[j].append(i)

    # ------------------------------
    # Initialise vacancies and ions
    # ------------------------------
    rng = np.random.default_rng()
    n_vacancies = int(params["vacancy_fraction"] * n_li_sites)
    occupied_flags = np.ones(n_li_sites, dtype=int)
    occupied_flags[:n_vacancies] = -1          # -1 marks vacancy
    rng.shuffle(occupied_flags)

    site_to_ion = np.full(n_li_sites, -1, dtype=int)   # -1 = vacancy
    ion_to_site = {}
    ion_initial_pos = {}
    ion_current_pos = {}

    ion_id = 0
    for site_idx, flag in enumerate(occupied_flags):
        if flag == -1:
            continue
        site_to_ion[site_idx] = ion_id
        ion_to_site[ion_id] = site_idx
        pos = coords[site_idx].copy()
        ion_initial_pos[ion_id] = pos
        ion_current_pos[ion_id] = pos
        ion_id += 1

    n_ions = ion_id

    # ------------------------------
    # KMC Simulator
    # ------------------------------
    class KMCSimulator:
        def __init__(self, structure, li_indices, neighbor_list, params,
                     site_to_ion, ion_to_site, ion_initial_pos, ion_current_pos):
            self.structure = structure
            self.li_indices = li_indices
            self.neighbor_list = neighbor_list
            self.params = params
            self.site_to_ion = site_to_ion
            self.ion_to_site = ion_to_site
            self.ion_initial_pos = ion_initial_pos
            self.ion_current_pos = ion_current_pos
            self.time = 0.0
            self.step = 0
            self.volume_cm3 = structure.lattice.volume * (angstrom_to_cm ** 3)  # Å³ → cm³

        def _vacant_neighbors(self, site_idx):
            return sum(1 for nb in self.neighbor_list[site_idx]
                       if self.site_to_ion[nb] == -1)

        def run(self):
            target = self.params["target_time"]
            print_int = self.params["print_interval"]
            while self.time < target:
                hops = []          # (ion_id, src, dst, disp_vector)
                rates = []         # corresponding rates

                # enumerate possible hops
                for ion_id, src in self.ion_to_site.items():
                    for dst in self.neighbor_list[src]:
                        if self.site_to_ion[dst] != -1:
                            continue  # not vacant
                        disp = (self.structure.cart_coords[self.li_indices[dst]] -
                                self.structure.cart_coords[self.li_indices[src]])
                        # environment‑dependent barrier
                        n_vac = self._vacant_neighbors(src)
                        Ea_eff = self.params["Ea"] - self.params["alpha_Ea"] * n_vac
                        rate = (self.params["nu"] *
                                np.exp(-(Ea_eff) / (kb_eV * self.params["T"])))
                        hops.append((ion_id, src, dst, disp))
                        rates.append(rate)

                if not hops:
                    print("No possible hops remaining.")
                    break

                rates = np.array(rates)
                total_rate = rates.sum()
                # Gillespie time increment
                dt = -np.log(rng.random()) / total_rate
                self.time += dt
                self.step += 1

                # Choose hop
                cum = np.cumsum(rates)
                r = rng.random() * total_rate
                hop_idx = np.searchsorted(cum, r)
                ion_id, src, dst, disp = hops[hop_idx]

                # Execute hop
                self.site_to_ion[src] = -1
                self.site_to_ion[dst] = ion_id
                self.ion_to_site[ion_id] = dst
                new_pos = self.structure.cart_coords[self.li_indices[dst]].copy()
                self.ion_current_pos[ion_id] = new_pos

                # Progress report
                if self.step % print_int == 0:
                    msd = self._mean_squared_displacement()
                    D = msd / (6.0 * self.time) if self.time > 0 else 0.0
                    sigma = (self._carrier_density() * e_charge**2 * D) / (kb_J * self.params["T"])
                    sigma_S_per_cm = sigma * 0.01          # S/m → S/cm
                    sigma_mS_per_cm = sigma_S_per_cm * 1e3
                    print(f"Step {self.step}, Time {self.time*1e9:.3f} ns, "
                          f"MSD {msd:.3e} Å², sigma {sigma_mS_per_cm:.3f} mS/cm")

            # Final results
            final_sigma = self.calculate_conductivity()
            print(f"Conductivity: {final_sigma:.6e} S/cm")

        def _mean_squared_displacement(self):
            displacements = []
            for ion_id in self.ion_initial_pos:
                d = self.ion_current_pos[ion_id] - self.ion_initial_pos[ion_id]
                displacements.append(np.dot(d, d))
            return np.mean(displacements) if displacements else 0.0

        def _carrier_density(self):
            # number of mobile Li ions per cm³
            return len(self.ion_to_site) / self.volume_cm3

        def calculate_conductivity(self):
            msd = self._mean_squared_displacement()
            D = msd / (6.0 * self.time) if self.time > 0 else 0.0
            sigma = (self._carrier_density() * e_charge**2 * D) / (kb_J * self.params["T"])
            return sigma * 0.01   # S/m → S/cm

    # ------------------------------
    # Run simulation
    # ------------------------------
    sim = KMCSimulator(structure, li_site_indices, neighbor_list, params,
                       site_to_ion, ion_to_site, ion_initial_pos, ion_current_pos)
    sim.run()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
