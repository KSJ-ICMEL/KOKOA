"""
KOKOA Simulation #6
Generated: 2026-01-13 11:00:37
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260113_105531')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    import scipy.constants as const
    from pymatgen.core import Structure

    # -------------------------------------------------
    # Fixed configuration (as required by the task)
    # -------------------------------------------------
    target_time = 5e-9                     # seconds (fixed)
    supercell_dim = [3, 3, 3]              # 3×3×3 expansion (fixed)
    temperature = 300.0                   # K
    nu = 1e13                              # attempt frequency (s⁻¹)
    E0 = 0.18 * const.eV                    # baseline barrier (J)
    alpha = -0.05 * const.eV                # concentration‑dependent term (J)
    cutoff = 4.0                           # Å, neighbor search radius
    print_interval = 2000                 # steps
    k_B = const.Boltzmann                  # J/K
    q_li = const.elementary_charge         # C

    # -------------------------------------------------
    # Load structure and build supercell
    # -------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(supercell_dim)

    lattice = structure.lattice.matrix          # Å³
    volume_ang3 = structure.lattice.volume      # Å³
    volume_m3 = volume_ang3 * 1e-30              # m³

    # -------------------------------------------------
    # Identify Li sites and build neighbor list
    # -------------------------------------------------
    all_sites = list(structure.sites)
    num_sites = len(all_sites)

    # indices of Li atoms (mobile species)
    li_indices = [i for i, site in enumerate(all_sites) if site.species_string == "Li"]
    num_li = len(li_indices)

    # occupancy array: True if Li occupies the site, False otherwise
    occupancy = np.zeros(num_sites, dtype=bool)
    occupancy[li_indices] = True

    # neighbor list for every site (within cutoff)
    coords = np.array([site.coords for site in all_sites])
    neighbor_list = [[] for _ in range(num_sites)]
    for i in range(num_sites):
        dvec = coords - coords[i]
        # apply minimum image convention using lattice vectors
        frac = np.linalg.solve(lattice.T, dvec.T).T
        frac -= np.rint(frac)
        dvec = (frac @ lattice).astype(float)
        dists = np.linalg.norm(dvec, axis=1)
        neigh = np.where((dists > 1e-3) & (dists <= cutoff))[0]
        neighbor_list[i] = neigh.tolist()

    # -------------------------------------------------
    # KMCSimulator definition
    # -------------------------------------------------
    class KMCSimulator:
        def __init__(self):
            self.time = 0.0                     # s
            self.step = 0
            self.occupancy = occupancy.copy()
            self.coords = coords.copy()
            # store initial positions of each Li ion (by site index)
            self.li_initial_pos = self.coords[self.occupancy].copy()
            # map from Li index (0..num_li-1) to current site index
            self.li_site = np.where(self.occupancy)[0]

        def _local_concentration(self, site_idx):
            """fraction of neighboring sites that are occupied by Li"""
            neigh = neighbor_list[site_idx]
            if not neigh:
                return 0.0
            return np.sum(self.occupancy[neigh]) / len(neigh)

        def _global_concentration(self):
            return np.sum(self.occupancy) / num_sites

        def _build_event_list(self):
            """list of possible hops (from_site, to_site, rate)"""
            events = []
            c_ref = self._global_concentration()
            for from_idx in self.li_site:
                for to_idx in neighbor_list[from_idx]:
                    if not self.occupancy[to_idx]:          # vacancy
                        c_local = self._local_concentration(from_idx)
                        E_eff = E0 + alpha * (c_local - c_ref)
                        rate = nu * np.exp(-E_eff / (k_B * temperature))
                        events.append((from_idx, to_idx, rate))
            return events

        def run_step(self):
            events = self._build_event_list()
            if not events:
                # no possible moves; advance time arbitrarily
                self.time = target_time
                return

            rates = np.array([ev[2] for ev in events])
            total_rate = rates.sum()
            # time increment
            r = np.random.random()
            dt = -np.log(r) / total_rate
            self.time += dt
            self.step += 1

            # choose event
            r2 = np.random.random() * total_rate
            cum = np.cumsum(rates)
            idx = np.searchsorted(cum, r2)
            from_idx, to_idx, _ = events[idx]

            # execute hop
            self.occupancy[from_idx] = False
            self.occupancy[to_idx] = True

            # update Li site list
            li_idx = np.where(self.li_site == from_idx)[0][0]
            self.li_site[li_idx] = to_idx

            # update coordinates (periodic images are handled via lattice vectors)
            self.coords[to_idx] = self.coords[to_idx]  # positions stay in Cartesian frame

        def calculate_msd(self):
            """Mean‑squared displacement of Li ions from their initial positions"""
            current_pos = self.coords[self.occupancy]
            disp = current_pos - self.li_initial_pos
            # apply minimum image convention
            frac = np.linalg.solve(lattice.T, disp.T).T
            frac -= np.rint(frac)
            disp = (frac @ lattice)
            msd = np.mean(np.sum(disp**2, axis=1))
            return msd

        def calculate_properties(self):
            """Run until target_time and return final MSD and conductivity"""
            while self.time < target_time:
                self.run_step()
                if self.step % print_interval == 0:
                    msd = self.calculate_msd()
                    D = msd / (6.0 * self.time) if self.time > 0 else 0.0
                    n_li = np.sum(self.occupancy) / volume_m3          # ions / m³
                    sigma = (D * q_li**2 * n_li) / (k_B * temperature)  # S/m
                    sigma_mS_cm = sigma * 0.01 * 1e3                     # mS/cm
                    print(f"Step {self.step:6d} | Time {self.time*1e9:8.3f} ns | "
                          f"MSD {msd:10.4e} Å² | sigma {sigma_mS_cm:8.3f} mS/cm")
            # final values
            final_msd = self.calculate_msd()
            D = final_msd / (6.0 * self.time) if self.time > 0 else 0.0
            n_li = np.sum(self.occupancy) / volume_m3
            sigma = (D * q_li**2 * n_li) / (k_B * temperature)   # S/m
            sigma_S_cm = sigma * 0.01                            # S/cm
            return final_msd, sigma_S_cm

    # -------------------------------------------------
    # Run simulation
    # -------------------------------------------------
    sim = KMCSimulator()
    final_msd, final_sigma = sim.calculate_properties()
    print(f"Conductivity: {final_sigma:.3e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
