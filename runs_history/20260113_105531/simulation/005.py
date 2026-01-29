"""
KOKOA Simulation #5
Generated: 2026-01-13 11:00:14
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
    E0 = 0.18 * const.eV                    # reduced baseline barrier (J)
    alpha = -0.05 * const.eV               # concentration‑dependent term (J) – not used here
    cutoff = 4.0                           # Å, neighbor search radius
    print_interval = 2000                 # steps
    epsilon_r = 10.0                      # relative permittivity (unused)
    q_li = const.elementary_charge         # C
    k_B = const.Boltzmann                  # J/K

    # -------------------------------------------------
    # Load structure and build supercell
    # -------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(supercell_dim)

    lattice = structure.lattice.matrix          # Å³ matrix
    volume_ang3 = structure.lattice.volume      # Å³
    volume_m3 = volume_ang3 * 1e-30              # m³

    # -------------------------------------------------
    # Identify Li sites and build neighbor list
    # -------------------------------------------------
    all_sites = list(structure.sites)
    li_indices = [i for i, site in enumerate(all_sites) if site.species_string == "Li"]
    n_li = len(li_indices)

    # neighbor list: for each Li site, list of neighboring Li site indices within cutoff
    neighbor_list = {i: [] for i in li_indices}
    for i in li_indices:
        site = all_sites[i]
        neighbors = structure.get_neighbors(site, cutoff)
        for nb in neighbors:
            j = nb.index
            if all_sites[j].species_string == "Li" and j != i:
                neighbor_list[i].append(j)

    # -------------------------------------------------
    # KMCSimulator class
    # -------------------------------------------------
    class KMCSimulator:
        def __init__(self, structure, li_indices, neighbor_list):
            self.structure = structure
            self.lattice = structure.lattice.matrix
            self.inv_lattice = np.linalg.inv(self.lattice)
            self.li_indices = li_indices               # current site indices for each ion
            self.neighbor_list = neighbor_list
            self.n_ions = len(li_indices)

            # initial Cartesian positions (Å)
            self.init_pos = np.array([structure.sites[idx].coords for idx in li_indices])
            self.curr_pos = self.init_pos.copy()

            # map ion -> site index (updates as hops occur)
            self.ion_site = np.array(li_indices, dtype=int)

            self.time = 0.0
            self.step = 0

            # pre‑compute constant rate (same for all hops in this simple model)
            self.rate_per_hop = nu * np.exp(-(E0) / (k_B * temperature))

        def _minimum_image(self, diff):
            """Apply minimum image convention to Cartesian differences (Å)."""
            frac = diff @ self.inv_lattice
            frac -= np.rint(frac)          # wrap into [-0.5,0.5]
            return frac @ self.lattice

        def run(self):
            while self.time < target_time:
                # build list of possible hops (ion index, destination site index)
                possible_hops = []
                for ion_idx, site_idx in enumerate(self.ion_site):
                    for dest in self.neighbor_list[site_idx]:
                        possible_hops.append((ion_idx, dest))

                n_hops = len(possible_hops)
                if n_hops == 0:
                    break  # no moves possible

                total_rate = n_hops * self.rate_per_hop
                # choose time increment
                self.time += -np.log(np.random.rand()) / total_rate

                # choose which hop occurs (uniform weighting because rates equal)
                hop_idx = np.random.randint(n_hops)
                ion, new_site = possible_hops[hop_idx]

                # update ion position and site mapping
                self.ion_site[ion] = new_site
                self.curr_pos[ion] = self.structure.sites[new_site].coords

                self.step += 1

                if self.step % print_interval == 0:
                    self.report()

        def calculate_msd(self):
            diff = self.curr_pos - self.init_pos          # Å
            diff = self._minimum_image(diff)              # apply PBC
            msd = np.mean(np.sum(diff**2, axis=1))        # Å²
            return msd

        def calculate_properties(self):
            msd_ang2 = self.calculate_msd()
            msd_m2 = msd_ang2 * 1e-20                     # Å² → m²
            D = msd_m2 / (6.0 * self.time)                # m²/s

            # Li number density (ions per m³)
            n_density = self.n_ions / volume_m3           # m⁻³

            sigma = (q_li**2 * n_density * D) / (k_B * temperature)   # S/m
            sigma_mS_per_cm = sigma * 1000 / 100.0        # (S/m) → (mS/cm)
            return msd_ang2, sigma_mS_per_cm

        def report(self):
            msd, sigma = self.calculate_properties()
            time_ns = self.time * 1e9
            print(f"Step: {self.step}, Time: {time_ns:.3f} ns, MSD: {msd:.4f} Å², sigma: {sigma:.4f} mS/cm")

    # -------------------------------------------------
    # Run simulation
    # -------------------------------------------------
    sim = KMCSimulator(structure, li_indices, neighbor_list)
    sim.run()

    # Final conductivity output
    final_msd, final_sigma = sim.calculate_properties()
    print(f"Conductivity: {final_sigma:.6e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
