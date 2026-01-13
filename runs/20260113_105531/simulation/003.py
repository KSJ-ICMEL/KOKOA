"""
KOKOA Simulation #3
Generated: 2026-01-13 10:57:23
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
    E0 = 0.2 * const.eV                    # baseline barrier (J)
    alpha = -0.05 * const.eV               # concentration‑dependent term (J)
    cutoff = 4.0                           # Å, neighbor search radius
    print_interval = 2000                 # steps
    q_li = const.elementary_charge         # C
    k_B = const.Boltzmann                  # J/K

    # -------------------------------------------------
    # Load structure and build supercell
    # -------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(supercell_dim)

    # -------------------------------------------------
    # Identify mobile Li sites and build neighbor list
    # -------------------------------------------------
    all_sites = np.array([site.coords for site in structure])
    site_species = np.array([list(site.species)[0].symbol for site in structure])

    li_indices = np.where(site_species == "Li")[0]
    num_sites = len(li_indices)
    if num_sites == 0:
        raise RuntimeError("No Li sites found in the structure.")

    # neighbor list (indices within cutoff)
    from scipy.spatial import cKDTree
    tree = cKDTree(all_sites)
    neighbors = tree.query_ball_point(all_sites, r=cutoff)

    # -------------------------------------------------
    # KMCSimulator definition
    # -------------------------------------------------
    class KMCSimulator:
        def __init__(self, structure, li_indices, neighbors):
            self.structure = structure
            self.li_indices = li_indices
            self.neighbors = neighbors
            self.num_sites = len(structure)
            self.occupancy = np.zeros(self.num_sites, dtype=bool)
            # initially occupy all Li sites
            self.occupancy[li_indices] = True

            # store positions of Li ions (for MSD)
            self.li_positions = all_sites[li_indices].copy()
            self.initial_positions = self.li_positions.copy()

            self.time = 0.0
            self.step = 0
            self.msd_history = []

            # volume in cm³
            self.volume_cm3 = structure.lattice.volume * 1e-24  # Å³ → cm³

        def possible_hops(self):
            """Return list of (origin, destination) tuples for all allowed hops."""
            hops = []
            occ = self.occupancy
            for i in self.li_indices:
                if not occ[i]:
                    continue
                for j in self.neighbors[i]:
                    if occ[j]:
                        continue  # destination already occupied
                    hops.append((i, j))
            return hops

        def hop_rate(self, dest):
            """Compute barrier‑modified rate for a hop to destination site."""
            neigh = self.neighbors[dest]
            # fractional local Li concentration around destination
            occupied = sum(self.occupancy[n] for n in neigh)
            c_loc = occupied / len(neigh) if neigh else 0.0
            E_barrier = E0 + alpha * c_loc          # J
            return nu * np.exp(-E_barrier / (k_B * temperature))

        def run_step(self):
            hops = self.possible_hops()
            if not hops:
                raise RuntimeError("No possible hops; simulation stuck.")

            rates = np.array([self.hop_rate(dest) for (_, dest) in hops])
            total_rate = rates.sum()
            # select hop
            r = np.random.rand() * total_rate
            cum = np.cumsum(rates)
            idx = np.searchsorted(cum, r)
            origin, dest = hops[idx]

            # perform hop
            self.occupancy[origin] = False
            self.occupancy[dest] = True

            # update Li positions (track the ion that moved)
            moving_li_idx = np.where(self.li_indices == origin)[0][0]
            self.li_positions[moving_li_idx] = self.structure[dest].coords

            # advance time
            dt = -np.log(np.random.rand()) / total_rate
            self.time += dt
            self.step += 1

        def calculate_msd(self):
            disp = self.li_positions - self.initial_positions
            # apply minimum image convention for periodic boundaries
            lattice = self.structure.lattice.matrix
            inv_lat = np.linalg.inv(lattice)
            frac_disp = disp @ inv_lat
            frac_disp -= np.rint(frac_disp)  # wrap into [-0.5,0.5)
            cart_disp = frac_disp @ lattice
            msd = np.mean(np.sum(cart_disp**2, axis=1))
            return msd

        def calculate_conductivity(self):
            """Return conductivity in S/cm."""
            t = self.time if self.time > 0 else 1e-30
            msd = self.calculate_msd()
            D = msd / (6 * t)                     # cm²/s (since msd in Å², convert)
            D *= 1e-16                             # Å² → cm²
            n = self.occupancy.sum() / self.volume_cm3  # carriers per cm³
            sigma = (q_li**2 * n * D) / (k_B * temperature)  # S/cm
            return sigma

        def run_until_target(self):
            while self.time < target_time:
                self.run_step()
                if self.step % print_interval == 0:
                    msd = self.calculate_msd()
                    sigma = self.calculate_conductivity()
                    print(f"Step: {self.step}, Time: {self.time*1e9:.3f} ns, "
                          f"MSD: {msd:.3e} Å², sigma: {sigma*1e3:.3f} mS/cm")
            # final report
            final_sigma = self.calculate_conductivity()
            print(f"Conductivity: {final_sigma:.6e} S/cm")

    # -------------------------------------------------
    # Execute simulation
    # -------------------------------------------------
    sim = KMCSimulator(structure, li_indices, neighbors)
    sim.run_until_target()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
