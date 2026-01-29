"""
KOKOA Simulation #4
Generated: 2026-01-13 10:59:52
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
    epsilon_r = 10.0                      # relative permittivity
    q_li = const.elementary_charge         # C
    k_B = const.Boltzmann                  # J/K
    pi = np.pi

    # -------------------------------------------------
    # Load structure and build supercell
    # -------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(supercell_dim)

    lattice = structure.lattice.matrix      # 3×3 Å matrix
    volume = structure.lattice.volume       # Å³

    # -------------------------------------------------
    # Identify Li sites
    # -------------------------------------------------
    all_coords = np.array([site.coords for site in structure])
    species = np.array([list(site.species)[0].symbol for site in structure])
    li_mask = species == "Li"
    li_coords = all_coords[li_mask]
    num_li_sites = li_coords.shape[0]

    # -------------------------------------------------
    # Neighbor list for Li sites (within cutoff)
    # -------------------------------------------------
    def build_neighbor_list(coords, cutoff):
        n = coords.shape[0]
        neigh_idx = [[] for _ in range(n)]
        neigh_vec = [[] for _ in range(n)]
        neigh_dist = [[] for _ in range(n)]
        for i in range(n):
            diff = coords - coords[i]                     # Å
            # Minimum image convention
            frac = np.linalg.solve(lattice.T, diff.T).T   # fractional
            frac -= np.rint(frac)                         # wrap to [-0.5,0.5]
            cart = (lattice @ frac.T).T                    # back to Å
            dists = np.linalg.norm(cart, axis=1)
            within = (dists > 1e-5) & (dists <= cutoff)
            neigh_idx[i] = np.where(within)[0].tolist()
            neigh_vec[i] = cart[within].tolist()
            neigh_dist[i] = dists[within].tolist()
        return neigh_idx, neigh_vec, neigh_dist

    nbr_idx, nbr_vec, nbr_dist = build_neighbor_list(li_coords, cutoff)

    # -------------------------------------------------
    # KMCSimulator class
    # -------------------------------------------------
    class KMCSimulator:
        def __init__(self, coords, nbr_idx, nbr_vec, nbr_dist):
            self.coords = coords                      # Å, shape (N,3)
            self.nbr_idx = nbr_idx
            self.nbr_vec = nbr_vec
            self.nbr_dist = nbr_dist
            self.N = coords.shape[0]

            # Occupancy: random 90% filled
            rng = np.random.default_rng()
            self.occupied = np.zeros(self.N, dtype=bool)
            occ_sites = rng.choice(self.N, size=int(0.9 * self.N), replace=False)
            self.occupied[occ_sites] = True

            # Assign each ion a unique ID (based on initial site)
            self.ion_id = -np.ones(self.N, dtype=int)
            self.next_id = 0
            for i in np.where(self.occupied)[0]:
                self.ion_id[i] = self.next_id
                self.next_id += 1

            # Unwrapped displacement vectors for each ion (Å)
            self.displacements = np.zeros((self.next_id, 3), dtype=float)

            # Simulation counters
            self.time = 0.0
            self.step = 0

        def local_concentration(self, site):
            """Fraction of occupied neighbours of a given site."""
            neigh = self.nbr_idx[site]
            if len(neigh) == 0:
                return 0.0
            return np.sum(self.occupied[neigh]) / len(neigh)

        def coulomb_sum(self, i, j):
            """Coulomb contribution for hop i→j."""
            # Union of neighbours of i and j (excluding i and j themselves)
            neigh_set = set(self.nbr_idx[i] + self.nbr_idx[j])
            neigh_set.discard(i)
            neigh_set.discard(j)
            total = 0.0
            for k in neigh_set:
                if self.occupied[k]:
                    # distance from i to k (use pre‑computed distances)
                    # find index of k in i's neighbour list
                    try:
                        idx = self.nbr_idx[i].index(k)
                        r = self.nbr_dist[i][idx]
                    except ValueError:
                        # fallback to j's list
                        idx = self.nbr_idx[j].index(k)
                        r = self.nbr_dist[j][idx]
                    total += q_li**2 / (4 * pi * const.epsilon_0 * epsilon_r * (r * 1e-10))
            return total  # Joules

        def compute_rates(self):
            """Build list of possible hops and their rates."""
            hops = []          # (i, j, rate, delta_vec)
            total_rate = 0.0
            for i in np.where(self.occupied)[0]:
                for n_idx, vec in zip(self.nbr_idx[i], self.nbr_vec[i]):
                    if not self.occupied[n_idx]:          # vacancy
                        # activation energy
                        c_loc = self.local_concentration(i)
                        E = E0 + alpha * c_loc + self.coulomb_sum(i, n_idx)
                        rate = nu * np.exp(-E / (k_B * temperature))
                        hops.append((i, n_idx, rate, vec))
                        total_rate += rate
            return hops, total_rate

        def run_step(self):
            hops, total_rate = self.compute_rates()
            if total_rate == 0.0:
                raise RuntimeError("No possible hops; simulation stalled.")
            # Choose hop
            r = np.random.random() * total_rate
            cum = 0.0
            for i, j, rate, delta in hops:
                cum += rate
                if r <= cum:
                    chosen = (i, j, delta)
                    break
            i, j, delta = chosen

            # Advance time
            dt = -np.log(np.random.random()) / total_rate
            self.time += dt
            self.step += 1

            # Move ion
            ion = self.ion_id[i]
            self.occupied[i] = False
            self.occupied[j] = True
            self.ion_id[i] = -1
            self.ion_id[j] = ion

            # Update displacement (handle periodic wrap)
            # delta is cartesian vector already respecting minimum image
            self.displacements[ion] += delta

        def calculate_properties(self):
            """Return MSD (Å²), diffusion coefficient D (m²/s), conductivity σ (S/m)."""
            if self.displacements.shape[0] == 0:
                return 0.0, 0.0, 0.0
            msd = np.mean(np.sum(self.displacements**2, axis=1))   # Å²
            if self.time == 0.0:
                D = 0.0
            else:
                D = msd / (6.0 * self.time)                        # Å²/s
            # Convert D to m²/s
            D_m2 = D * (1e-10)**2
            n_ions = np.sum(self.occupied)                        # number of carriers
            n_density = n_ions / (volume * 1e-30)                  # m⁻³ (Å³ → m³)
            sigma = (D_m2 * q_li**2 * n_density) / (k_B * temperature)  # S/m
            return msd, D, sigma

        def run(self):
            while self.time < target_time:
                self.run_step()
                if self.step % print_interval == 0:
                    msd, D, sigma = self.calculate_properties()
                    print(f"Step {self.step}, Time {self.time*1e9:.3f} ns, "
                          f"MSD {msd:.3f} Å², sigma {sigma*100:.3f} mS/cm")
            # Final output
            _, _, sigma = self.calculate_properties()
            print(f"Conductivity: {sigma:.6e} S/cm")

    # -------------------------------------------------
    # Execute simulation
    # -------------------------------------------------
    sim = KMCSimulator(li_coords, nbr_idx, nbr_vec, nbr_dist)
    sim.run()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
