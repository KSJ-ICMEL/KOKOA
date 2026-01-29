"""
KOKOA Simulation #2
Generated: 2026-01-13 10:56:42
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
    target_time = 5e-9                     # seconds
    supercell_dim = [3, 3, 3]              # 3×3×3 expansion
    temperature = 300.0                   # K
    nu = 1e13                              # attempt frequency (s⁻¹)
    E0 = 0.2 * const.eV                    # baseline barrier (J)
    cutoff = 4.0                           # Å, neighbor search radius
    print_interval = 2000                 # steps
    kappa_ang = 1.0                        # Å⁻¹ (inverse screening length)
    q_li = const.elementary_charge         # C

    # -------------------------------------------------
    # Load structure and build supercell
    # -------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    structure.make_supercell(supercell_dim)

    # -------------------------------------------------
    # Identify Li sites (mobile ions)
    # -------------------------------------------------
    li_site_indices = [i for i, site in enumerate(structure)
                       if any(el.symbol == "Li" for el in site.species)]

    num_sites = len(li_site_indices)
    if num_sites == 0:
        raise RuntimeError("No Li sites found in the structure.")

    # -------------------------------------------------
    # Initial occupancy: all Li sites are occupied
    # -------------------------------------------------
    occupied = np.ones(num_sites, dtype=bool)

    # -------------------------------------------------
    # Pre‑compute neighbor pairs within cutoff
    # -------------------------------------------------
    neighbor_pairs = []   # (i, j, distance, vector)
    lattice = structure.lattice
    inv_lattice = np.linalg.inv(lattice.matrix)

    for idx_i, site_i in enumerate(structure.sites):
        if idx_i not in li_site_indices:
            continue
        i = li_site_indices.index(idx_i)
        neighbors = structure.get_neighbors(site_i, cutoff)
        for nb in neighbors:
            idx_j = nb.index
            if idx_j not in li_site_indices:
                continue
            j = li_site_indices.index(idx_j)
            if i == j:
                continue
            # store each unordered pair once
            if i < j:
                vec = nb.frac_coords - site_i.frac_coords
                vec -= np.rint(vec)               # minimum image in fractional
                cart_vec = vec @ lattice.matrix
                neighbor_pairs.append((i, j, nb.distance, cart_vec))

    # -------------------------------------------------
    # Helper functions
    # -------------------------------------------------
    def screened_coulomb(r):
        """Debye‑Hückel screened potential (J) for distance r in Å."""
        if r == 0.0:
            return 0.0
        r_m = r * 1e-10                     # Å → m
        prefactor = q_li**2 / (4 * const.pi * const.epsilon_0 * r_m)
        return prefactor * np.exp(-kappa_ang * r)   # kappa in Å⁻¹

    def delta_E_coulomb(i, j, occupied):
        """Coulomb contribution for a hop i→j."""
        dE = 0.0
        for k, occ in enumerate(occupied):
            if not occ or k == i or k == j:
                continue
            # distance i‑k
            _, _, _, vec_ik = next(p for p in neighbor_pairs if (p[0]==i and p[1]==k) or (p[0]==k and p[1]==i))
            r_ik = np.linalg.norm(vec_ik)
            # distance j‑k
            _, _, _, vec_jk = next(p for p in neighbor_pairs if (p[0]==j and p[1]==k) or (p[0]==k and p[1]==j))
            r_jk = np.linalg.norm(vec_jk)
            dE += screened_coulomb(r_jk) - screened_coulomb(r_ik)
        return dE

    def minimum_image(delta):
        """Apply minimum image convention to a Cartesian displacement."""
        frac = delta @ inv_lattice
        frac -= np.rint(frac)
        return frac @ lattice.matrix

    # -------------------------------------------------
    # KMCSimulator class
    # -------------------------------------------------
    class KMCSimulator:
        def __init__(self, structure, li_indices, occupied, neighbor_pairs):
            self.structure = structure
            self.li_indices = li_indices
            self.occupied = occupied.copy()
            self.neighbor_pairs = neighbor_pairs
            self.num_ions = np.count_nonzero(occupied)
            # store initial Cartesian positions of each ion
            self.init_pos = np.zeros((self.num_ions, 3))
            self.current_pos = np.zeros_like(self.init_pos)
            ion_counter = 0
            for site_idx, occ in zip(li_indices, occupied):
                if occ:
                    cart = structure.sites[site_idx].coords
                    self.init_pos[ion_counter] = cart
                    self.current_pos[ion_counter] = cart
                    ion_counter += 1
            self.time = 0.0
            self.step = 0

        def _available_hops(self):
            """Generate list of possible hops with their rates."""
            hops = []
            for i, j, dist, vec in self.neighbor_pairs:
                if self.occupied[i] and not self.occupied[j]:
                    dE_coul = delta_E_coulomb(i, j, self.occupied)
                    rate = nu * np.exp(-(E0 + dE_coul) / (const.k * temperature))
                    hops.append((i, j, rate, vec))
                elif self.occupied[j] and not self.occupied[i]:
                    # reverse direction also possible
                    dE_coul = delta_E_coulomb(j, i, self.occupied)
                    rate = nu * np.exp(-(E0 + dE_coul) / (const.k * temperature))
                    hops.append((j, i, rate, -vec))
            return hops

        def run_step(self):
            hops = self._available_hops()
            if not hops:
                raise RuntimeError("No possible hops; simulation stuck.")
            rates = np.array([h[2] for h in hops])
            total_rate = rates.sum()
            # Gillespie time increment
            r = np.random.random()
            dt = -np.log(r) / total_rate
            self.time += dt
            # Choose hop
            cum = np.cumsum(rates)
            pick = np.random.random() * total_rate
            idx = np.searchsorted(cum, pick)
            i, j, _, vec = hops[idx]
            # Update occupancy
            self.occupied[i] = False
            self.occupied[j] = True
            # Update ion positions (track which ion moved)
            # Find which ion corresponds to site i
            ion_idx = None
            count = 0
            for site_idx, occ in zip(self.li_indices, self.occupied):
                if occ and site_idx == i:
                    # this site is now occupied after move, so the ion came from j
                    continue
            # Simpler: recompute positions from occupancy after move
            ion_counter = 0
            for site_idx, occ in zip(self.li_indices, self.occupied):
                if occ:
                    self.current_pos[ion_counter] = self.structure.sites[site_idx].coords
                    ion_counter += 1
            self.step += 1

        def calculate_properties(self):
            """Return MSD (Å²), diffusion coefficient (cm²/s), conductivity (S/cm)."""
            # MSD using minimum image
            deltas = self.current_pos - self.init_pos
            for k in range(self.num_ions):
                deltas[k] = minimum_image(deltas[k])
            msd = np.mean(np.sum(deltas**2, axis=1))   # Å²
            D = msd / (6 * self.time)                 # Å²/s
            D_cm2 = D * 1e-16                          # Å² → cm²
            volume_cm3 = self.structure.lattice.volume * 1e-24  # Å³ → cm³
            n = self.num_ions / volume_cm3            # ions per cm³
            sigma = n * q_li**2 * D_cm2 / (const.k * temperature)  # S/cm
            return msd, D_cm2, sigma

    # -------------------------------------------------
    # Run the simulation
    # -------------------------------------------------
    sim = KMCSimulator(structure, li_site_indices, occupied, neighbor_pairs)

    while sim.time < target_time:
        sim.run_step()
        if sim.step % print_interval == 0:
            msd, D_cm2, sigma = sim.calculate_properties()
            print(f"Step {sim.step:6d} | Time {sim.time*1e9:8.3f} ns | "
                  f"MSD {msd:8.3f} Å² | sigma {sigma*1e3:8.3f} mS/cm")

    # Final properties
    msd, D_cm2, sigma = sim.calculate_properties()
    print(f"Conductivity: {sigma:.6e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
