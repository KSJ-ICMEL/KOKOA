"""
KOKOA Simulation #2
Generated: 2026-01-13 10:40:41
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260113_103940')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from scipy.constants import k as kB, e as e_charge, N_A
    from pymatgen.core import Structure

    # ------------------- Configuration -------------------
    cutoff = 4.0                     # Å, neighbor search radius
    E0 = 0.30                        # eV, base activation energy
    alpha = 0.05                     # eV/Å, distance penalty
    nu0 = 1e13                       # Hz, base attempt frequency
    beta = 0.20                      # Å⁻¹, distance decay for attempt freq
    T = 300.0                        # K, temperature
    target_time = 5e-9               # s, fixed simulation length
    # -----------------------------------------------------

    # ------------------- Load Structure -----------------
    structure = Structure.from_file(_CIF_PATH)   # _CIF_PATH is pre‑injected
    structure.make_supercell([2, 2, 2])          # enlarge to reduce finite size
    vol = structure.lattice.volume * 1e-24       # cm³ (Å³ → cm³)
    # -----------------------------------------------------

    # ------------------- Identify Li Sites ---------------
    li_sites = [i for i, site in enumerate(structure) if "Li" in site.species_string]
    num_ions = len(li_sites)
    # -----------------------------------------------------

    # ------------------- Build Adjacency -----------------
    adjacency = {}
    for idx in li_sites:
        neighbors = structure.get_neighbors(structure[idx], r=cutoff, include_index=True)
        # keep only Li neighbors (including same site -> ignore)
        neigh_list = [(nbr[2], nbr[1]) for nbr in neighbors if "Li" in nbr[0].species_string and nbr[2] != idx]
        adjacency[idx] = neigh_list
    # -----------------------------------------------------

    class KMCSimulator:
        def __init__(self, li_sites, adjacency):
            self.li_sites = np.array(li_sites)          # immutable list of site ids
            self.adjacency = adjacency
            # initial positions: each ion starts at its own site
            self.current_sites = np.copy(self.li_sites)
            self.initial_coords = np.array([structure[i].coords for i in self.current_sites])
            self.time = 0.0
            self.displacements = np.zeros((len(self.current_sites), 3))  # cumulative displacement

        def _hop_rate(self, distance):
            """distance‑dependent rate (Hz)"""
            return nu0 * np.exp(-beta * distance) * np.exp(-(E0 + alpha * distance) / (kB * T))

        def run_step(self):
            """Perform a single kMC event."""
            rates = []
            moves = []  # (ion_idx, target_site, distance, vector)
            for ion_idx, site in enumerate(self.current_sites):
                for neigh_site, dist in self.adjacency[site]:
                    rate = self._hop_rate(dist)
                    if rate > 0:
                        rates.append(rate)
                        # vector from current to neighbor (consider periodic images)
                        vec = structure[neigh_site].coords - structure[site].coords
                        # wrap into [-0.5,0.5) fractional then to cartesian
                        frac = structure.lattice.get_fractional_coords(vec)
                        frac = frac - np.rint(frac)
                        vec_cart = structure.lattice.get_cartesian_coords(frac)
                        moves.append((ion_idx, neigh_site, dist, vec_cart))
            if not rates:
                return False  # no possible moves
            rates = np.array(rates)
            total_rate = rates.sum()
            # time increment
            r = np.random.random()
            dt = -np.log(r) / total_rate
            self.time += dt
            # select event
            cum = np.cumsum(rates)
            r2 = np.random.random() * total_rate
            ev_idx = np.searchsorted(cum, r2)
            ion_idx, target_site, _, vec = moves[ev_idx]
            # update ion position and displacement
            self.displacements[ion_idx] += vec
            self.current_sites[ion_idx] = target_site
            return True

        def run_until(self, t_target):
            """Run kMC until cumulative time reaches t_target."""
            while self.time < t_target:
                if not self.run_step():
                    break

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient, and ionic conductivity."""
            # mean‑square displacement (Å²)
            msd = np.mean(np.sum(self.displacements**2, axis=1))
            # diffusion coefficient D (cm²/s)
            D = msd * 1e-16 / (6.0 * self.time)   # Å²→cm²
            # number density of mobile ions (cm⁻³)
            n = num_ions / vol
            # conductivity (S/cm) via Nernst‑Einstein
            sigma = n * (e_charge**2) * D / (kB * T)
            return msd, D, sigma

    # ------------------- Run Simulation -----------------
    sim = KMCSimulator(li_sites, adjacency)
    sim.run_until(target_time)
    msd, D, sigma = sim.calculate_properties()
    print(f"Conductivity: {sigma:.3e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
