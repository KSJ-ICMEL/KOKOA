"""
KOKOA Simulation #1
Generated: 2026-01-13 10:40:20
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

    # ----------------------------------------------------------------------
    # Configuration (can be tuned)
    # ----------------------------------------------------------------------
    cutoff = 4.0                     # Å, neighbor search radius
    E0 = 0.30                        # eV, base activation energy
    alpha = 0.05                     # eV/Å, distance penalty
    nu = 1e13                        # Hz, attempt frequency
    T = 300.0                        # K, temperature
    target_time = 5e-9               # s, fixed simulation length

    # ----------------------------------------------------------------------
    # Load structure
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)          # _CIF_PATH is pre‑injected
    # expand to a reasonable supercell to reduce finite‑size effects
    supercell = [2, 2, 2]
    structure.make_supercell(supercell)

    # ----------------------------------------------------------------------
    # Identify Li sites and build adjacency list with distances
    # ----------------------------------------------------------------------
    li_indices = [i for i, site in enumerate(structure) if "Li" in site.species_string]
    num_li = len(li_indices)

    # neighbor data for all sites (including non‑Li, we will filter later)
    all_neighbors = structure.get_all_neighbors(r=cutoff, include_index=True)

    # adjacency: {li_index: [(neighbor_li_index, distance), ...], ...}
    adjacency = {i: [] for i in li_indices}
    distances = []   # collect all Li‑Li distances to obtain r0

    for i in li_indices:
        for nb in all_neighbors[i]:
            j = nb.index
            if j not in li_indices:
                continue
            # avoid double counting (i<j)
            if j <= i:
                continue
            r = nb.distance
            adjacency[i].append((j, r))
            adjacency[j].append((i, r))
            distances.append(r)

    # nearest‑neighbor distance r0 (global minimum)
    r0 = min(distances) if distances else 0.0

    # ----------------------------------------------------------------------
    # Pre‑compute hop rates for each possible jump (will be updated if needed)
    # ----------------------------------------------------------------------
    def hop_rate(r):
        """Rate for a hop of length r (Å)."""
        Ea = E0 + alpha * (r - r0)          # eV
        return nu * np.exp(-Ea / (kB * T / e_charge))   # convert kB*T to eV

    # store rates in a dict keyed by (i, j) tuple (i<j)
    hop_rates = {}
    for i in li_indices:
        for j, r in adjacency[i]:
            if i < j:
                hop_rates[(i, j)] = hop_rate(r)

    # ----------------------------------------------------------------------
    # KMCSimulator class
    # ----------------------------------------------------------------------
    class KMCSimulator:
        def __init__(self, structure, li_indices, adjacency, hop_rates):
            self.structure = structure
            self.li_indices = li_indices
            self.adjacency = adjacency
            self.hop_rates = hop_rates

            # particle positions (cartesian) indexed by Li site index
            self.positions = {i: structure[i].coords.copy() for i in li_indices}

            # time bookkeeping
            self.time = 0.0

            # record trajectory for MSD (list of (time, positions_dict))
            self.trajectory = [(self.time, self._snapshot_positions())]

        def _snapshot_positions(self):
            """Return a copy of current positions as a dict {i: np.array}."""
            return {i: pos.copy() for i, pos in self.positions.items()}

        def run_step(self):
            """Perform a single kMC event."""
            # Build list of possible events and their rates
            events = []
            rates = []
            for i in self.li_indices:
                for j, _ in self.adjacency[i]:
                    # consider hop i -> j only once (i<j) and decide direction randomly later
                    if i < j:
                        rate = self.hop_rates[(i, j)]
                        events.append((i, j, rate))
                        rates.append(rate)

            if not events:
                return False  # no possible hops

            total_rate = np.sum(rates)
            # draw time increment
            rand = np.random.rand()
            dt = -np.log(rand) / total_rate
            self.time += dt

            # select which event occurs
            cum_rates = np.cumsum(rates)
            r = np.random.rand() * total_rate
            idx = np.searchsorted(cum_rates, r)
            i, j, _ = events[idx]

            # decide direction of hop (i->j or j->i) with equal probability
            if np.random.rand() < 0.5:
                src, dst = i, j
            else:
                src, dst = j, i

            # execute hop: move particle from src to dst
            self.positions[dst] = self.positions[src]  # occupy dst with src's particle
            # src becomes vacant (we keep its position but it is not used further)
            # In this simple model we do not track vacancies explicitly.

            # record snapshot
            self.trajectory.append((self.time, self._snapshot_positions()))
            return True

        def run(self, target_time):
            """Run kMC until target_time is reached."""
            while self.time < target_time:
                if not self.run_step():
                    break   # no more moves possible

        def calculate_properties(self):
            """Compute MSD, diffusion coefficient D, and ionic conductivity."""
            # Use final positions to compute MSD relative to initial
            t_final = self.time
            init_pos = self.trajectory[0][1]
            final_pos = self.trajectory[-1][1]

            displacements = []
            for i in self.li_indices:
                dr = final_pos[i] - init_pos[i]
                displacements.append(np.dot(dr, dr))
            msd = np.mean(displacements)   # Å^2

            # Convert Å^2 to m^2
            msd_m2 = msd * 1e-20

            # Diffusion coefficient D = MSD / (6 * t)
            D = msd_m2 / (6.0 * t_final)    # m^2/s

            # Number density of Li (per m^3)
            volume = self.structure.lattice.volume * 1e-30   # Å^3 -> m^3
            n_li = len(self.li_indices) / volume            # m^-3

            # Conductivity σ = n * q^2 * D / (kB * T)
            sigma = n_li * (e_charge**2) * D / (kB * T)      # S/m
            sigma_cgs = sigma * 1e2                         # S/cm

            print(f"Conductivity: {sigma_cgs:.3e} S/cm")
            return {"MSD (Å^2)": msd, "D (m^2/s)": D, "sigma (S/cm)": sigma_cgs}

    # ----------------------------------------------------------------------
    # Execute simulation
    # ----------------------------------------------------------------------
    sim = KMCSimulator(structure, li_indices, adjacency, hop_rates)
    sim.run(target_time)
    sim.calculate_properties()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
