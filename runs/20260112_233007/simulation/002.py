"""
KOKOA Simulation #2
Generated: 2026-01-12 23:31:12
"""
import os, sys, traceback

# Pre-calculated paths
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_233007')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure

    # ----------------------------------------------------------------------
    # Constants
    # ----------------------------------------------------------------------
    kB_eV = 8.617333262e-5          # eV/K
    kB_J = 1.380649e-23            # J/K
    e_charge = 1.602176634e-19     # C
    T = 300.0                      # K
    nu = 1e13                      # 1/s, attempt frequency
    base_Ea = 0.30                 # eV, original barrier
    Ea_offset = 0.10               # eV, hypothesis increment
    target_time = 1e-8             # s, fixed simulation time
    cutoff = 4.0                   # Å, neighbor search radius
    dim = 3                        # dimensionality for MSD

    # ----------------------------------------------------------------------
    # Load structure
    # ----------------------------------------------------------------------
    structure = Structure.from_file(_CIF_PATH)
    # make a modest supercell to have enough sites
    structure.make_supercell([2, 2, 2])

    # ----------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------
    def get_li_sites(struct):
        """Return indices of sites containing Li."""
        li_idxs = []
        for i, site in enumerate(struct):
            if any(el.symbol == "Li" for el in site.species.elements):
                li_idxs.append(i)
        return li_idxs

    def build_neighbor_dict(struct, li_idxs, r):
        """For each Li site, list neighboring Li site indices within radius r."""
        neighbor_dict = {i: [] for i in li_idxs}
        all_neighbors = struct.get_all_neighbors(r=r, include_index=True)
        for i in li_idxs:
            for nb in all_neighbors[i]:
                j = nb[2]  # neighbor site index
                if j in li_idxs and j != i:
                    neighbor_dict[i].append(j)
        return neighbor_dict

    # ----------------------------------------------------------------------
    # KMCSimulator class
    # ----------------------------------------------------------------------
    class KMCSimulator:
        def __init__(self, struct):
            self.struct = struct
            self.li_sites = get_li_sites(struct)
            self.neighbor_dict = build_neighbor_dict(struct, self.li_sites, cutoff)

            # Occupancy: 1 if Li present, 0 otherwise (initially all Li sites occupied)
            self.occupancy = {i: 1 for i in self.li_sites}

            # Record initial Cartesian positions of each Li ion
            self.initial_positions = {}
            for i in self.li_sites:
                self.initial_positions[i] = struct[i].coords.copy()

            # Trajectory storage for MSD (list of dicts: site->position)
            self.positions_history = []

            # Pre‑compute rates for each possible hop (i -> j)
            self.rate_dict = {}
            Ea = base_Ea + Ea_offset  # apply hypothesis
            rate = nu * np.exp(-Ea / (kB_eV * T))
            for i, neighs in self.neighbor_dict.items():
                for j in neighs:
                    self.rate_dict[(i, j)] = rate

            self.time = 0.0

        def run_step(self):
            """Perform one kMC event using the Gillespie algorithm."""
            # Gather all feasible hops (occupied -> empty)
            feasible = [(i, j, self.rate_dict[(i, j)])
                        for i, neighs in self.neighbor_dict.items()
                        for j in neighs
                        if self.occupancy[i] == 1 and self.occupancy[j] == 0]

            if not feasible:
                # No possible moves; advance time to target directly
                self.time = target_time
                return

            rates = np.array([r for _, _, r in feasible])
            R_tot = rates.sum()
            # Time increment
            dt = -np.log(np.random.rand()) / R_tot
            self.time += dt

            # Choose event
            cum_rates = np.cumsum(rates)
            rnd = np.random.rand() * R_tot
            idx = np.searchsorted(cum_rates, rnd)
            i_sel, j_sel, _ = feasible[idx]

            # Execute hop: move Li from i_sel to j_sel
            self.occupancy[i_sel] = 0
            self.occupancy[j_sel] = 1

            # Record positions after this event
            current_pos = {}
            for i in self.li_sites:
                if self.occupancy[i] == 1:
                    current_pos[i] = self.struct[i].coords.copy()
            self.positions_history.append(current_pos)

        def run_until(self, t_target):
            """Run kMC until cumulative time reaches t_target."""
            while self.time < t_target:
                self.run_step()

        def calculate_msd(self):
            """Mean squared displacement of Li ions (Å^2)."""
            if not self.positions_history:
                return 0.0
            msd_sum = 0.0
            count = 0
            for snapshot in self.positions_history:
                for idx, pos in snapshot.items():
                    dr = pos - self.initial_positions[idx]
                    msd_sum += np.dot(dr, dr)
                    count += 1
            return msd_sum / count if count > 0 else 0.0

        def calculate_diffusion(self):
            """Diffusion coefficient D (m^2/s) from MSD."""
            msd = self.calculate_msd()          # Å^2
            D_ang2_s = msd / (2 * dim * self.time) if self.time > 0 else 0.0
            return D_ang2_s * (1e-10)**2        # convert Å^2 to m^2

        def calculate_conductivity(self):
            """Conductivity σ (S/cm) using Nernst‑Einstein relation."""
            D = self.calculate_diffusion()      # m^2/s
            # Number density of mobile Li (per m^3)
            vol_ang3 = self.struct.volume          # Å^3
            vol_m3 = vol_ang3 * (1e-10)**3
            n_li = len(self.li_sites) / vol_m3     # Li per m^3
            sigma_S_m = (n_li * e_charge**2 * D) / (kB_J * T)
            sigma_S_cm = sigma_S_m * 0.01          # 1 S/m = 0.01 S/cm
            return sigma_S_cm

    # ----------------------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------------------
    sim = KMCSimulator(structure)
    sim.run_until(target_time)
    conductivity = sim.calculate_conductivity()
    print(f"Conductivity: {conductivity} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
