"""
KOKOA Simulation #4
Generated: 2026-01-25 18:08:36
"""
import os, sys, traceback

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260125_180629')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    '''kMC Simulation for Li-ion Conductivity in Solid Electrolyte with Site‑Energy Differentiation'''
    import numpy as np
    import os
    from pymatgen.core import Structure

    # === 1. Structure Loading ===
    # Use absolute path based on this file's location, with fallback options
    cif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLUBA.cif")
    if not os.path.exists(cif_path):
        # Fallback to parent directory if not found in the script's directory
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLUBA.cif")
        if os.path.exists(alt_path):
            cif_path = alt_path
        else:
            raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)

    N = 4  # Supercell expansion
    structure.make_supercell([N, N, N])
    print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

    # === 2. Build Adjacency Graph ===
    cutoff = 4.0  # Angstrom
    neighbors_data = structure.get_all_neighbors(r=cutoff)
    adj_list = {}

    for i, site in enumerate(structure):
        # consider only Li sites
        if "Li" not in [el.symbol for el in site.species.elements]:
            continue
        neighbors = []
        for nb in neighbors_data[i]:
            if "Li" in [el.symbol for el in structure[nb.index].species.elements]:
                # fractional displacement including periodic image
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                neighbors.append((nb.index, cart_disp))
        adj_list[i] = neighbors

    print(f"Graph built (cutoff={cutoff}A)")

    # === 3. kMC Simulator (BKL Algorithm) with Site‑Energy Differentiation ===
    class KMCSimulator:
        def __init__(self, structure, adj_list, params):
            self.structure = structure
            self.adj_list = adj_list
            self.params = params
            self.kb = 8.617e-5  # eV/K
            self.T = params.get('temperature', 300.0)  # K
            self.nu = params.get('attempt_frequency', 1e13)  # Hz
            self.Ea = params.get('base_barrier', 0.30)  # eV, base migration barrier
            self.delta_E_site = params.get('delta_E_site', 0.15)  # eV, energy offset for high‑energy sites

            # ------------------------------------------------------------
            # 1) Assign site energies (0 eV for tetrahedral‑like, +ΔE for octahedral‑like)
            # ------------------------------------------------------------
            self.site_energies = {}
            for idx in self.adj_list.keys():
                # simple heuristic: tetrahedral sites have 4 Li neighbours in the adjacency list
                coord_num = len(self.adj_list[idx])
                if coord_num <= 4:
                    self.site_energies[idx] = 0.0
                else:
                    self.site_energies[idx] = self.delta_E_site

            # ------------------------------------------------------------
            # 2) Initialise Li occupancy using Boltzmann probabilities
            # ------------------------------------------------------------
            # Determine total number of Li ions from the stoichiometry supplied in params
            total_li_sites = len(self.adj_list)
            target_li_fraction = params.get('li_fraction', 0.5)  # fraction of sites that should be occupied
            target_num_li = int(round(total_li_sites * target_li_fraction))

            # Boltzmann weight for each site
            boltz_weights = np.array([np.exp(-E / (self.kb * self.T)) for E in self.site_energies.values()])
            prob_dist = boltz_weights / boltz_weights.sum()
            site_indices = list(self.site_energies.keys())
            chosen = np.random.choice(site_indices, size=target_num_li, replace=False, p=prob_dist)
            self.occupancy = np.zeros(total_li_sites, dtype=int)
            for idx in chosen:
                self.occupancy[idx] = 1

            # bookkeeping for particles (each Li ion gets a unique id)
            self.site_to_particle = {}
            self.particle_positions = {}
            p_id = 0
            for idx, occ in enumerate(self.occupancy):
                if occ == 1:
                    cart = self.structure.lattice.get_cartesian_coords(self.structure[idx].frac_coords)
                    self.site_to_particle[idx] = p_id
                    self.particle_positions[p_id] = {'start': cart.copy(), 'current': cart.copy()}
                    p_id += 1
            self.num_particles = p_id
            self.current_time = 0.0
            self.step_count = 0

            # reference hop distance (average Li‑Li distance in the adjacency list)
            dists = []
            for src, neighs in self.adj_list.items():
                for _, vec in neighs:
                    dists.append(np.linalg.norm(vec))
            self.d0 = np.mean(dists) if dists else 2.5  # Å, fallback

        # ------------------------------------------------------------
        # Helper: compute rate for a specific hop i -> j
        # ------------------------------------------------------------
        def hop_rate(self, i, j):
            # base barrier + site‑energy difference (final - initial)
            delta_E = self.site_energies[j] - self.site_energies[i]
            barrier = self.Ea + delta_E
            return self.nu * np.exp(-barrier / (self.kb * self.T))

        # ------------------------------------------------------------
        # Perform a single BKL step
        # ------------------------------------------------------------
        def run_step(self):
            # Build list of possible events and their rates
            events = []  # each entry: (i, j, rate)
            for i, occ in enumerate(self.occupancy):
                if occ == 0:
                    continue
                for j, _vec in self.adj_list[i]:
                    if self.occupancy[j] == 0:  # vacancy
                        rate = self.hop_rate(i, j)
                        if rate > 0:
                            events.append((i, j, rate))
            if not events:
                # No possible moves – simulation stalls
                return False
            rates = np.array([ev[2] for ev in events])
            total_rate = rates.sum()
            # BKL time increment
            rand = np.random.rand()
            dt = -np.log(rand) / total_rate
            self.current_time += dt
            # Choose event
            cum_rates = np.cumsum(rates)
            r = np.random.rand() * total_rate
            idx = np.searchsorted(cum_rates, r)
            i, j, _ = events[idx]
            # Execute hop i -> j
            particle_id = self.site_to_particle.pop(i)
            self.site_to_particle[j] = particle_id
            self.occupancy[i] = 0
            self.occupancy[j] = 1
            # Update particle position
            disp = self.structure.lattice.get_cartesian_coords(self.structure[j].frac_coords) - \
                   self.structure.lattice.get_cartesian_coords(self.structure[i].frac_coords)
            self.particle_positions[particle_id]['current'] += disp
            self.step_count += 1
            return True

        # ------------------------------------------------------------
        # Run the simulation until a target physical time is reached
        # ------------------------------------------------------------
        def run(self, target_time):
            while self.current_time < target_time:
                progressed = self.run_step()
                if not progressed:
                    print("No further hops possible; terminating early.")
                    break
            # After simulation, compute conductivity via Nernst‑Einstein
            return self.compute_conductivity()

        # ------------------------------------------------------------
        # Conductivity calculation
        # ------------------------------------------------------------
        def compute_conductivity(self):
            # Mean‑squared displacement (MSD) of all particles
            msd = 0.0
            for pid, pos in self.particle_positions.items():
                delta = pos['current'] - pos['start']
                msd += np.dot(delta, delta)
            msd /= self.num_particles
            # Diffusion coefficient D = MSD / (6 * t)
            D = msd / (6.0 * self.current_time)  # Å^2 / ps -> need unit conversion
            # Convert Å^2/ps to cm^2/s (1 Å = 1e-8 cm, 1 ps = 1e-12 s)
            D_cm2_s = D * (1e-8)**2 / (1e-12)
            # Number density of Li (per cm^3)
            volume_cm3 = self.structure.lattice.volume * (1e-24)  # Å^3 -> cm^3
            n_li = self.num_particles / volume_cm3
            q = 1.602e-19  # C
            sigma = n_li * q**2 * D_cm2_s / (self.kb * 1.380649e-23 * self.T)  # using kB in J/K
            return sigma

    # === 4. Simulation Parameters ===
    sim_params = {
        'temperature': 300.0,          # K
        'attempt_frequency': 1e13,    # Hz
        'base_barrier': 0.30,          # eV
        'delta_E_site': 0.15,          # eV (energy offset for high‑energy sites)
        'li_fraction': 0.5,            # Approximate Li occupancy fraction
    }

    # Target physical time for conductivity evaluation (seconds)
    # Convert to picoseconds because our dt is in ps (ν in Hz -> s⁻¹, but we keep dt in ps for convenience)
    # 1 s = 1e12 ps
    target_time_ps = 1e-6 * 1e12  # 1 µs -> 1e6 ps (adjust as needed)

    sim = KMCSimulator(structure, adj_list, sim_params)
    conductivity = sim.run(target_time_ps)
    print(f"Conductivity: {conductivity:.3e} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
