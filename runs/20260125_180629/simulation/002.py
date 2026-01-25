"""
KOKOA Simulation #2
Generated: 2026-01-25 18:07:25
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
    '''kMC Simulation for Li-ion Conductivity in Solid Electrolyte with Environment‑dependent Kinetics'''
    import numpy as np
    from pymatgen.core import Structure
    import os

    # === 1. Structure Loading ===
    # Use absolute path based on this file's location
    cif_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLUBA.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)

    N = 4  # Supercell expansion
    structure.make_supercell([N, N, N])
    print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

    # Initialize Li sites with occupancy probability
    initial_sites = []
    for site in structure:
        if "Li" in [s.symbol for s in site.species.elements]:
            prob = site.species.get("Li", 0)
            state = 1 if np.random.rand() < prob else 0
            initial_sites.append({"coords": site.frac_coords, "state": state})

    print(f"Li sites initialized: {len(initial_sites)}")

    # === 2. Build Adjacency Graph ===
    cutoff = 4.0  # Angstrom
    neighbors_data = structure.get_all_neighbors(r=cutoff)
    adj_list = {}

    for i, site in enumerate(structure):
        if "Li" not in site.species.elements[0].symbol:
            continue
        neighbors = []
        for nb in neighbors_data[i]:
            if "Li" in structure[nb.index].species.elements[0].symbol:
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                neighbors.append((nb.index, cart_disp))
        adj_list[i] = neighbors

    print(f"Graph built (cutoff={cutoff}A)")

    # === 3. kMC Simulator (BKL Algorithm) with heterogeneous rates ===
    class KMCSimulator:
        def __init__(self, structure, adj_list, initial_sites, params):
            self.params = params
            self.adj_list = adj_list
            self.structure = structure
            self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        
            # particle bookkeeping
            self.site_to_particle = {}
            self.particle_positions = {}
            p_id = 0
            for idx, s in enumerate(initial_sites):
                if s['state'] == 1:
                    start = structure.lattice.get_cartesian_coords(s['coords'])
                    self.site_to_particle[idx] = p_id
                    self.particle_positions[p_id] = {'start': np.array(start), 'current': np.array(start)}
                    p_id += 1
        
            self.li_indices = set(self.site_to_particle.keys())
            self.num_particles = len(self.li_indices)
            self.current_time = 0.0
            self.step_count = 0
        
            # physical constants
            self.kb = 8.617e-5  # eV/K
        
            # reference hop distance (average Li‑Li distance in the adjacency list)
            dists = []
            for src, neighs in self.adj_list.items():
                for _, vec in neighs:
                    dists.append(np.linalg.norm(vec))
            self.d0 = np.mean(dists) if dists else 2.5  # Å, fallback
        
        # ---------------------------------------------------------------------
        def _activation_energy(self, distance, n_occ):
            """Linear model for Ea based on bottleneck size and local occupancy.
            Ea = Ea0 + α·(d – d0) + β·N_occ
            """
            Ea0 = self.params.get('Ea0', 0.30)      # eV
            alpha = self.params.get('alpha', 0.10)  # eV/Å
            beta = self.params.get('beta', 0.05)    # eV per occupied neighbour
            d0 = self.d0
            return Ea0 + alpha * (distance - d0) + beta * n_occ

        def _attempt_frequency(self, distance):
            """Modulate ν with bottleneck size: ν = ν0·exp[-γ·(d – d0)]"""
            nu0 = self.params.get('nu0', 1e13)      # Hz
            gamma = self.params.get('gamma', 0.05)  # 1/Å
            d0 = self.d0
            return nu0 * np.exp(-gamma * (distance - d0))

        def _local_occupied_neighbors(self, midpoint):
            """Count occupied Li sites within neighbor_radius of the hop midpoint."""
            radius = self.params.get('neighbor_radius', 3.0)  # Å
            count = 0
            # Cartesian coordinates of all Li sites
            for idx, occ in enumerate(self.occupancy):
                if occ == 1:
                    cart = self.structure.lattice.get_cartesian_coords(self.structure[idx].frac_coords)
                    if np.linalg.norm(cart - midpoint) <= radius:
                        count += 1
            return count

        def _rate_for_hop(self, src, tgt, vec):
            """Compute the individual hop rate k = ν·exp(-Ea/kT)."""
            distance = np.linalg.norm(vec)
            midpoint = self.structure.lattice.get_cartesian_coords(
                (self.structure[src].frac_coords + self.structure[tgt].frac_coords) / 2.0)
            n_occ = self._local_occupied_neighbors(midpoint)
            Ea = self._activation_energy(distance, n_occ)
            nu = self._attempt_frequency(distance)
            return nu * np.exp(-Ea / (self.kb * self.params['T']))

        def run_step(self):
            events = []
            rates_cumulative = []
            total_rate = 0.0
            # enumerate all possible hops
            for src in self.li_indices:
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[tgt] == 0:
                        rate = self._rate_for_hop(src, tgt, vec)
                        if rate <= 0:
                            continue
                        total_rate += rate
                        events.append((src, tgt, vec))
                        rates_cumulative.append(total_rate)
        
            if total_rate == 0.0:
                return False  # deadlock
        
            # BKL time advance
            self.current_time += -np.log(np.random.rand()) / total_rate
            self.step_count += 1
        
            # select event
            r = np.random.rand() * total_rate
            idx = np.searchsorted(rates_cumulative, r)
            src, tgt, vec = events[idx]
        
            # execute hop
            p_id = self.site_to_particle.pop(src)
            self.particle_positions[p_id]['current'] += vec
            self.occupancy[src], self.occupancy[tgt] = 0, 1
            self.site_to_particle[tgt] = p_id
            self.li_indices.discard(src)
            self.li_indices.add(tgt)
            return True

        def calculate_properties(self):
            if self.current_time == 0:
                return 0, 0
            msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])
            D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
            n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
            sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])
            return msd, sigma

    # === 4. Run Simulation ===
    # New kinetic parameters for heterogeneous barriers
    sim_params = {
        'T': 298,
        'Ea0': 0.30,          # baseline activation energy (eV)
        'nu0': 1e13,          # baseline attempt frequency (Hz)
        'alpha': 0.10,        # eV per Å deviation from reference distance
        'beta': 0.05,         # eV per occupied neighbour in the local sphere
        'gamma': 0.05,        # 1/Å modulation of attempt frequency
        'neighbor_radius': 3.0,  # Å for counting nearby Li
        'volume': structure.volume
    }

    sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

    target_time = 1000e-9  # 1000ns timeout
    log_interval = 100
    sigma_history = []

    while sim.current_time < target_time:
        if not sim.run_step():
            print("Deadlock - stopping")
            break
        if sim.step_count % log_interval == 0:
            msd, sigma = sim.calculate_properties()
            sigma_history.append(sigma)
            if len(sigma_history) > 1000:
                sigma_history.pop(0)
            if len(sigma_history) == 1000:
                avg_sigma = np.mean(sigma_history)
                std_sigma = np.std(sigma_history)
                rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0
                print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")
                if rsd < 0.05:
                    print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                    break
            else:
                print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

    # Final result
    msd, sigma = sim.calculate_properties()
    D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0

    print(f"\n=== Simulation Complete ===")
    print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
    print(f"D={D:.4e} cm^2/s")
    print(f"Conductivity: {sigma:.4e} S/cm")

    # Save result to JSON
    import json
    result = {
        "is_success": True,
        "conductivity": sigma,
        "diffusivity": D,
        "msd": msd,
        "simulation_time_ns": sim.current_time * 1e9,
        "temperature_K": sim_params['T'],
        "steps": sim.step_count,
        "error_message": None,
        "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns"
    }

    result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\n📁 결과 저장: {result_path}")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
