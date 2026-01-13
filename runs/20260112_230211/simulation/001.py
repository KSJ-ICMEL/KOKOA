"""
KOKOA Simulation #1
Generated: 2026-01-12 23:02:46
"""
import os, sys, traceback

# Add project root to path for kokoa imports
# runs/xxx/simulation/xxx.py -> 3 levels up = project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_230211')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure
    import os
    import sys
    import json

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kokoa.config import Config

    # Load structure
    cif_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Li4.47La3Zr2O12.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)
    N = 4
    structure.make_supercell([N, N, N])
    print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

    # Initialize Li sites
    initial_sites = []
    for site in structure:
        if "Li" in site.species.elements:
            prob = site.species.get("Li", 0)
            state = 1 if np.random.rand() < prob else 0
            initial_sites.append({"coords": site.frac_coords, "state": state})
    print(f"Li sites initialized: {len(initial_sites)}")

    # Build adjacency graph
    cutoff = 4.0
    neighbors_data = structure.get_all_neighbors(r=cutoff)
    adj_list = {}
    for i, site in enumerate(structure):
        if "Li" not in site.species.elements:
            continue
        neighbors = []
        for nb in neighbors_data[i]:
            if "Li" in structure[nb.index].species.elements:
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
                neighbors.append((nb.index, cart_disp))
        adj_list[i] = neighbors
    print(f"Graph built (cutoff={cutoff}A)")

    # Coulomb constants
    q = 1.602e-19  # C
    eps0 = 8.854e-12  # F/m
    epsr = 10.0
    coulomb_const = (q**2) / (4*np.pi*eps0*epsr) / 1.602e-19  # eV*m

    class KMCSimulator:
        def __init__(self, structure, adj_list, initial_sites, params):
            self.params = params
            self.adj_list = adj_list
            self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
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
            self.kb = 8.617e-5  # eV/K
            self.nu = params['nu']
            self.Ea = params['E_a']
            self.T = params['T']

        def run_step(self):
            events = []
            rates_cum = []
            total = 0.0
            for src in self.li_indices:
                p_id = self.site_to_particle[src]
                src_pos = self.particle_positions[p_id]['current']
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[tgt] == 0:
                        tgt_pos = src_pos + vec
                        # Coulomb energy difference
                        sum_old = 0.0
                        sum_new = 0.0
                        for other_id, pos in self.particle_positions.items():
                            if other_id == p_id:
                                continue
                            d_old = np.linalg.norm(src_pos - pos['current'])
                            d_new = np.linalg.norm(tgt_pos - pos['current'])
                            sum_old += 1.0/(d_old*1e-10)
                            sum_new += 1.0/(d_new*1e-10)
                        dE = coulomb_const * (sum_new - sum_old)
                        if dE < 0:
                            dE = 0.0
                        rate = self.nu * np.exp(-(self.Ea + dE)/(self.kb*self.T))
                        total += rate
                        events.append((src, tgt, vec, p_id))
                        rates_cum.append(total)
            if total == 0:
                return False
            self.current_time += -np.log(np.random.rand()) / total
            self.step_count += 1
            idx = np.searchsorted(rates_cum, np.random.uniform(0, total))
            src, tgt, vec, p_id = events[idx]
            # Update positions and mappings
            self.particle_positions[p_id]['current'] += vec
            self.occupancy[src], self.occupancy[tgt] = 0, 1
            self.site_to_particle.pop(src)
            self.site_to_particle[tgt] = p_id
            self.li_indices.discard(src)
            self.li_indices.add(tgt)
            return True

        def calculate_properties(self):
            if self.current_time == 0:
                return 0.0, 0.0
            msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])
            D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
            n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
            sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.T)  # S/cm
            return msd, sigma

    # Simulation parameters
    sim_params = {'T': 300, 'E_a': 0.28, 'nu': 1e13, 'volume': structure.volume}
    sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

    target_time = Config.SIMULATION_TIME
    log_interval = 2000

    while sim.current_time < target_time:
        if not sim.run_step():
            print("Deadlock - stopping")
            break
        if sim.step_count % log_interval == 0:
            msd, sigma = sim.calculate_properties()
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

    msd, sigma = sim.calculate_properties()
    D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0

    print(f"\n=== Simulation Complete ===")
    print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
    print(f"D={D:.4e} cm^2/s")
    print(f"Conductivity: {sigma:.4e} S/cm")

    # Save result
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
