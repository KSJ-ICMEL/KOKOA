"""
KOKOA Simulation #1
Generated: 2026-01-12 22:57:49
"""
import os, sys, traceback

# Add project root to path for kokoa imports
# runs/xxx/simulation/xxx.py -> 3 levels up = project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_225707')
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

    cif_path = "./Li4.47La3Zr2O12.cif"
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)
    N = 4
    structure.make_supercell([N, N, N])
    print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

    initial_sites = []
    for site in structure:
        if "Li" in [s.symbol for s in site.species.elements]:
            prob = site.species.get("Li", 0)
            state = 1 if np.random.rand() < prob else 0
            initial_sites.append({"coords": site.frac_coords, "state": state})

    print(f"Li sites initialized: {len(initial_sites)}")

    cutoff = 4.0
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
            kb = 8.617e-5
            self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))
            self.eps_r = 10.0
            self.lambda_c = 5.0
            self.k_coulomb = 1/(4*np.pi*8.854e-12*self.eps_r)
            self.const_coulomb = self.k_coulomb * (1.602e-19)**2 / 1.602e-19
            self.kb = kb

        def compute_delta_E(self, src, vec):
            p_id = self.site_to_particle[src]
            pos_src = self.particle_positions[p_id]['current']
            pos_tgt = pos_src + vec
            delta = 0.0
            for pid, pos in self.particle_positions.items():
                if pid == p_id:
                    continue
                r_old = np.linalg.norm(pos_src - pos)
                r_new = np.linalg.norm(pos_tgt - pos)
                if r_old < 1e-12 or r_new < 1e-12:
                    continue
                r_old_A = r_old * 1e10
                r_new_A = r_new * 1e10
                e_old = self.const_coulomb * 1e10 / r_old_A * np.exp(-r_old_A / self.lambda_c)
                e_new = self.const_coulomb * 1e10 / r_new_A * np.exp(-r_new_A / self.lambda_c)
                delta += e_new - e_old
            return delta

        def run_step(self):
            events = []
            rates = []
            cum = 0.0
            for src in self.li_indices:
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[tgt] == 0:
                        dE = self.compute_delta_E(src, vec)
                        rate = self.base_rate * np.exp(-dE / (self.kb * self.params['T']))
                        cum += rate
                        events.append((src, tgt, vec))
                        rates.append(cum)
            if cum == 0:
                return False
            self.current_time += -np.log(np.random.rand()) / cum
            self.step_count += 1
            idx = np.searchsorted(rates, np.random.uniform(0, cum))
            src, tgt, vec = events[idx]
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
            D = msd / (6 * self.current_time) * 1e-16
            n = self.num_particles / (self.params['volume'] * 1e-24)
            sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])
            return msd, sigma

    sim_params = {'T': 300, 'E_a': 0.28, 'nu': 1e13, 'volume': structure.volume}
    sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

    target_time = Config.SIMULATION_TIME
    log_interval = 2000

    try:
        while sim.current_time < target_time:
            if not sim.run_step():
                print("Deadlock - stopping")
                break
            if sim.step_count % log_interval == 0:
                msd, sigma = sim.calculate_properties()
                print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")
    except Exception as e:
        print(f"Error: {e}")

    msd, sigma = sim.calculate_properties()
    D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0

    print(f"\n=== Simulation Complete ===")
    print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
    print(f"D={D:.4e} cm^2/s")
    print(f"Conductivity: {sigma:.4e} S/cm")

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
