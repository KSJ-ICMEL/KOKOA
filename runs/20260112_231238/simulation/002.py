"""
KOKOA Simulation #2
Generated: 2026-01-12 23:13:39
"""
import os, sys, traceback

# Project root (pre-calculated by Simulator)
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_231238')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure
    import os, json

    cif_path = _CIF_PATH
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)
    N = 4
    structure.make_supercell([N, N, N])

    # Li sites
    li_sites = [i for i, s in enumerate(structure) if "Li" in s.species.elements]
    occupancy = np.zeros(len(structure), dtype=int)
    for i in li_sites:
        prob = structure[i].species.get("Li", 0)
        occupancy[i] = 1 if np.random.rand() < prob else 0

    # adjacency
    cutoff = 4.0
    neighbors_data = structure.get_all_neighbors(r=cutoff)
    adj_list = {}
    for i in li_sites:
        neigh = []
        for nb in neighbors_data[i]:
            if nb.index in li_sites:
                frac_diff = structure[nb.index].frac_coords - structure[i].frac_coords + nb.image
                vec = structure.lattice.get_cartesian_coords(frac_diff)
                neigh.append((nb.index, vec))
        adj_list[i] = neigh

    # site‑specific Ea
    base_Ea = 0.42  # eV
    pair_Ea = {}
    for src, neigh in adj_list.items():
        for tgt, vec in neigh:
            dist = np.linalg.norm(vec)
            pair_Ea[(src, tgt)] = base_Ea + 0.05 * (dist - 2.0)

    class KMCSimulator:
        def __init__(self, structure, adj_list, occupancy, params):
            self.structure = structure
            self.adj_list = adj_list
            self.occupancy = occupancy
            self.site_to_particle = {}
            self.particle_positions = {}
            p_id = 0
            for idx in li_sites:
                if occupancy[idx]:
                    pos = structure.lattice.get_cartesian_coords(structure[idx].frac_coords)
                    self.site_to_particle[idx] = p_id
                    self.particle_positions[p_id] = {'start': pos.copy(), 'current': pos.copy()}
                    p_id += 1
            self.li_indices = {i for i in li_sites if occupancy[i]}
            self.num_particles = p_id
            self.current_time = 0.0
            self.step_count = 0
            self.kb = 8.617e-5
            self.nu = params['nu']
            self.T = params['T']

        def run_step(self):
            events, cum_rates = [], []
            total = 0.0
            for src in self.li_indices:
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[tgt] == 0:
                        Ea = pair_Ea[(src, tgt)]
                        rate = self.nu * np.exp(-Ea / (self.kb * self.T))
                        total += rate
                        events.append((src, tgt, vec))
                        cum_rates.append(total)
            if total == 0:
                return False
            self.current_time += -np.log(np.random.rand()) / total
            self.step_count += 1
            idx = np.searchsorted(cum_rates, np.random.rand() * total)
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
                return 0.0, 0.0
            msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])
            D = msd * 1e-16 / (6 * self.current_time)
            n = self.num_particles / (self.structure.volume * 1e-24)
            sigma = n * (1.602e-19)**2 * D / (1.38e-23 * self.T)
            return msd, sigma

    sim_params = {'T': 300, 'nu': 1e13}
    sim = KMCSimulator(structure, adj_list, occupancy, sim_params)

    target_time = 1e-08
    log_interval = 2000

    try:
        while sim.current_time < target_time:
            if not sim.run_step():
                break
            if sim.step_count % log_interval == 0:
                msd, sigma = sim.calculate_properties()
                print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")
        msd, sigma = sim.calculate_properties()
        D = msd * 1e-16 / (6 * sim.current_time) if sim.current_time > 0 else 0
        print(f"\n=== Simulation Complete ===")
        print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
        print(f"D={D:.4e} cm^2/s")
        print(f"Conductivity: {sigma} S/cm")
    except Exception as e:
        print(f"Error: {e}")

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
    result_path = os.path.join(os.getcwd(), "initial_state.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
