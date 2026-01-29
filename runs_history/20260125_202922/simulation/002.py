"""
KOKOA Simulation #2
Generated: 2026-01-25 20:30:16
"""
import os, sys, traceback
import numpy as np
from pymatgen.core import Structure

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/LLZO.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260125_202922')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

# Pre-load structure (available as 'structure' variable)
structure = Structure.from_file(_CIF_PATH)
print(f"Structure loaded: {len(structure)} atoms")

try:
    import os
    import json
    import numpy as np
    from pymatgen import Structure

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

    # === 3. kMC Simulator (BKL Algorithm) ===
    class KMCSimulator:
        def __init__(self, structure, adj_list, initial_sites, params):
            self.params = params
            self.adj_list = adj_list
            self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
        
            # store site types (tetrahedral "tet" or octahedral "oct") using Wyckoff labels if available
            self.site_types = []
            for site in structure:
                wyck = site.properties.get('wyckoff') if hasattr(site, 'properties') else None
                if wyck and wyck.startswith('24d'):
                    self.site_types.append('tet')
                elif wyck and wyck.startswith('96h'):
                    self.site_types.append('oct')
                else:
                    # fallback: treat unknown as tetrahedral (common for Li sites in LLZO)
                    self.site_types.append('tet')
        
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
            self.nu = params['nu']
            self.T = params['T']
        
            # base barriers for hop types (eV)
            self.base_barriers = {
                ('tet', 'oct'): params.get('E_a_tet_oct', 0.30),
                ('oct', 'tet'): params.get('E_a_oct_tet', 0.30),
                ('tet', 'tet'): params.get('E_a_tet_tet', 0.35),
                ('oct', 'oct'): params.get('E_a_oct_oct', 0.35)
            }
        
            # additional parameters for environment‑dependent barriers
            self.penalty_per_neighbor = params.get('penalty_per_neighbor', 0.05)  # eV per occupied neighbor
            self.alpha = params.get('alpha', 0.05)  # eV/Å geometric factor
            self.r_crit = params.get('r_crit', 1.0)  # Å critical bottleneck radius

        def compute_Ea(self, src, tgt, vec):
            """Calculate the local activation energy for a hop src→tgt.
            Includes base barrier, Coulombic penalty from occupied neighbors,
            and a geometric term based on the bottleneck size.
            """
            src_type = self.site_types[src]
            tgt_type = self.site_types[tgt]
            base = self.base_barriers.get((src_type, tgt_type), self.params['E_a'])
            # Coulombic penalty: count occupied neighbors of the source (excluding the target)
            n_occ = 0
            for nb_idx, _ in self.adj_list.get(src, []):
                if nb_idx == tgt:
                    continue
                if self.occupancy[nb_idx] == 1:
                    n_occ += 1
            penalty = self.penalty_per_neighbor * n_occ
            # Geometric factor from bottleneck radius (approx. half the hop distance)
            distance = np.linalg.norm(vec)
            r = distance / 2.0
            geom = self.alpha * max(0.0, r - self.r_crit)
            return base + penalty + geom

        def run_step(self):
            events = []
            rates_cumulative = []
            total_rate = 0.0
            # generate all possible hops with their instantaneous rates
            for src in self.li_indices:
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[tgt] == 0:
                        Ea = self.compute_Ea(src, tgt, vec)
                        rate = self.nu * np.exp(-Ea / (self.kb * self.T))
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
            rnd = np.random.uniform(0, total_rate)
            idx = np.searchsorted(rates_cumulative, rnd)
            src, tgt, vec = events[idx]
            # execute hop
            p_id = self.site_to_particle.pop(src)
            self.particle_positions[p_id]['current'] += vec
            self.occupancy[src] = 0
            self.occupancy[tgt] = 1
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
    sim_params = {
        'T': 298,
        'E_a': 0.30,  # fallback global barrier (eV)
        'nu': 1e13,
        'volume': structure.volume,
        # new parameters for environment‑dependent barriers
        'penalty_per_neighbor': 0.05,  # eV per occupied neighbor
        'alpha': 0.05,                # eV/Å geometric factor
        'r_crit': 1.0,                # Å critical radius
        # optional explicit base barriers per hop type (can be omitted)
        'E_a_tet_oct': 0.30,
        'E_a_oct_tet': 0.30,
        'E_a_tet_tet': 0.35,
        'E_a_oct_oct': 0.35
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

    print("\n=== Simulation Complete ===")
    print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
    print(f"D={D:.4e} cm^2/s")
    print(f"Conductivity: {sigma:.4e} S/cm")

    # Save result to JSON
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
    print(f"\nSaved result to: {result_path}")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
