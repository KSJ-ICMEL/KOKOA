"""
KOKOA Simulation #2
Generated: 2026-01-12 22:25:43
"""
import os, sys, traceback

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_222437')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure
    import os, sys, math

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kokoa.config import Config

    cif_path = "./Li4.47La3Zr2O12.cif"
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)
    N = 8
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
            self.particle_frac = {}
            p_id = 0
            for idx, s in enumerate(initial_sites):
                if s['state'] == 1:
                    cart = structure.lattice.get_cartesian_coords(s['coords'])
                    self.site_to_particle[idx] = p_id
                    self.particle_positions[p_id] = {'start': np.array(cart), 'current': np.array(cart)}
                    self.particle_frac[p_id] = np.array(s['coords'])
                    p_id += 1
            self.li_indices = set(self.site_to_particle.keys())
            self.num_particles = len(self.li_indices)
            self.current_time = 0.0
            self.step_count = 0
            self.q = 1.602e-19
            self.eps0 = 8.854e-12
            self.epsr = 20.0
            self.kb = 8.617e-5
            self.prefactor = (self.q**2/(4*np.pi*self.eps0*self.epsr)) / (1.602e-19) * 1e10
            self.cutoff_coulomb = 10.0
            self.electric_field = np.array(getattr(Config, 'ELECTRIC_FIELD', [0.0, 0.0, 0.0]), dtype=float)

        def _coulomb_rate(self, src, tgt, vec):
            r_src_frac = np.array(structure[src].frac_coords)
            r_tgt_frac = np.array(structure[tgt].frac_coords)
            src_id = self.site_to_particle[src]
            sum_term = 0.0
            for pid, pos in self.particle_positions.items():
                if pid == src_id:
                    continue
                frac_j = self.particle_frac[pid]
                diff_frac = frac_j - r_src_frac
                diff_frac -= np.round(diff_frac)
                dist_src = np.linalg.norm(structure.lattice.get_cartesian_coords(diff_frac))
                diff_frac = frac_j - r_tgt_frac
                diff_frac -= np.round(diff_frac)
                dist_tgt = np.linalg.norm(structure.lattice.get_cartesian_coords(diff_frac))
                if dist_src < self.cutoff_coulomb:
                    sum_term += 1.0/dist_tgt - 1.0/dist_src
            delta_E = self.prefactor * sum_term
            field_term = np.dot(self.electric_field, vec*1e-10)
            delta_E += field_term
            rate = self.params['nu'] * math.exp(-(self.params['E_a'] + delta_E)/(self.kb*self.params['T']))
            return rate

        def run_step(self):
            events, rates, total = [], [], 0.0
            for src in self.li_indices:
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[tgt] == 0:
                        rate = self._coulomb_rate(src, tgt, vec)
                        total += rate
                        events.append((src, tgt, vec))
                        rates.append(total)
            if total == 0:
                return False
            self.current_time += -np.log(np.random.rand())/total
            self.step_count += 1
            idx = np.searchsorted(rates, np.random.uniform(0, total))
            src, tgt, vec = events[idx]
            src_id = self.site_to_particle.pop(src)
            self.particle_positions[src_id]['current'] += vec
            self.particle_frac[src_id] = np.array(structure[tgt].frac_coords)
            self.occupancy[src], self.occupancy[tgt] = 0, 1
            self.site_to_particle[tgt] = src_id
            self.li_indices.discard(src)
            self.li_indices.add(tgt)
            return True

        def calculate_properties(self):
            if self.current_time == 0:
                return 0.0, 0.0
            msd = np.mean([np.sum((p['current']-p['start'])**2) for p in self.particle_positions.values()])
            D = msd/(6*self.current_time)*1e-16
            n = self.num_particles/(self.params['volume']*1e-24)
            sigma = n*self.q**2*D/(1.38e-23*self.params['T'])
            return msd, sigma

    sim_params = {'T':300,'E_a':0.28,'nu':1e13,'volume':structure.volume}
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
    print(f"\n=== Simulation Complete ===")
    print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
    print(f"D={msd/(6*sim.current_time)*1e-16:.4e} cm^2/s")
    print(f"Conductivity: {sigma} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
