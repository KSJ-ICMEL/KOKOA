"""
KOKOA Simulation #1
Generated: 2026-01-12 23:09:02
"""
import os, sys, traceback

# Add project root to path for kokoa imports
# runs/xxx/simulation/xxx.py -> 3 levels up = project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_230828')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import os
    import numpy as np
    from pymatgen.core import Structure

    # === 1. Structure Loading ===
    cif_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Li4.47La3Zr2O12.cif",
    )
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    structure = Structure.from_file(cif_path)
    N = 4
    structure.make_supercell([N, N, N])

    # Identify Li sites
    li_sites = []
    li_index_map = {}
    for idx, site in enumerate(structure):
        if site.species.get("Li", 0) > 0:
            li_index_map[idx] = len(li_sites)
            li_sites.append(idx)

    # Occupancy array
    occupancy = np.zeros(len(li_sites), dtype=int)
    for idx in li_sites:
        prob = structure[idx].species.get("Li", 0)
        occupancy[li_index_map[idx]] = 1 if np.random.rand() < prob else 0

    # Build adjacency list for Li sites
    cutoff = 4.0
    neighbors_data = structure.get_all_neighbors(r=cutoff)
    adj_list = {}
    for idx in li_sites:
        neighs = []
        for nb in neighbors_data[idx]:
            if nb.index in li_index_map:
                frac_diff = nb.frac_coords - structure[idx].frac_coords + nb.image
                vec = structure.lattice.get_cartesian_coords(frac_diff)
                neighs.append((nb.index, vec))
        adj_list[idx] = neighs

    # === 2. kMC Simulator (BKL Algorithm) ===
    class KMCSimulator:
        def __init__(self, structure, adj_list, occupancy, li_index_map, params):
            self.structure = structure
            self.adj_list = adj_list
            self.occupancy = occupancy
            self.li_index_map = li_index_map
            self.params = params
            self.site_to_particle = {}
            self.particle_positions = {}
            p_id = 0
            for idx in li_sites:
                if occupancy[li_index_map[idx]] == 1:
                    pos = structure.lattice.get_cartesian_coords(structure[idx].frac_coords)
                    self.site_to_particle[idx] = p_id
                    self.particle_positions[p_id] = {"start": pos.copy(), "current": pos.copy()}
                    p_id += 1
            self.li_indices = {idx for idx in li_sites if occupancy[li_index_map[idx]] == 1}
            self.current_time = 0.0
            self.step_count = 0
            self.kb = 8.617e-5  # eV/K

        def run_step(self):
            events = []
            rates = []
            total = 0.0
            for src in self.li_indices:
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[self.li_index_map[tgt]] == 0:
                        n_neighbors = sum(
                            self.occupancy[self.li_index_map[n]]
                            for n, _ in self.adj_list.get(tgt, [])
                        )
                        rate = self.params["nu"] * np.exp(
                            -(self.params["E_a"] + self.params["alpha"] * n_neighbors)
                            / (self.kb * self.params["T"])
                        )
                        total += rate
                        events.append((src, tgt, vec))
                        rates.append(total)
            if total == 0:
                return False
            self.current_time += -np.log(np.random.rand()) / total
            self.step_count += 1
            idx = np.searchsorted(rates, np.random.rand() * total)
            src, tgt, vec = events[idx]
            p_id = self.site_to_particle.pop(src)
            self.particle_positions[p_id]["current"] += vec
            self.occupancy[self.li_index_map[src]] = 0
            self.occupancy[self.li_index_map[tgt]] = 1
            self.site_to_particle[tgt] = p_id
            self.li_indices.discard(src)
            self.li_indices.add(tgt)
            return True

        def calculate_properties(self):
            if self.current_time == 0:
                return 0.0, 0.0
            msd = np.mean(
                [
                    np.sum((p["current"] - p["start"]) ** 2)
                    for p in self.particle_positions.values()
                ]
            )
            D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
            n = len(self.particle_positions) / (self.params["volume"] * 1e-24)  # ions/cm^3
            sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params["T"])
            return msd, sigma

    # === 3. Run Simulation ===
    params = {
        "T": 300.0,
        "E_a": 0.28,
        "nu": 1e13,
        "volume": structure.volume,
        "alpha": 0.05,
    }
    sim = KMCSimulator(structure, adj_list, occupancy, li_index_map, params)
    target_time = 1e-08  # seconds
    log_interval = 2000

    while sim.current_time < target_time:
        if not sim.run_step():
            break
        if sim.step_count % log_interval == 0:
            msd, sigma = sim.calculate_properties()
            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")

    msd, sigma = sim.calculate_properties()
    print(f"Conductivity: {sigma} S/cm")
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
