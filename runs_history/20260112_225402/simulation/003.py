"""
KOKOA Simulation #3
Generated: 2026-01-12 22:55:21
"""
import os, sys, traceback

# Add project root to path for kokoa imports
# runs/xxx/simulation/xxx.py -> 3 levels up = project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_225402')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import numpy as np
    from pymatgen.core import Structure
    import os, sys, json

    # keep simulation time fixed
    try:
        from kokoa.config import Config
        target_time = Config.SIMULATION_TIME
    except Exception:
        target_time = 1e-9

    # === 1. Load structure ===
    cif_path = "./Li4.47La3Zr2O12.cif"
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    structure = Structure.from_file(cif_path)
    N = 4
    structure.make_supercell([N, N, N])

    # === 2. Build adjacency with site‑specific E_a ===
    cutoff = 4.0
    nbrs = structure.get_all_neighbors(r=cutoff)
    adj = {}
    for i, site in enumerate(structure):
        if "Li" not in site.species.elements[0].symbol:
            continue
        lst = []
        for nb in nbrs[i]:
            if "Li" in structure[nb.index].species.elements[0].symbol:
                frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
                disp = structure.lattice.get_cartesian_coords(frac_diff)
                dist = np.linalg.norm(disp)
                Ea = 0.18 + 0.01 * dist  # eV
                lst.append((nb.index, disp, Ea))
        adj[i] = lst

    # === 3. kMC simulator ===
    class KMCSimulator:
        kb = 8.617e-5  # eV/K
        eps0 = 8.854e-12
        epsr = 10.0
        q = 1.602e-19

        def __init__(self, struct, adj, init_sites, params):
            self.struct = struct
            self.adj = adj
            self.params = params
            self.nsites = len(struct)
            self.occ = np.zeros(self.nsites, dtype=int)
            self.site_to_p = {}
            self.p_pos = {}
            pid = 0
            for s in init_sites:
                idx = s["idx"]
                if s["state"]:
                    self.occ[idx] = 1
                    pos = struct.lattice.get_cartesian_coords(s["coords"])
                    self.site_to_p[idx] = pid
                    self.p_pos[pid] = {"start": pos.copy(), "cur": pos.copy()}
                    pid += 1
            self.li_idx = {i for i in self.site_to_p}
            self.np = pid
            self.t = 0.0
            self.steps = 0

        def _coulomb(self, src_idx, new_pos):
            pos_frac = self.struct.lattice.get_fractional_coords(new_pos)
            energy = 0.0
            for pid, pos in self.p_pos.items():
                if self.site_to_p.get(src_idx) == pid:
                    continue
                other_frac = self.struct.lattice.get_fractional_coords(pos["cur"])
                diff = other_frac - pos_frac
                diff -= np.round(diff)
                dist = np.linalg.norm(self.struct.lattice.get_cartesian_coords(diff))
                if dist > 0:
                    energy += self.q**2/(4*np.pi*self.eps0*self.epsr*dist)
            return energy / self.q  # eV (since q in C, energy in J, divide by eV)

        def run_step(self):
            events = []
            cum = []
            total = 0.0
            nu = self.params["nu"]
            T = self.params["T"]
            for src in self.li_idx:
                pid = self.site_to_p[src]
                cur_pos = self.p_pos[pid]["cur"]
                for tgt, disp, Ea in self.adj.get(src, []):
                    if self.occ[tgt] == 0:
                        new_pos = cur_pos + disp
                        dE = self._coulomb(src, new_pos)
                        rate = nu * np.exp(-(Ea + dE) / (self.kb * T))
                        total += rate
                        events.append((src, tgt, disp))
                        cum.append(total)
            if total == 0:
                return False
            self.t += -np.log(np.random.rand()) / total
            self.steps += 1
            r = np.random.uniform(0, total)
            idx = np.searchsorted(cum, r)
            src, tgt, disp = events[idx]
            pid = self.site_to_p.pop(src)
            self.p_pos[pid]["cur"] += disp
            self.occ[src], self.occ[tgt] = 0, 1
            self.site_to_p[tgt] = pid
            self.li_idx.discard(src)
            self.li_idx.add(tgt)
            return True

        def properties(self):
            if self.t == 0:
                return 0.0, 0.0
            msd = np.mean([np.sum((p["cur"] - p["start"])**2) for p in self.p_pos.values()])
            D = msd / (6 * self.t) * 1e-16  # cm^2/s
            n = self.np / (self.params["vol"] * 1e-24)  # cm^-3
            sigma = n * (1.602e-19)**2 * D / (1.38e-23 * self.params["T"])
            return msd, sigma

    # === 4. Prepare initial sites ===
    init_sites = []
    for i, site in enumerate(structure):
        if "Li" in site.species.elements[0].symbol:
            prob = site.species.get("Li", 0)
            state = 1 if np.random.rand() < prob else 0
            init_sites.append({"idx": i, "coords": site.frac_coords, "state": state})

    # === 5. Run simulation ===
    params = {"T": 400, "nu": 5e13, "vol": structure.volume}
    sim = KMCSimulator(structure, adj, init_sites, params)

    while sim.t < target_time:
        if not sim.run_step():
            break

    msd, sigma = sim.properties()
    D = msd / (6 * sim.t) * 1e-16 if sim.t > 0 else 0

    print(f"\n=== Simulation Complete ===")
    print(f"T={params['T']}K, Time={sim.t*1e9:.2f}ns")
    print(f"D={D:.4e} cm^2/s")
    print(f"Conductivity: {sigma:.4e} S/cm")

    # === 6. Save results ===
    res = {
        "is_success": True,
        "conductivity": sigma,
        "diffusivity": D,
        "msd": msd,
        "simulation_time_ns": sim.t * 1e9,
        "temperature_K": params["T"],
        "steps": sim.steps,
        "error_message": None,
        "execution_log": f"Completed {sim.steps} steps in {sim.t*1e9:.2f}ns"
    }
    out_path = os.path.join(os.getcwd(), "initial_state.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
