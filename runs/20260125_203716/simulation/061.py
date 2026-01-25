"""
KOKOA Simulation #61
Generated: 2026-01-25 20:42:56
"""
import os, sys, json, traceback
import numpy as np
from pymatgen.core import Structure

_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/LLZO.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260125_203716')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

# Pre-load structure (available as 'structure' variable)
structure = Structure.from_file(_CIF_PATH)
print(f"Structure loaded: {len(structure)} atoms")

try:
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

    # === Define initial sites (all Li sites occupied) ===
    initial_sites = [
        {"state": 1, "coords": site.frac_coords}
        for site in structure
        if "Li" in site.species.elements[0].symbol
    ]

    # === 3. kMC Simulator (BKL Algorithm) ===
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
        
            # Physical constants
            self.kb = 8.617e-5  # eV/K
            self.T = params['T']
            self.base_nu = params.get('nu', 1e13)
            self.base_Ea = params.get('E_a', 0.30)
        
            # New parameters for Coulomb interaction
            self.epsilon_eff = params.get('epsilon_eff', 25.0)   # effective dielectric constant
            self.lambda_screen = params.get('lambda', 5.0)       # screening length in Å
            self.q_e = 1.602e-19                                 # elementary charge (C)
            self.epsilon_0 = 8.854e-12                           # vacuum permittivity (F/m)
            self.eV_to_J = 1.602e-19
            self.ang_to_m = 1e-10
        
            # Pre‑compute Cartesian coordinates of all Li sites
            self.li_site_indices = [i for i, site in enumerate(structure) if "Li" in site.species.elements[0].symbol]
            self.site_cart = {i: structure.lattice.get_cartesian_coords(site.frac_coords) for i, site in enumerate(structure) if "Li" in site.species.elements[0].symbol}
        
            # Pre‑compute Coulomb neighbor list (within 8 Å) for speed
            self.coulomb_cutoff = 8.0  # Å
            self.coulomb_neighbors = {i: [] for i in self.li_site_indices}
            all_nb = structure.get_all_neighbors(r=self.coulomb_cutoff)
            for i in self.li_site_indices:
                for nb in all_nb[i]:
                    j = nb.index
                    if j == i:
                        continue
                    if "Li" not in structure[j].species.elements[0].symbol:
                        continue
                    vec = structure.lattice.get_cartesian_coords(nb.frac_coords - structure[i].frac_coords + nb.image) / self.ang_to_m
                    self.coulomb_neighbors[i].append((j, nb.distance, vec))
        
            # Track net charge displacement for Haven ratio
            self.charge_disp = np.zeros(3)
    
        def _yukawa_energy(self, r_ang):
            """Screened Coulomb energy (eV) for a pair separated by r_ang (Å)."""
            r_m = r_ang * self.ang_to_m
            lam_m = self.lambda_screen * self.ang_to_m
            prefactor = (self.q_e ** 2) / (4 * np.pi * self.epsilon_0 * self.epsilon_eff * self.eV_to_J)
            return prefactor * np.exp(-r_m / lam_m) / r_m

        def _site_energy(self, idx):
            """Energy of occupied site idx due to all other occupied Li ions (eV)."""
            if self.occupancy[idx] == 0:
                return 0.0
            E = 0.0
            for j, dist, _ in self.coulomb_neighbors[idx]:
                if self.occupancy[j] == 1:
                    E += self._yukawa_energy(dist)
            return E

        def _saddle_energy(self, src_idx, tgt_idx):
            """Energy at the midpoint of a hop src→tgt due to all other occupied Li ions (eV)."""
            src_pos = self.site_cart[src_idx]
            tgt_pos = self.site_cart[tgt_idx]
            mid_pos = 0.5 * (src_pos + tgt_pos)
            E = 0.0
            for j in self.li_site_indices:
                if j == src_idx:
                    continue
                if self.occupancy[j] == 1:
                    r_ang = np.linalg.norm(mid_pos - self.site_cart[j]) / self.ang_to_m
                    E += self._yukawa_energy(r_ang)
            return E

        def get_hop_rate(self, src, tgt, vec):
            """Compute environment‑dependent hop rate.
            Returns ν·exp(‑ΔE/(k_B T)).
            """
            distance = np.linalg.norm(vec)
            if distance <= 2.0:
                base_Ea = 0.35  # eV
                nu = 1e13       # Hz
            else:
                base_Ea = 0.55  # eV
                nu = 5e12       # Hz
            # Coulomb contribution
            E_site = self._site_energy(src)
            E_saddle = self._saddle_energy(src, tgt)
            delta_E = E_saddle - E_site
            Ea_eff = max(base_Ea + delta_E, 0.0)
            rate = nu * np.exp(-Ea_eff / (self.kb * self.T))
            return rate

        def run_step(self):
            events = []
            cumulative_rates = []
            total_rate = 0.0
            for src in self.li_indices:
                for tgt, vec in self.adj_list.get(src, []):
                    if self.occupancy[tgt] == 0:
                        hop_rate = self.get_hop_rate(src, tgt, vec)
                        if hop_rate <= 0:
                            continue
                        total_rate += hop_rate
                        events.append((src, tgt, vec))
                        cumulative_rates.append(total_rate)
        
            if total_rate == 0:
                return False  # Deadlock
        
            # BKL time advance
            self.current_time += -np.log(np.random.rand()) / total_rate
            self.step_count += 1
        
            # Select event
            r = np.random.uniform(0, total_rate)
            idx = np.searchsorted(cumulative_rates, r)
            src, tgt, vec = events[idx]
        
            # Execute hop
            p_id = self.site_to_particle.pop(src)
            self.particle_positions[p_id]['current'] += vec
            self.occupancy[src], self.occupancy[tgt] = 0, 1
            self.site_to_particle[tgt] = p_id
            self.li_indices.discard(src)
            self.li_indices.add(tgt)
        
            # Update net charge displacement (vector sum of all hops)
            self.charge_disp += vec
            return True

        def calculate_properties(self):
            if self.current_time == 0:
                return 0, 0
            # Tracer MSD (average over particles) in Å^2
            msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()])
            # Tracer diffusion coefficient D_tracer (cm^2/s)
            D_tracer = msd / (6 * self.current_time) * 1e-16
            # Collective charge displacement squared (Å^2)
            charge_msd = np.dot(self.charge_disp, self.charge_disp)
            # Charge diffusion coefficient D_sigma (cm^2/s)
            d = 3  # dimensionality
            D_sigma = charge_msd / (2 * d * self.num_particles * self.current_time) * 1e-16
            # Haven ratio (optional diagnostic)
            H = D_tracer / D_sigma if D_sigma != 0 else 0
            # Number density (cm^-3)
            n = self.num_particles / (self.params['volume'] * 1e-24)
            # Conductivity using charge diffusivity (Nernst‑Einstein with H)
            sigma = (n * self.q_e**2 * D_sigma) / (self.kb * self.eV_to_J * self.T)
            return msd, sigma

    # === Simulation parameters ===
    sim_params = {
        'T': 300,
        'volume': structure.lattice.volume,  # Å^3
        'epsilon_eff': 25.0,
        'lambda': 5.0,
        'nu': 1e13,
        'E_a': 0.30
    }

    sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)

    while sim.step_count < 100000:
        if not sim.run_step():
            break
        if sim.step_count % 1000 == 0:
            msd, sigma = sim.calculate_properties()
            print(f"Step {sim.step_count}: time={sim.current_time:.2e}s, σ={sigma:.3e} S/m")

    # Final properties
    msd, sigma = sim.calculate_properties()
    print(f"Final tracer MSD = {msd:.3f} Å^2, Conductivity = {sigma:.3e} S/m")

    # Save results (optional)
    output = {
        'steps': sim.step_count,
        'time': sim.current_time,
        'msd': msd,
        'conductivity': sigma
    }

    with open('kmc_output.json', 'w') as f:
        json.dump(output, f, indent=2)
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
