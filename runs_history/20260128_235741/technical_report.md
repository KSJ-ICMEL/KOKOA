# Batch Technical Report

## Result: NEUTRAL

## Analysis

The update introduced **geometry-dependent, heterogeneous migration barriers** intended to mimic **local lattice relaxation (“framework breathing”) around migrating Li**. Instead of a single, configuration‑independent activation energy, each Li–Li hop now has a barrier \(E_a(d,s)\) that depends on:

- Li–Li separation \(d\) (proxy for local path width/strain)
- An asymmetry metric \(s\) derived from differences in local Li–Li environments (proxy for local distortion/volume imbalance)

The barriers are constrained to a physically reasonable range, \(E_a \in [0.60, 1.00]\) eV, consistent with NEB-derived estimates for LLZO-like systems, and the global “fallback” barrier was raised from 0.30 to 0.80 eV.

**Physics rationale:**  
In garnet-type Li conductors like LLZO, Li migration is coupled to host-lattice deformation: oxygens and polyhedra tilt and “breathe” as Li hops, which modifies bottleneck widths and local site energies. This generally leads to:

- **Higher and more dispersed barriers** than a single rigid-lattice value.
- Strong dependence on local coordination and Li ordering.

The new model encodes this by:

1. Computing a local environment metric (average Li–Li distance within 3.5 Å) as a **proxy for local volume/strain**.
2. Using differences in this metric between a pair of Li sites as an **asymmetry measure**, penalizing hops between strongly mismatched environments.
3. Increasing \(E_a\) with both \(d\) and asymmetry, then computing heterogeneous rates \(r_{ij} = \nu \exp[-E_a(d,s)/(k_BT)]\).

In the kMC loop, the BKL algorithm is correctly generalized to use hop-specific rates: cumulative rates instead of a uniform base rate, and geometry-dependent selection of events. This is physically more realistic than the previous homogeneous-barrier model.

**Why the result is NEUTRAL (error increased by +0.98 orders):**

- Even though the **code is stable and physically more detailed**, the **predicted conductivity dropped** substantially (log σ from −5.71 target to −9.78), meaning Li ions diffuse too slowly.
- This is consistent with having **over‑penalized migration pathways**:
  - The new barrier window (0.60–1.00 eV) plus the fallback 0.80 eV is significantly higher than the previous global 0.30 eV. At 300 K, \(\exp(-\Delta E/(k_BT))\) is extremely sensitive: adding 0.3–0.5 eV can reduce rates by several orders of magnitude.
  - The asymmetry penalty and distance penalty are both scaled with a factor of 0.7, which may systematically push many frequently used hops close to the upper bound (1.0 eV), effectively “throttling” the diffusive network.
- Because the new barriers are **heuristically mapped** rather than directly fitted to NEB data for each path, the physics is directionally correct (framework relaxation tends to raise and diversify barriers) but **quantitatively too severe**, leading to underestimation of σ and thus increased error.

Thus, the physics-driven refinement (lattice relaxation, heterogeneous barriers) is plausible and implemented without numerical instability, but the chosen parameterization overshoots, degrading agreement with the target. This warrants a NEUTRAL classification rather than SUCCESS: the model is richer but not yet calibrated.

## Advice for Future Scientists

1. **Calibrate the geometry–barrier mapping directly to NEB (or DFT) data.**  
   - Extract a set of representative Li hops from NEB, recording:
     - Exact Li–Li hop distances and local structural descriptors (e.g., bottleneck O–O distances, polyhedral distortion metrics).
     - The corresponding NEB barriers.
   - Fit a simple regression (linear or low-order) from descriptors \((d, s, \text{…})\) to \(E_a\), rather than manually choosing E_min/E_max and scaling factors (0.7). This will anchor the heterogeneous model to actual lattice-relaxed energetics and should reduce the large underestimation of conductivity.

2. **Systematically explore barrier scaling to bracket the experimental conductivity.**  
   - Treat the current \(E_{\min}, E_{\max}\), and the 0.7 prefactors as tunable hyperparameters:
     - Try narrower ranges, e.g. \(E_{\min}=0.45\)–0.55 eV, \(E_{\max}=0.80\)–0.90 eV.
     - Reduce sensitivity to asymmetry (e.g. 0.3–0.5 instead of 0.7) so that not all asymmetric hops are heavily suppressed.
   - Run short kMC tests and track log σ vs. these parameters to quickly identify a regime where σ is within ~1 order of magnitude of the target, then refine.

3. **Refine the local environment descriptor to better represent framework breathing.**  
   - The current metric is an average Li–Li distance, which conflates Li crowding with host-lattice strain. Consider:
     - Including non‑Li neighbors (e.g., average O–O distances along the hop bottleneck, or local Voronoi volumes) if available from the structure, to more directly encode “breathing” of the anion framework.
     - Differentiating between hops within spacious channels and those near partially occupied, crowded sites, possibly by counting Li neighbors as a separate occupancy term.
   - Keep the functional form simple, but ensure that descriptors are more tightly tied to the actual deformation modes that NEB shows are rate-determining.

These steps will move the model from a physically motivated heuristic towards a quantitatively calibrated, lattice-relaxation-aware kMC description that can match experimental/DFT conductivities without sacrificing numerical robustness.

## Full Diff
```diff
--- previous.py+++ current.py@@ -12,39 +12,114 @@ 
 # Initialize Li sites with occupancy probability
 initial_sites = []
-for site in structure:
+li_site_indices = []
+for idx, site in enumerate(structure):
     if "Li" in [s.symbol for s in site.species.elements]:
         prob = site.species.get("Li", 0)
         state = 1 if np.random.rand() < prob else 0
         initial_sites.append({"coords": site.frac_coords, "state": state})
+        li_site_indices.append(idx)
 
 print(f"Li sites initialized: {len(initial_sites)}")
 
-# === 2. Build Adjacency Graph ===
+# Map from structure index to Li-sublattice index and back
+struct_to_li = {s_idx: li_idx for li_idx, s_idx in enumerate(li_site_indices)}
+li_to_struct = {li_idx: s_idx for li_idx, s_idx in enumerate(li_site_indices)}
+
+# === 2. Build Adjacency Graph with Geometry-Dependent Barriers ===
 cutoff = 4.0  # Angstrom
 neighbors_data = structure.get_all_neighbors(r=cutoff)
+
+# Parameters for geometry-dependent activation energy
+E_min = 0.60  # eV, lower bound for relatively unstrained hops (from NEB ~0.67 eV range)
+E_max = 1.00  # eV, upper bound for strongly strained / narrow paths
+d_ref = 3.0   # Å, reference Li-Li distance for low-strain hops
+s_ref = 0.5   # Å, reference asymmetry scale
+
+def compute_activation_barrier(distance, asymmetry):
+    """
+    Geometry-dependent activation barrier E_a(d, s).
+
+    distance: cartesian Li-Li separation (Å)
+    asymmetry: |d1 - d2|, where d1 and d2 are distances from both Li sites to a shared local environment proxy.
+               Here we approximate asymmetry using local Li-Li neighbor imbalance.
+    The barrier increases with both distance (narrower, more strained connections) and asymmetry
+    to mimic lattice-relaxation effects making many paths less favorable.
+    """
+    # Distance term: penalize longer hops (proxy for stronger local distortion when relaxed)
+    delta_d = max(distance - d_ref, 0.0)
+    # Asymmetry term: penalize more asymmetric environments
+    delta_s = max(asymmetry - s_ref, 0.0)
+
+    # Scale contributions so that E_a spans [E_min, E_max] over typical geometry variations
+    # Functional form is linear in geometry descriptors, consistent with using NEB-derived ranges.
+    geom_factor = 1.0 + 0.7 * delta_d + 0.7 * delta_s
+    E_a = E_min * geom_factor
+    # Cap within [E_min, E_max]
+    if E_a < E_min:
+        E_a = E_min
+    if E_a > E_max:
+        E_a = E_max
+    return E_a
+
+# Precompute a simple local environment descriptor for each Li site:
+# average distance to Li neighbors within a smaller cutoff, as a proxy for local volume/strain.
+local_env_cutoff = 3.5  # Å
+li_cart_coords = [structure[li_to_struct[i]].coords for i in range(len(li_site_indices))]
+li_cart_coords = np.array(li_cart_coords)
+
+local_env_metric = np.zeros(len(li_site_indices), dtype=float)
+for i, coord in enumerate(li_cart_coords):
+    # Distances to all other Li sites (periodic via Pymatgen can be more exact, but we keep current neighbors_data style)
+    dists = np.linalg.norm(li_cart_coords - coord, axis=1)
+    mask = (dists > 1e-3) & (dists < local_env_cutoff)
+    if np.any(mask):
+        local_env_metric[i] = np.mean(dists[mask])
+    else:
+        # Isolated or sparse environment; assign a larger effective distance (more open volume)
+        local_env_metric[i] = local_env_cutoff
+
 adj_list = {}
-
-for i, site in enumerate(structure):
-    if "Li" not in site.species.elements[0].symbol:
-        continue
+barrier_dict = {}
+
+for li_idx, s_idx in enumerate(li_site_indices):
+    site = structure[s_idx]
     neighbors = []
-    for nb in neighbors_data[i]:
-        if "Li" in structure[nb.index].species.elements[0].symbol:
-            frac_diff = structure[nb.index].frac_coords - site.frac_coords + nb.image
-            cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
-            neighbors.append((nb.index, cart_disp))
-    adj_list[i] = neighbors
-
-print(f"Graph built (cutoff={cutoff}A)")
-
-# === 3. kMC Simulator (BKL Algorithm) ===
+    for nb in neighbors_data[s_idx]:
+        nb_s_idx = nb.index
+        # Only Li-Li hops are considered
+        if nb_s_idx not in struct_to_li:
+            continue
+        tgt_li_idx = struct_to_li[nb_s_idx]
+
+        # Compute cartesian displacement using given image
+        frac_diff = structure[nb_s_idx].frac_coords - site.frac_coords + nb.image
+        cart_disp = structure.lattice.get_cartesian_coords(frac_diff)
+        distance = np.linalg.norm(cart_disp)
+
+        # Compute simple asymmetry metric using difference in local environment descriptor
+        asymmetry = abs(local_env_metric[li_idx] - local_env_metric[tgt_li_idx])
+
+        # Geometry-dependent activation energy for this hop
+        E_a_ij = compute_activation_barrier(distance, asymmetry)
+
+        neighbors.append((tgt_li_idx, cart_disp))
+        barrier_dict[(li_idx, tgt_li_idx)] = E_a_ij
+
+    adj_list[li_idx] = neighbors
+
+print(f"Graph built (cutoff={cutoff}A) with geometry-dependent barriers")
+
+# === 3. kMC Simulator (BKL Algorithm) with Heterogeneous Barriers ===
 class KMCSimulator:
-    def __init__(self, structure, adj_list, initial_sites, params):
+    def __init__(self, structure, adj_list, initial_sites, params, barrier_dict):
         self.params = params
         self.adj_list = adj_list
+        self.barrier_dict = barrier_dict
+
+        # Occupancy only on Li sublattice
         self.occupancy = np.array([s['state'] for s in initial_sites], dtype=int)
-        
+
         self.site_to_particle = {}
         self.particle_positions = {}
         p_id = 0
@@ -52,37 +127,61 @@             if s['state'] == 1:
                 start = structure.lattice.get_cartesian_coords(s['coords'])
                 self.site_to_particle[idx] = p_id
-                self.particle_positions[p_id] = {'start': np.array(start), 'current': np.array(start)}
+                self.particle_positions[p_id] = {
+                    'start': np.array(start, dtype=float),
+                    'current': np.array(start, dtype=float),
+                }
                 p_id += 1
-        
+
         self.li_indices = set(self.site_to_particle.keys())
         self.num_particles = len(self.li_indices)
         self.current_time = 0.0
         self.step_count = 0
-        
-        kb = 8.617e-5  # eV/K
-        self.base_rate = params['nu'] * np.exp(-params['E_a'] / (kb * params['T']))
+
+        self.kb = 8.617e-5  # eV/K
+        self.nu = params['nu']
+        self.T = params['T']
+
+        # Precompute maximum possible rate factor to avoid repeated exponentials where possible
+        self.rate_cache = {}
+
+    def get_rate(self, src, tgt):
+        key = (src, tgt)
+        if key in self.rate_cache:
+            return self.rate_cache[key]
+        E_a = self.barrier_dict.get(key, self.params['E_a'])
+        rate = self.nu * np.exp(-E_a / (self.kb * self.T))
+        self.rate_cache[key] = rate
+        return rate
 
     def run_step(self):
-        events, rates, total = [], [], 0.0
+        events = []
+        cumulative_rates = []
+        total_rate = 0.0
+
+        # Build event list with heterogeneous rates
         for src in self.li_indices:
             for tgt, vec in self.adj_list.get(src, []):
                 if self.occupancy[tgt] == 0:
-                    total += self.base_rate
+                    r = self.get_rate(src, tgt)
+                    if r <= 0.0:
+                        continue
+                    total_rate += r
                     events.append((src, tgt, vec))
-                    rates.append(total)
-        
-        if total == 0:
+                    cumulative_rates.append(total_rate)
+
+        if total_rate == 0.0:
             return False  # Deadlock
-        
+
         # BKL time advance
-        self.current_time += -np.log(np.random.rand()) / total
+        self.current_time += -np.log(np.random.rand()) / total_rate
         self.step_count += 1
-        
+
         # Select and execute event
-        idx = np.searchsorted(rates, np.random.uniform(0, total))
+        r_select = np.random.uniform(0, total_rate)
+        idx = np.searchsorted(cumulative_rates, r_select)
         src, tgt, vec = events[idx]
-        
+
         p_id = self.site_to_particle.pop(src)
         self.particle_positions[p_id]['current'] += vec
         self.occupancy[src], self.occupancy[tgt] = 0, 1
@@ -93,16 +192,19 @@ 
     def calculate_properties(self):
         if self.current_time == 0:
-            return 0, 0
-        msd = np.mean([np.sum((p['current'] - p['start'])**2) for p in self.particle_positions.values()]) # Mean Square Displacement (Å^2)
-        D = msd / (6 * self.current_time) * 1e-16  # Diffusivity (cm^2/s), MSD(t)=6Dt
-        n = self.num_particles / (self.params['volume'] * 1e-24)  # Ion concentration (ions/cm^3)
-        sigma = (n * (1.602e-19)**2 * D) / (1.38e-23 * self.params['T'])  # Nernst-Einstein Equation: σ = (n*e^2*D)/(k*T) (S/cm)
+            return 0.0, 0.0
+        msd = np.mean(
+            [np.sum((p['current'] - p['start']) ** 2) for p in self.particle_positions.values()]
+        )  # Å^2
+        D = msd / (6 * self.current_time) * 1e-16  # cm^2/s
+        n = self.num_particles / (self.params['volume'] * 1e-24)  # ions/cm^3
+        sigma = (n * (1.602e-19) ** 2 * D) / (1.38e-23 * self.params['T'])  # S/cm
         return msd, sigma
 
 # === 4. Run Simulation ===
-sim_params = {'T': 300, 'E_a': 0.30, 'nu': 1e13, 'volume': structure.volume}
-sim = KMCSimulator(structure, adj_list, initial_sites, sim_params)
+# Use a reference E_a (only as backup if geometry data missing, main barriers from barrier_dict)
+sim_params = {'T': 300, 'E_a': 0.80, 'nu': 1e13, 'volume': structure.volume}
+sim = KMCSimulator(structure, adj_list, initial_sites, sim_params, barrier_dict)
 
 target_time = 1000e-9  # 1000ns timeout
 log_interval = 100
@@ -115,27 +217,33 @@     if sim.step_count % log_interval == 0:
         msd, sigma = sim.calculate_properties()
         sigma_history.append(sigma)
-        
+
         # Check convergence
         if len(sigma_history) > 1000:
-            sigma_history.pop(0) # Keep last 1000
-            
+            sigma_history.pop(0)  # Keep last 1000
+
         if len(sigma_history) == 1000:
             avg_sigma = np.mean(sigma_history)
             std_sigma = np.std(sigma_history)
             rsd = std_sigma / avg_sigma if avg_sigma > 0 else 0
-            
-            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%")
-            
-            if rsd < 0.05: # 5% convergence criteria
+
+            print(
+                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
+                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm, RSD={rsd*100:.2f}%"
+            )
+
+            if rsd < 0.05:  # 5% convergence criteria
                 print(f"Convergence reached (RSD < 5%) at {sim.current_time*1e9:.2f}ns")
                 break
         else:
-            print(f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm")
+            print(
+                f"Step {sim.step_count}: {sim.current_time*1e9:.2f}ns, "
+                f"MSD={msd:.2f}A^2, sigma={sigma*1e3:.4f}mS/cm"
+            )
 
 # Final result
 msd, sigma = sim.calculate_properties()
-D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0
+D = msd / (6 * sim.current_time) * 1e-16 if sim.current_time > 0 else 0.0
 
 print(f"\n=== Simulation Complete ===")
 print(f"T={sim_params['T']}K, Time={sim.current_time*1e9:.2f}ns")
@@ -143,7 +251,6 @@ print(f"Conductivity: {sigma:.4e} S/cm")
 
 # Save result to JSON
-import json
 result = {
     "is_success": True,
     "conductivity": sigma,
@@ -153,7 +260,7 @@     "temperature_K": sim_params['T'],
     "steps": sim.step_count,
     "error_message": None,
-    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns"
+    "execution_log": f"Completed {sim.step_count} steps in {sim.current_time*1e9:.2f}ns",
 }
 
 result_path = os.path.join(os.path.dirname(__file__), "initial_state.json")

```