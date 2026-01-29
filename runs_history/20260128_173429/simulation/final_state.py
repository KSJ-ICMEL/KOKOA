import os, sys, json
import numpy as np
from pymatgen.core import Structure

# === 1. Structure Loading ===
script_dir = os.path.dirname(os.path.abspath(__file__))
cif_path = os.path.join(script_dir, "LLZO.cif")
structure = Structure.from_file(cif_path)
N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

# === 2. Site Identification and Classification ===
# Extract all potential Li sites (24d and 96h)
li_indices = [i for i, s in enumerate(structure) if "Li" in [el.symbol for el in s.species.elements]]
li_sites = [structure[i] for i in li_indices]
num_sites = len(li_sites)
lattice = structure.lattice
vol_cm3 = lattice.volume * 1e-24

# Identify split-site pairs (96h sites are pairs ~0.6 A apart)
split_partners = {}
for i in range(num_sites):
    if i in split_partners: continue
    for j in range(i + 1, num_sites):
        if li_sites[i].distance(li_sites[j]) < 0.9:
            split_partners[i] = j
            split_partners[j] = i
            break

# Classify sites: 24d (Tetrahedral, T) vs 96h (Octahedral, O)
# 96h sites have a split partner; 24d sites do not.
site_types = [] # 0 for 24d (T), 1 for 96h (O)
site_energies = []
E_T = 0.0   # Ground state (eV)
E_O = 0.12  # Higher energy site (eV) - Trapping fix

for i in range(num_sites):
    if i in split_partners:
        site_types.append(1)
        site_energies.append(E_O)
    else:
        site_types.append(0)
        site_energies.append(E_T)

# Build Adjacency List (T-O jumps only, distance ~1.7 A)
adj = [[] for _ in range(num_sites)]
for i in range(num_sites):
    for j in range(i + 1, num_sites):
        d = li_sites[i].distance(li_sites[j])
        if 1.5 < d < 2.0: # Excludes split partners (<0.9) and O-O/T-T (>2.1)
            adj[i].append(j)
            adj[j].append(i)

# === 3. kMC Initialization ===
T = 300.0
kB_eV = 8.617333e-5
kB_J = 1.380649e-23
q_C = 1.602176e-19
nu0 = 1.0e12 # Attempt frequency (Hz)
E_barrier = 0.26 # Base bottleneck barrier (eV)
E_min = 0.26 # Minimum activation energy (O -> T)

# Li7La3Zr2O12 has 56 Li per unit cell
num_li = 56 * (N**3)
occupied_sites = np.zeros(num_sites, dtype=bool)
li_site_indices = [] # Maps Li_ID to Site_Index

# Randomly distribute Li ions respecting split-site exclusion
all_indices = np.arange(num_sites)
np.random.shuffle(all_indices)
for idx in all_indices:
    if len(li_site_indices) >= num_li: break
    partner = split_partners.get(idx, -1)
    if not occupied_sites[idx] and (partner == -1 or not occupied_sites[partner]):
        occupied_sites[idx] = True
        li_site_indices.append(idx)

li_site_indices = np.array(li_site_indices)
initial_frac_coords = np.array([li_sites[idx].frac_coords for idx in li_site_indices])
li_displacements = np.zeros((num_li, 3)) # Unwrapped fractional displacement

# === 4. kMC Simulation Loop ===
# Using Rejection kMC for efficiency in high-occupancy lattice
steps = 1000000
avg_z = np.mean([len(n) for n in adj])
dt_per_attempt = 1.0 / (num_li * avg_z * nu0 * np.exp(-E_min / (kB_eV * T)))
total_time = 0.0

for step in range(steps):
    # 1. Pick a random Li ion
    li_id = np.random.randint(num_li)
    curr_site = li_site_indices[li_id]
    
    # 2. Pick a random neighbor
    neighbors = adj[curr_site]
    if not neighbors: continue
    next_site = neighbors[np.random.randint(len(neighbors))]
    
    # 3. Check occupancy and split-site constraint
    if not occupied_sites[next_site]:
        partner = split_partners.get(next_site, -1)
        if partner == -1 or not occupied_sites[partner]:
            # 4. Energy Barrier Check (Metropolis-Hastings / TST)
            dE = site_energies[next_site] - site_energies[curr_site]
            E_act = E_barrier + max(0, dE)
            
            # Probability relative to the fastest possible jump (E_min)
            if np.random.rand() < np.exp(-(E_act - E_min) / (kB_eV * T)):
                # Perform Move
                old_frac = li_sites[curr_site].frac_coords
                new_frac = li_sites[next_site].frac_coords
                
                # Update unwrapped displacement with PBC correction
                diff = new_frac - old_frac
                diff -= np.round(diff)
                li_displacements[li_id] += diff
                
                # Update occupancy state
                occupied_sites[curr_site] = False
                occupied_sites[next_site] = True
                li_site_indices[li_id] = next_site
    
    total_time += dt_per_attempt

# === 5. Analysis and Conductivity ===
# Calculate MSD in Cartesian coordinates
cart_displacements = np.dot(li_displacements, lattice.matrix)
msd = np.mean(np.sum(cart_displacements**2, axis=1)) # Angstrom^2

# Diffusion coefficient D (cm^2/s)
# D = MSD / (6 * t)
D = (msd * 1e-16) / (6 * total_time)

# Ionic Conductivity sigma (S/cm)
# sigma = (n * q^2 * D) / (kB * T)
n_li_cm3 = num_li / vol_cm3
sigma = (n_li_cm3 * (q_C**2) * D) / (kB_J * T)

print(f"Simulation Time: {total_time:.2e} s")
print(f"MSD: {msd:.4f} A^2")
print(f"Diffusion Coefficient D: {D:.2e} cm^2/s")
print(f"Ionic Conductivity: {sigma:.2e} S/cm")

# Output results
results = {
    "conductivity_S_cm": sigma,
    "diffusion_cm2_s": D,
    "msd_A2": msd,
    "time_s": total_time
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)