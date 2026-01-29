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

# === 2. Site Identification and Energy Landscape Parameters ===
# Based on [Diagnosis] and [Knowledge Context] (npj Computational Materials 2025)
E_BASE = 0.30      # Base migration barrier (eV)
E_SITE_DIFF = 0.12 # Energy difference between 24d and 96h sites (eV)
E_NN = 0.06        # Li-Li repulsion energy (eV)
T = 300            # Temperature (K)
KB = 8.6173e-5     # Boltzmann constant (eV/K)
NU_0 = 1e12        # Attempt frequency (Hz)
JUMP_CUTOFF = 2.2  # Max distance for a Li hop (Angstrom)
REPULSION_CUTOFF = 2.5 # Max distance for Li-Li repulsion (Angstrom)
SPLIT_SITE_CUTOFF = 1.0 # Distance below which sites cannot be simultaneously occupied

# Identify Li sites and categorize them
li_sites = [i for i, site in enumerate(structure) if site.specie.symbol == "Li"]
site_coords = np.array([structure[i].coords for i in li_sites])
site_frac_coords = np.array([structure[i].frac_coords for i in li_sites])

# Categorize sites: 24d (tetrahedral) vs 96h (octahedral)
# 24d sites in Ia-3d are at (1/8, 0, 1/4) and symmetry equivalents
site_types = [] # 0 for 24d, 1 for 96h
site_energies = []
for i in li_sites:
    frac = structure[i].frac_coords % 1.0
    # Check if fractional coordinates are close to multiples of 0.125 (1/8)
    is_24d = any(np.isclose(frac, val, atol=0.02).all() for val in [[0.125, 0, 0.25], [0.375, 0, 0.75]]) # Simplified check
    if is_24d:
        site_types.append(0)
        site_energies.append(0.0) # 24d is reference
    else:
        site_types.append(1)
        site_energies.append(E_SITE_DIFF) # 96h is higher energy

site_energies = np.array(site_energies)

# === 3. Adjacency and Neighbor Mapping ===
n_sites = len(li_sites)
adj_list = [[] for _ in range(n_sites)]
repulsion_list = [[] for _ in range(n_sites)]
split_site_map = [[] for _ in range(n_sites)]

for i in range(n_sites):
    dists = np.linalg.norm(site_coords - site_coords[i], axis=1)
    # Jumps
    neighbors = np.where((dists > 0.1) & (dists < JUMP_CUTOFF))[0]
    adj_list[i] = neighbors.tolist()
    # Repulsion
    rep_neighbors = np.where((dists > 0.1) & (dists < REPULSION_CUTOFF))[0]
    repulsion_list[i] = rep_neighbors.tolist()
    # Split sites (exclusion volume)
    split_neighbors = np.where((dists > 0.01) & (dists < SPLIT_SITE_CUTOFF))[0]
    split_site_map[i] = split_neighbors.tolist()

# === 4. kMC Initialization ===
# Target Li content: Li7La3Zr2O12 -> 56 Li per unit cell (8 formula units)
# Supercell N=4 contains 64 unit cells -> 3584 Li ions
n_li_target = 3584 
occupancy = np.zeros(n_sites, dtype=int)

# Randomly place Li ions respecting split-site constraint
indices = np.random.permutation(n_sites)
placed = 0
for idx in indices:
    if placed >= n_li_target: break
    if any(occupancy[nb] == 1 for nb in split_site_map[idx]):
        continue
    occupancy[idx] = 1
    placed += 1

print(f"Initialized {placed} Li ions on {n_sites} sites.")

# === 5. kMC Simulation Loop ===
n_steps = 50000
current_time = 0.0
msd_sum = 0.0
li_positions = site_coords[occupancy == 1]
initial_li_positions = li_positions.copy()
li_indices = np.where(occupancy == 1)[0]

def get_config_energy(site_idx, current_occ):
    # Local energy contribution: site energy + repulsion
    e = site_energies[site_idx]
    for nb in repulsion_list[site_idx]:
        if current_occ[nb] == 1:
            e += E_NN
    return e

for step in range(n_steps):
    rates = []
    possible_jumps = []
    
    # Identify all valid jumps (occupied to empty, respecting split-site)
    for i_li, start_node in enumerate(li_indices):
        for end_node in adj_list[start_node]:
            if occupancy[end_node] == 0:
                # Check split-site constraint at destination
                if any(occupancy[nb] == 1 for nb in split_site_map[end_node] if nb != start_node):
                    continue
                
                # Calculate barrier using path-specific energies (A8/A4 fix)
                # E_act = E_base + 0.5 * (E_final - E_initial)
                e_initial = get_config_energy(start_node, occupancy)
                # Temporarily move to calculate final energy
                occupancy[start_node] = 0
                occupancy[end_node] = 1
                e_final = get_config_energy(end_node, occupancy)
                occupancy[end_node] = 0
                occupancy[start_node] = 1
                
                delta_e = e_final - e_initial
                barrier = E_BASE + 0.5 * delta_e
                rate = NU_0 * np.exp(-max(0, barrier) / (KB * T))
                
                rates.append(rate)
                possible_jumps.append((i_li, start_node, end_node))
    
    if not rates: break
    
    rates = np.array(rates)
    total_rate = np.sum(rates)
    
    # Time increment
    dt = -np.log(np.random.rand()) / total_rate
    current_time += dt
    
    # Select jump
    r = np.random.rand() * total_rate
    jump_idx = np.searchsorted(np.cumsum(rates), r)
    li_idx_in_list, start, end = possible_jumps[jump_idx]
    
    # Update state
    occupancy[start] = 0
    occupancy[end] = 1
    li_indices[li_idx_in_list] = end
    li_positions[li_idx_in_list] += (site_coords[end] - site_coords[start])
    
    if step % 10000 == 0:
        msd = np.mean(np.sum((li_positions - initial_li_positions)**2, axis=1))
        print(f"Step {step}: Time = {current_time:.2e} s, MSD = {msd:.2f} A^2")

# === 6. Conductivity Calculation ===
msd = np.mean(np.sum((li_positions - initial_li_positions)**2, axis=1))
D = (msd * 1e-16) / (6 * current_time) # cm^2/s
vol_cm3 = structure.volume * 1e-24
conc_li = n_li_target / vol_cm3 # ions/cm^3
q = 1.602e-19 # C
kb_j = 1.38e-23 # J/K
sigma = (conc_li * (q**2) * D) / (kb_j * T) # S/cm

print(f"\n--- Results ---")
print(f"Diffusion Coefficient (D): {D:.2e} cm^2/s")
print(f"Ionic Conductivity (sigma): {sigma:.2e} S/cm")