import os, sys, json
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# === 1. Structure Loading ===
script_dir = os.path.dirname(os.path.abspath(__file__))
cif_path = os.path.join(script_dir, "LLZO.cif")
structure = Structure.from_file(cif_path)

# Identify Li sites and their Wyckoff positions in the unit cell
sga = SpacegroupAnalyzer(structure)
symm_struct = sga.get_symmetrized_structure()
unit_li_indices = [i for i, s in enumerate(structure) if s.species.symbol == "Li"]
unit_wyckoffs = []
for idx in unit_li_indices:
    found = False
    for group in symm_struct.equivalent_sites:
        if structure[idx] in group:
            unit_wyckoffs.append(symm_struct.get_wyckoff_label(group[0]))
            found = True
            break
    if not found: unit_wyckoffs.append("96h")

num_unit_li_sites = len(unit_li_indices)

N = 4  # Supercell expansion
structure.make_supercell([N, N, N])
print(f"Supercell: {N}x{N}x{N}, Total atoms: {len(structure)}")

# === 2. Site Classification and Adjacency ===
li_indices = [i for i, s in enumerate(structure) if s.species.symbol == "Li"]
num_sites = len(li_indices)
site_types = []
for i in range(num_sites):
    # Map supercell index back to unit cell Wyckoff label
    unit_idx = i % num_unit_li_sites
    site_types.append(unit_wyckoffs[unit_idx])

# Build neighbor list (cutoff 3.0 A based on LLZO lattice)
adj = [[] for _ in range(num_sites)]
coords = structure.cart_coords[li_indices]
lattice = structure.lattice

for i in range(num_sites):
    # Use pymatgen's neighbor search for periodic boundaries
    neighbors = structure.get_neighbors(structure[li_indices[i]], 3.0)
    for nb in neighbors:
        if nb.species.symbol == "Li":
            # Find index in li_indices
            # This is a bit slow but only done once
            dist_sq = np.sum((coords - nb.coords)**2, axis=1)
            nb_idx = np.argmin(dist_sq)
            if dist_sq[nb_idx] < 0.01: continue # Self
            adj[i].append((nb_idx, nb.nn_distance))

# === 3. kMC Parameters ===
T = 300.0  # Temperature (K)
kB = 8.617333262145e-5  # Boltzmann constant (eV/K)
nu = 1e13  # Attempt frequency (Hz)

# Energy Landscape (Evidence-based from Knowledge Context)
# 24d is the low-energy site. 96h is higher.
# Bottlenecks (96h-96h) have higher migration barriers.
E_site = {'24d': 0.0, '96h': 0.15} # eV
# Migration barriers (intrinsic part)
# 24d <-> 96h is easier than 96h <-> 96h
E_mig_base = {
    ('24d', '96h'): 0.45,
    ('96h', '24d'): 0.45,
    ('96h', '96h'): 0.55,
    ('24d', '24d'): 0.60
}

def get_activation_energy(i, j):
    ti, tj = site_types[i], site_types[j]
    # Barrier = max(0, Ej - Ei) + Emig
    # This ensures detailed balance
    de = E_site.get(tj, 0.15) - E_site.get(ti, 0.0)
    emig = E_mig_base.get((ti, tj), 0.50)
    return max(0.0, de) + emig

# === 4. Initialization ===
# LLZO has ~56 Li per unit cell (120 sites total)
# Occupancy ~ 0.46
target_li_count = 56 * (N**3)
occupied = np.zeros(num_sites, dtype=bool)

# Enforce "no two Li in split 96h sites" (dist < 1.3 A)
available_indices = list(range(num_sites))
np.random.shuffle(available_indices)
li_count = 0
for idx in available_indices:
    if li_count >= target_li_count: break
    # Check if any neighbor is too close
    too_close = False
    for nb_idx, dist in adj[idx]:
        if dist < 1.3 and occupied[nb_idx]:
            too_close = True
            break
    if not too_close:
        occupied[idx] = True
        li_count += 1

current_li_indices = np.where(occupied)[0]
li_pos = coords[current_li_indices]
li_start_pos = np.copy(li_pos)
total_time = 0.0
steps = 50000

# === 5. kMC Loop ===
for step in range(steps):
    rates = []
    jumps = []
    
    # Identify all valid jumps (occupied -> empty)
    for li_idx_in_list, site_idx in enumerate(current_li_indices):
        for nb_idx, dist in adj[site_idx]:
            if not occupied[nb_idx]:
                # Check split-site constraint for destination
                too_close = False
                for nnb_idx, nnb_dist in adj[nb_idx]:
                    if nnb_dist < 1.3 and occupied[nnb_idx] and nnb_idx != site_idx:
                        too_close = True
                        break
                if too_close: continue
                
                ea = get_activation_energy(site_idx, nb_idx)
                rate = nu * np.exp(-ea / (kB * T))
                rates.append(rate)
                jumps.append((li_idx_in_list, site_idx, nb_idx))
    
    rates = np.array(rates)
    total_rate = np.sum(rates)
    
    if total_rate == 0: break
    
    # Select jump
    r = np.random.rand() * total_rate
    jump_idx = np.searchsorted(np.cumsum(rates), r)
    li_idx_in_list, old_site, new_site = jumps[jump_idx]
    
    # Update time
    dt = -np.log(np.random.rand()) / total_rate
    total_time += dt
    
    # Update state
    occupied[old_site] = False
    occupied[new_site] = True
    current_li_indices[li_idx_in_list] = new_site
    
    # Update displacement (handling PBC)
    disp = coords[new_site] - coords[old_site]
    # Minimal image convention for displacement
    frac_disp = lattice.get_fractional_coords(disp)
    frac_disp = frac_disp - np.round(frac_disp)
    real_disp = lattice.get_cartesian_coords(frac_disp)
    li_pos[li_idx_in_list] += real_disp

# === 6. Analysis ===
msd = np.mean(np.sum((li_pos - li_start_pos)**2, axis=1))
D = msd / (6.0 * total_time) # A^2/s

# Conductivity sigma = (D * n * e^2) / (kB * T)
# n = Li concentration (ions / cm^3)
vol_cm3 = structure.volume * 1e-24
n = li_count / vol_cm3
e = 1.602176634e-19 # C
kB_J = 1.380649e-23 # J/K
D_cm2s = D * 1e-16

sigma = (D_cm2s * n * (e**2)) / (kB_J * T)

print(f"Simulation Time: {total_time:.2e} s")
print(f"MSD: {msd:.4f} A^2")
print(f"Diffusion Coefficient: {D_cm2s:.2e} cm^2/s")
print(f"Ionic Conductivity: {sigma:.2e} S/cm")