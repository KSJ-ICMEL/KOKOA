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
# LLZO has 24d (tetrahedral) and 96h (octahedral) sites.
# 96h sites are split pairs (~0.81 A apart).
li_indices = [i for i, s in enumerate(structure) if s.species_string == 'Li']
num_li_sites = len(li_indices)
idx_map = {li_idx: i for i, li_idx in enumerate(li_indices)}

site_type = {}  # 0...num_li_sites-1 -> '24d' or '96h'
split_partners = {} # 0...num_li_sites-1 -> partner index or None

# Identify split sites and classify by coordination/proximity
for i, li_idx in enumerate(li_indices):
    # Find neighbors within 1.1 A to identify split 96h pairs
    neighs = structure.get_neighbors(structure[li_idx], 1.1)
    li_neighs = [n for n in neighs if n.species_string == 'Li']
    
    if len(li_neighs) > 0:
        site_type[i] = '96h'
        split_partners[i] = idx_map[li_neighs[0].index]
    else:
        site_type[i] = '24d'
        split_partners[i] = None

# === 3. Network Connectivity (T-O-T) ===
# Transitions occur between neighboring tetrahedral (24d) and octahedral (96h) sites.
adj = [[] for _ in range(num_li_sites)]
for i, li_idx in enumerate(li_indices):
    neighs = structure.get_neighbors(structure[li_idx], 2.2)
    for n in neighs:
        if n.species_string == 'Li':
            j = idx_map[n.index]
            dist = n.nn_distance
            # T-O distance is typically 1.6 - 2.1 A
            if 1.5 < dist < 2.2:
                if site_type[i] != site_type[j]:
                    adj[i].append(j)

# === 4. Energetics and kMC Parameters ===
T = 300.0  # Temperature (K)
kb_ev = 8.617333e-5
kb_si = 1.380649e-23
e_si = 1.6021766e-19
nu_0 = 1e13  # Attempt frequency (Hz)

# Diagnosis-based energy improvements:
# 1. Lattice relaxation (A1) contribution to barrier
E_A1 = 0.15  # eV
# 2. Base activation energy (higher than the failed 0.3 eV)
E_base = 0.35  # eV
# 3. Thermodynamic traps: 96h sites are energetically favorable (lower energy)
# Site energies (relative)
energies = {i: (0.15 if site_type[i] == '24d' else 0.0) for i in range(num_li_sites)}

# === 5. Initial Li Distribution ===
# Stoichiometric Li content: 56 Li per 120 sites (7/15 occupancy)
num_li_to_place = int(num_li_sites * 56 / 120)
occupied = np.zeros(num_li_sites, dtype=bool)
li_at_sites = []

indices = np.arange(num_li_sites)
np.random.shuffle(indices)
for idx in indices:
    if len(li_at_sites) >= num_li_to_place:
        break
    # Respect split-site exclusion: two Li cannot occupy the same 96h pair
    partner = split_partners[idx]
    if not occupied[idx] and (partner is None or not occupied[partner]):
        occupied[idx] = True
        li_at_sites.append(idx)

# === 6. kMC Simulation Loop ===
num_steps = 5000
total_time = 0.0
displacements = np.zeros((num_li_to_place, 3))

def get_vector(idx_i, idx_j):
    """Calculate shortest displacement vector between two structure indices."""
    diff = structure[idx_j].frac_coords - structure[idx_i].frac_coords
    diff = diff - np.round(diff)
    return structure.lattice.get_cartesian_coords(diff)

print(f"Starting kMC for {num_steps} steps...")
for step in range(num_steps):
    rates = []
    events = [] # (li_id, current_site, target_site)
    
    for li_id, s_i in enumerate(li_at_sites):
        for s_j in adj[s_i]:
            # Check if target site is empty
            if not occupied[s_j]:
                # Check split partner exclusion for target site
                partner_j = split_partners[s_j]
                if partner_j is None or not occupied[partner_j]:
                    # Calculate rate: nu = nu0 * exp(-(E_base + E_A1 + max(0, dE)) / kT)
                    dE = energies[s_j] - energies[s_i]
                    barrier = E_base + E_A1 + max(0, dE)
                    rate = nu_0 * np.exp(-barrier / (kb_ev * T))
                    rates.append(rate)
                    events.append((li_id, s_i, s_j))
    
    total_rate = sum(rates)
    if total_rate == 0:
        print("No possible hops remaining.")
        break
    
    # Time step
    dt = -np.log(np.random.random()) / total_rate
    total_time += dt
    
    # Select event
    r = np.random.random() * total_rate
    event_idx = np.searchsorted(np.cumsum(rates), r)
    li_id, s_i, s_j = events[event_idx]
    
    # Update state
    occupied[s_i] = False
    occupied[s_j] = True
    li_at_sites[li_id] = s_j
    
    # Track displacement
    vec = get_vector(li_indices[s_i], li_indices[s_j])
    displacements[li_id] += vec

# === 7. Conductivity Calculation ===
msd = np.mean(np.sum(displacements**2, axis=1))
# D = MSD / (6 * t)
msd_si = msd * 1e-20 # A^2 to m^2
vol_si = structure.volume * 1e-30 # A^3 to m^3
# Sigma = (n * q^2 * D) / (kb * T) = (N_li * q^2 * MSD) / (6 * t * V * kb * T)
sigma = (num_li_to_place * (e_si**2) * msd_si) / (6 * total_time * vol_si * kb_si * T)

print(f"Simulation complete.")
print(f"Total Time: {total_time:.4e} s")
print(f"MSD: {msd:.4f} A^2")
print(f"Ionic Conductivity: {sigma * 0.01:.4e} S/cm") # S/m to S/cm