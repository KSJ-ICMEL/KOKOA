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

# === 2. Physical Parameters ===
T = 300  # Temperature (K)
kB = 8.617333262145e-5  # Boltzmann constant (eV/K)
nu0 = 1.0e13  # Phonon frequency (Hz)
r_Li = 0.76  # Li+ ionic radius (A)
r_O = 1.38   # O2- ionic radius (A)
d_ideal = r_Li + r_O  # Ideal Li-O distance for bottleneck opening
E_static = 0.15  # Static activation energy (eV)
K_dist = 5.8     # Elastic constant for lattice distortion (eV/A^2)

# === 3. Site Identification and Hop Pre-calculation ===
li_indices = [i for i, s in enumerate(structure) if s.species_string == 'Li']
o_indices = [i for i, s in enumerate(structure) if s.species_string == 'O']

# Stoichiometry: LLZO has ~56 Li per unit cell (occupancy ~0.46)
num_li_to_occupy = int(len(li_indices) * 0.46)
occupied_indices = np.random.choice(li_indices, size=num_li_to_occupy, replace=False)
is_occupied = np.zeros(len(structure), dtype=bool)
is_occupied[occupied_indices] = True

# Pre-calculate hops between Li sites within 2.5 A
all_hops = []
hops_from = {i: [] for i in li_indices}
hops_to = {i: [] for i in li_indices}

print("Pre-calculating hops with bottleneck correction...")
for i in li_indices:
    neighbors = structure.get_neighbors(structure[i], r=2.5)
    for neigh in neighbors:
        if neigh.species_string == 'Li':
            j = neigh.index
            # Calculate bottleneck penalty (Phonon-assisted hopping)
            midpoint = (structure[i].coords + neigh.coords) / 2.0
            # Find nearest Oxygen to the hop midpoint
            nearest_os = structure.get_neighbors_in_shell(midpoint, 0, 3.0)
            d_min = min([n.distance(midpoint) for n in nearest_os if n.species_string == 'O'], default=d_ideal)
            
            # E_a = E_static + E_distortion (addresses Frozen Framework diagnosis)
            E_dist = K_dist * max(0, d_ideal - d_min)**2
            E_a = E_static + E_dist
            rate = nu0 * np.exp(-E_a / (kB * T))
            
            hop_idx = len(all_hops)
            all_hops.append({
                'from': i,
                'to': j,
                'rate': rate,
                'vec': neigh.coords - structure[i].coords
            })
            hops_from[i].append(hop_idx)
            hops_to[j].append(hop_idx)

# === 4. kMC Simulation ===
steps = 50000
total_time = 0.0
msd_sum = 0.0
# Track unwrapped displacements for MSD
displacements = {i: np.zeros(3) for i in occupied_indices}
# Map site index to the ion ID currently occupying it
site_to_ion = {idx: i for i, idx in enumerate(occupied_indices)}

# Initialize active hops (occupied -> empty)
active_hop_indices = []
for h_idx, h in enumerate(all_hops):
    if is_occupied[h['from']] and not is_occupied[h['to']]:
        active_hop_indices.append(h_idx)

print(f"Starting kMC simulation for {steps} steps...")
for step in range(steps):
    if not active_hop_indices:
        break
        
    rates = np.array([all_hops[idx]['rate'] for idx in active_hop_indices])
    r_total = np.sum(rates)
    
    # Time increment
    dt = -np.log(np.random.rand()) / r_total
    total_time += dt
    
    # Select hop
    r_val = np.random.rand() * r_total
    chosen_idx = active_hop_indices[np.searchsorted(np.cumsum(rates), r_val)]
    hop = all_hops[chosen_idx]
    
    # Execute hop
    u, v = hop['from'], hop['to']
    ion_id = site_to_ion[u]
    
    # Update MSD tracking
    displacements[ion_id] += hop['vec']
    
    # Update occupancy
    is_occupied[u] = False
    is_occupied[v] = True
    site_to_ion[v] = site_to_ion.pop(u)
    
    # Update active hops list (local update for efficiency)
    # This is a simplified update for the script's scope
    active_hop_indices = [h_idx for h_idx, h in enumerate(all_hops) 
                          if is_occupied[h['from']] and not is_occupied[h['to']]]

# === 5. Conductivity Calculation ===
# MSD = <|r(t) - r(0)|^2>
total_msd = sum(np.sum(d**2) for d in displacements.values())
msd_avg = total_msd / num_li_to_occupy
D = msd_avg / (6.0 * total_time) # A^2/s

# Convert D to cm^2/s
D_cm2s = D * 1e-16

# Conductivity sigma = (n * e^2 * D) / (kB * T)
# n = number of Li per volume (cm^-3)
vol_cm3 = structure.volume * 1e-24
n = num_li_to_occupy / vol_cm3
e_charge = 1.602176634e-19 # C
kB_J = 1.380649e-23 # J/K

sigma = (n * (e_charge**2) * D_cm2s) / (kB_J * T)

print(f"Simulation complete.")
print(f"Total Time: {total_time:.2e} s")
print(f"Diffusion Coefficient: {D_cm2s:.2e} cm^2/s")
print(f"Li-ion Conductivity: {sigma:.2e} S/cm")