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

# ==============================================================
# 2. Global Parameters and Physical Constants
# ==============================================================

k_B = 8.617333262e-5          # eV/K
e_charge = 1.602176634e-19    # C
coulomb_const_eV_A = 14.3996  # eV·Å (e^2/(4πϵ0) in eV·Å)
temperature = 300.0           # K
attempt_freq = 1e13           # Hz
base_barrier = 0.30           # eV, intrinsic barrier for Li hop
min_barrier = 0.05            # eV, lower bound to avoid negative rates
dielectric_const = 10.0       # Relative dielectric constant (screening)
neighbor_cutoff = 4.5          # Å, distance to consider a hop possible
coulomb_cutoff = 10.0          # Å, cutoff for Coulomb interactions
mc_steps = 20000              # Metropolis MC steps for equilibration
kmc_steps = 200000            # Number of kMC events
record_interval = 1000        # Steps between MSD recordings

# ==============================================================
# 3. Identify Li Sites and Build Neighbor List
# ==============================================================

# Map global site indices to local Li indices
li_global_indices = [i for i, site in enumerate(structure.sites) if site.species_string == "Li"]
num_li_sites = len(li_global_indices)
global_to_local = {g: l for l, g in enumerate(li_global_indices)}
local_to_global = {l: g for g, l in global_to_local.items()}

# Fractional and Cartesian coordinates of Li sites
li_frac_coords = np.array([structure.sites[g].frac_coords for g in li_global_indices])
li_cart_coords = np.array([structure.sites[g].coords for g in li_global_indices])

# Build adjacency list of possible hops (neighbors within cutoff)
adjacency = {i: [] for i in range(num_li_sites)}
for local_i, global_i in enumerate(li_global_indices):
    site_i = structure.sites[global_i]
    neighbors = structure.get_neighbors(site_i, neighbor_cutoff, include_index=True)
    for neighbor_site, distance, image, neighbor_index in neighbors:
        if neighbor_site.species_string != "Li":
            continue
        global_j = neighbor_index
        if global_j not in global_to_local:
            continue
        local_j = global_to_local[global_j]
        if local_j == local_i:
            continue
        adjacency[local_i].append(local_j)

# ==============================================================
# 4. Initialize Occupancy (Li/Vacancy Distribution)
# ==============================================================

# Total number of Li atoms in the supercell (from composition)
total_li_atoms = int(round(structure.composition.get_element_count("Li")))
if total_li_atoms > num_li_sites:
    raise ValueError("More Li atoms than Li sites; check the structure or supercell size.")

# Randomly occupy exactly total_li_atoms sites
occupied = np.zeros(num_li_sites, dtype=bool)
initial_occupied_indices = np.random.choice(num_li_sites, size=total_li_atoms, replace=False)
occupied[initial_occupied_indices] = True

# Assign unique ion IDs to occupied sites
site_to_ion = {}   # local site index -> ion ID
ion_to_site = {}   # ion ID -> local site index
for ion_id, site_idx in enumerate(np.where(occupied)[0]):
    site_to_ion[site_idx] = ion_id
    ion_to_site[ion_id] = site_idx

# Store initial positions for MSD tracking (fractional)
ion_initial_frac = {ion_id: li_frac_coords[site_idx].copy() for ion_id, site_idx in ion_to_site.items()}

# ==============================================================
# 5. Helper Functions
# ==============================================================

def compute_coulomb_energies(occupied_mask):
    """Compute site-specific Coulomb repulsion energies (eV) for all Li sites."""
    # Positions of occupied sites (fractional)
    occ_indices = np.where(occupied_mask)[0]
    if len(occ_indices) == 0:
        return np.zeros(num_li_sites)
    occ_frac = li_frac_coords[occ_indices]  # shape (Nocc, 3)

    # Compute pairwise distance matrix using minimum image convention
    diff = occ_frac[:, np.newaxis, :] - occ_frac[np.newaxis, :, :]  # (Nocc, Nocc, 3)
    diff -= np.rint(diff)  # wrap into [-0.5, 0.5]
    cart_diff = np.tensordot(diff, structure.lattice.matrix, axes=([2], [0]))  # (Nocc, Nocc, 3)
    distances = np.linalg.norm(cart_diff, axis=2)  # (Nocc, Nocc)

    # Mask out self-interactions and distances beyond cutoff
    mask = (distances > 1e-8) & (distances <= coulomb_cutoff)
    # Coulomb energy per pair: C/(ε * r)
    pair_energies = np.zeros_like(distances)
    pair_energies[mask] = coulomb_const_eV_A / (dielectric_const * distances[mask])

    # Site energies: sum over columns (or rows) of pair_energies
    site_energies_occ = np.sum(pair_energies, axis=1)  # shape (Nocc,)

    # Map back to full site list
    site_energies = np.zeros(num_li_sites)
    site_energies[occ_indices] = site_energies_occ
    return site_energies

def compute_hopping_rates(occupied_mask, site_energies):
    """Return list of possible hops and their rates."""
    hops = []   # each entry: (i_local, j_local, rate)
    for i in np.where(occupied_mask)[0]:
        for j in adjacency[i]:
            if not occupied_mask[j]:  # vacancy
                # Barrier = base + 0.5*(E_j - E_i)
                delta_E = site_energies[j] - site_energies[i]
                barrier = base_barrier + 0.5 * delta_E
                if barrier < min_barrier:
                    barrier = min_barrier
                rate = attempt_freq * np.exp(-barrier / (k_B * temperature))
                hops.append((i, j, rate))
    return hops

def frac_diff(frac1, frac2):
    """Minimum-image fractional difference."""
    diff = frac2 - frac1
    diff -= np.rint(diff)  # wrap into [-0.5, 0.5]
    return diff

def frac_to_cart(frac_vec):
    """Convert fractional vector to Cartesian using lattice."""
    return np.dot(frac_vec, structure.lattice.matrix)

def compute_msd(ion_positions_frac, ion_initial_frac):
    """Mean squared displacement (Å^2) over all ions."""
    displacements = []
    for ion_id, cur_frac in ion_positions_frac.items():
        init_frac = ion_initial_frac[ion_id]
        d_frac = frac_diff(init_frac, cur_frac)
        d_cart = frac_to_cart(d_frac)
        displacements.append(np.dot(d_cart, d_cart))
    return np.mean(displacements) if displacements else 0.0

def compute_net_disp_sq(ion_positions_frac, ion_initial_frac):
    """Squared magnitude of net charge displacement vector (Å^2)."""
    net_disp = np.zeros(3)
    for ion_id, cur_frac in ion_positions_frac.items():
        init_frac = ion_initial_frac[ion_id]
        d_frac = frac_diff(init_frac, cur_frac)
        net_disp += frac_to_cart(d_frac)
    return np.dot(net_disp, net_disp)

# ==============================================================
# 6. Metropolis Monte Carlo Equilibration of Li/Vacancy Distribution
# ==============================================================

print("Starting Metropolis MC equilibration...")
for step in range(mc_steps):
    # Randomly pick an occupied site i and a vacant site j
    occ_sites = np.where(occupied)[0]
    vac_sites = np.where(~occupied)[0]
    if len(occ_sites) == 0 or len(vac_sites) == 0:
        break
    i = np.random.choice(occ_sites)
    j = np.random.choice(vac_sites)

    # Propose swap: i becomes vacant, j becomes occupied
    occupied_trial = occupied.copy()
    occupied_trial[i] = False
    occupied_trial[j] = True

    # Compute energies before and after
    E_before = compute_coulomb_energies(occupied)
    E_after = compute_coulomb_energies(occupied_trial)
    total_E_before = 0.5 * np.sum(E_before)
    total_E_after = 0.5 * np.sum(E_after)
    dE = total_E_after - total_E_before

    # Metropolis acceptance
    if dE <= 0.0 or np.random.rand() < np.exp(-dE / (k_B * temperature)):
        # Accept move
        occupied = occupied_trial
        # Update ion mappings
        ion_id = site_to_ion.pop(i)
        site_to_ion[j] = ion_id
        ion_to_site[ion_id] = j
        # Update initial positions for the moved ion (keep same initial frac)
        # No need to change ion_initial_frac because it tracks the ion, not the site
        # (the ion moved, its initial position remains the same)
        # (Optionally could update if you want to reset after equilibration)
    # else reject (do nothing)

print("Metropolis MC equilibration completed.")

# ==============================================================
# 7. Prepare for Kinetic Monte Carlo
# ==============================================================

# Compute initial site energies
site_energies = compute_coulomb_energies(occupied)

# Build initial list of hops and rates
hops = compute_hopping_rates(occupied, site_energies)
if not hops:
    raise RuntimeError("No possible hops found; check neighbor cutoff or occupancy.")

# Precompute cumulative rates for efficient selection
def build_cumulative_rates(hops):
    rates = np.array([h[2] for h in hops])
    cum_rates = np.cumsum(rates)
    total = cum_rates[-1]
    return cum_rates, total

cum_rates, total_rate = build_cumulative_rates(hops)

# Tracking variables
time = 0.0
msd_record = []
time_record = []
net_disp_sq_record = []

# Current fractional positions of each ion (updated after each hop)
ion_positions_frac = {ion_id: li_frac_coords[site_idx].copy() for ion_id, site_idx in ion_to_site.items()}

print("Starting kinetic Monte Carlo simulation...")
for step in range(1, kmc_steps + 1):
    if total_rate <= 0.0:
        print("Total rate zero; terminating kMC.")
        break

    # Time increment
    r1 = np.random.rand()
    dt = -np.log(r1) / total_rate
    time += dt

    # Choose which hop occurs
    r2 = np.random.rand() * total_rate
    hop_index = np.searchsorted(cum_rates, r2)
    i, j, rate_ij = hops[hop_index]

    # Execute hop: move ion from i to j
    ion_id = site_to_ion.pop(i)
    site_to_ion[j] = ion_id
    ion_to_site[ion_id] = j
    occupied[i] = False
    occupied[j] = True

    # Update ion's fractional position
    ion_positions_frac[ion_id] = li_frac_coords[j].copy()

    # Update site energies locally (recompute all for simplicity)
    site_energies = compute_coulomb_energies(occupied)

    # Rebuild hops and cumulative rates (could be optimized to local updates)
    hops = compute_hopping_rates(occupied, site_energies)
    if not hops:
        print("No hops left after step {}. Terminating.".format(step))
        break
    cum_rates, total_rate = build_cumulative_rates(hops)

    # Record observables at intervals
    if step % record_interval == 0 or step == kmc_steps:
        msd = compute_msd(ion_positions_frac, ion_initial_frac)  # Å^2
        net_disp_sq = compute_net_disp_sq(ion_positions_frac, ion_initial_frac)  # Å^2
        msd_record.append(msd)
        net_disp_sq_record.append(net_disp_sq)
        time_record.append(time)

# ==============================================================
# 8. Post‑Processing: Diffusion Coefficients, Haven Ratio, Conductivity
# ==============================================================

if time <= 0.0:
    raise RuntimeError("Simulation time is zero; cannot compute diffusion coefficients.")

# Average MSD over the whole trajectory (last recorded value)
final_msd = msd_record[-1] if msd_record else compute_msd(ion_positions_frac, ion_initial_frac)
final_net_disp_sq = net_disp_sq_record[-1] if net_disp_sq_record else compute_net_disp_sq(ion_positions_frac, ion_initial_frac)

D_tracer = final_msd / (6.0 * time)          # Å^2 / s → convert to cm^2/s later
D_charge = final_net_disp_sq / (6.0 * time)  # Å^2 / s

# Convert diffusion from Å^2/s to cm^2/s
angstrom_to_cm = 1e-8
D_tracer_cm2 = D_tracer * (angstrom_to_cm ** 2)
D_charge_cm2 = D_charge * (angstrom_to_cm ** 2)

# Haven ratio
haven_ratio = D_tracer / D_charge if D_charge > 0 else np.nan

# Number density of Li ions (m⁻³)
volume_A3 = structure.lattice.volume  # Å³ of the supercell
volume_cm3 = volume_A3 * 1e-24        # convert Å³ → cm³
n_li = total_li_atoms / volume_cm3    # ions per cm³

# Conductivity (S/m)
sigma = n_li * (e_charge ** 2) * D_charge_cm2 / (k_B * temperature)  # S·m⁻¹

# ==============================================================
# 9. Output Results
# ==============================================================

results = {
    "temperature_K": temperature,
    "total_time_s": time,
    "total_Li_ions": total_li_atoms,
    "number_density_m3": n_li,
    "D_tracer_cm2_s": D_tracer_cm2,
    "D_charge_cm2_s": D_charge_cm2,
    "haven_ratio": haven_ratio,
    "conductivity_S_m": sigma,
    "msd_record_A2": msd_record,
    "time_record_s": time_record,
    "net_disp_sq_record_A2": net_disp_sq_record
}

print("\n=== Simulation Summary ===")
print(f"Total simulation time: {time:.4e} s")
print(f"Tracer diffusion coefficient D_tracer: {D_tracer_cm2:.4e} cm^2/s")
print(f"Charge diffusion coefficient D_charge: {D_charge_cm2:.4e} cm^2/s")
print(f"Haven ratio (D_tracer/D_charge): {haven_ratio:.4f}")
print(f"Ionic conductivity σ: {sigma:.4e} S/m")

# Optionally save results to JSON
output_path = "kmc_llzo_results.json"
with open(output_path, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"Results saved to {output_path}")