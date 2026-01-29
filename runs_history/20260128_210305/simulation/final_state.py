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

###############################################################################
# 2. Li sublattice identification and basic lattice info
###############################################################################

# Correct species access: use site.species (Composition-like) not site.specie
li_indices = [
    i for i, site in enumerate(structure)
    if any(sp.symbol == "Li" for sp in site.species.keys())
]
if not li_indices:
    raise RuntimeError("No Li sites found in structure; cannot run Li-ion kMC.")

li_frac_coords = np.array([structure[i].frac_coords for i in li_indices])
lattice = structure.lattice
volume = lattice.volume

print(f"Number of Li sites in supercell: {len(li_indices)}")
print(f"Lattice volume (Å^3): {volume:.2f}")

###############################################################################
# 3. Static (rigid-lattice) migration network (nearest-neighbor Li-Li graph)
###############################################################################

def minimum_image_vector(latt, f1, f2):
    """Minimum-image displacement vector (cartesian) between two fractional coords."""
    dv = f2 - f1
    dv -= np.round(dv)
    return latt.get_cartesian_coords(dv)

def minimum_image_distance(latt, f1, f2):
    """Minimum image distance between two fractional coords under PBC."""
    return np.linalg.norm(minimum_image_vector(latt, f1, f2))

# Build neighbor list with a cutoff that captures Li migration hops
LI_HOP_CUTOFF = 3.0  # Å, typical Li-Li hop distance in garnets

neighbors = [[] for _ in li_indices]
hop_vectors = {}  # (i,j) -> cartesian displacement from i to j (minimum image)

for i, idx_i in enumerate(li_indices):
    fi = structure[idx_i].frac_coords
    for j, idx_j in enumerate(li_indices):
        if j <= i:
            continue
        fj = structure[idx_j].frac_coords
        d = minimum_image_distance(lattice, fi, fj)
        if d < LI_HOP_CUTOFF:
            neighbors[i].append(j)
            neighbors[j].append(i)
            vec_ij = minimum_image_vector(lattice, fi, fj)
            vec_ji = -vec_ij
            hop_vectors[(i, j)] = vec_ij
            hop_vectors[(j, i)] = vec_ji

num_edges = sum(len(nl) for nl in neighbors) // 2
print(f"Identified {num_edges} Li-Li neighbor pairs (rigid lattice network)")

###############################################################################
# 4. Baseline (rigid-lattice) migration barrier and attempt frequency
#
# Use representative bulk c-LLZO values from context:
#   - Activation energy Ea_bulk ≈ 0.24 eV (c-LLZO, NEB/AIMD).
#   - Experiment / more complete models give ≈0.29–0.40 eV, so the rigid
#     barrier is too low and yields too high conductivity.
###############################################################################

E_MIG_RIGID = 0.24  # eV, reference saddle-point barrier for ideal, rigid host
NU_0_BASE = 1.0e13  # s^-1, typical phonon attempt frequency

###############################################################################
# 5. Lattice-relaxation penalty and soft-phonon prefactor model
###############################################################################

k_B_eV = 8.617333262145e-5  # eV/K

# Target: increase effective barrier by ~0.1–0.2 eV at 300 K to match
# experimental activation energies while allowing T-dependent softening.

DELTA_E_MEAN_300K = 0.16  # eV, mean extra penalty at 300 K
DELTA_E_STD_300K = 0.06   # eV, local variability
RELAX_PENALTY_ALPHA = 0.6  # ΔE_mean(T) ~ (300/T)^alpha

def mean_relax_penalty(T):
    """Mean extra barrier ΔE_relax(T) in eV."""
    T_eff = max(T, 150.0)
    scale = (300.0 / T_eff) ** RELAX_PENALTY_ALPHA
    return DELTA_E_MEAN_300K * scale

def std_relax_penalty(T):
    """Std-deviation of extra barrier ΔE_relax(T) in eV."""
    T_eff = max(T, 150.0)
    scale = (300.0 / T_eff) ** RELAX_PENALTY_ALPHA
    return DELTA_E_STD_300K * scale

# Prefactor softening: ν_0(T) = ν_0_BASE * S(T),
# S(T) = 1 - s0 * exp(-T/T_soft), clipped to [0.05, 1].

SOFT_MODE_S0 = 0.8
SOFT_MODE_TSOFT = 250.0  # K

def soft_mode_prefactor(T):
    """Temperature-dependent attempt frequency ν_0(T) in s^-1."""
    S = 1.0 - SOFT_MODE_S0 * np.exp(-T / SOFT_MODE_TSOFT)
    S = float(np.clip(S, 0.05, 1.0))
    return NU_0_BASE * S

###############################################################################
# 6. Configuration-dependent lattice-relaxation penalty sampling
###############################################################################

LOCAL_ENV_CUTOFF = 3.5  # Å, radius to count local Li around hop midpoint

# Precompute Li positions in cartesian
li_cart_coords = np.array([
    lattice.get_cartesian_coords(structure[i].frac_coords) for i in li_indices
])

def count_local_li(midpoint_cart):
    """Count Li ions within LOCAL_ENV_CUTOFF of a cartesian midpoint."""
    diff = li_cart_coords - midpoint_cart
    # Convert to fractional coordinates to apply minimum-image convention
    inv_lat = np.linalg.inv(lattice.matrix)
    diff_frac = diff @ inv_lat
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ lattice.matrix
    dists = np.linalg.norm(diff_cart, axis=1)
    return int(np.count_nonzero(dists < LOCAL_ENV_CUTOFF))

def estimate_reference_crowding():
    """Estimate mean local Li crowding around all unique hops."""
    counts = []
    for i, nbrs in enumerate(neighbors):
        ri = li_cart_coords[i]
        for j in nbrs:
            if j < i:
                continue
            rj = li_cart_coords[j]
            midpoint = 0.5 * (ri + rj)
            counts.append(count_local_li(midpoint))
    if not counts:
        return 0.0
    return float(np.mean(counts))

N_REF = estimate_reference_crowding()
print(f"Estimated reference local Li crowding N_ref ≈ {N_REF:.2f}")

CROWDING_BETA = 0.02  # eV penalty per excess local Li beyond N_REF

rng = np.random.default_rng(seed=42)

def sample_relax_penalty_for_hop(i, j, T):
    """
    Sample configuration- and temperature-dependent lattice relaxation
    penalty for hop i->j, in eV.
    """
    ri = li_cart_coords[i]
    rj = li_cart_coords[j]
    midpoint = 0.5 * (ri + rj)
    n_local = count_local_li(midpoint)

    mu_T = mean_relax_penalty(T)
    sigma_T = std_relax_penalty(T)

    stochastic = sigma_T * rng.normal()
    crowding_term = CROWDING_BETA * (n_local - N_REF)

    deltaE = mu_T + stochastic + crowding_term
    # Enforce non-negative penalty (relaxation does not reduce barrier here).
    return max(deltaE, 0.0)

###############################################################################
# 7. kMC hop rate calculation with dynamic barriers
###############################################################################

def hop_rate(i, j, T):
    """
    Compute hop rate (s^-1) and barrier (eV) for Li hop i->j at temperature T.
    Ea_ij(T) = E_MIG_RIGID + ΔE_relax(i->j, T)
    ν_ij(T)  = ν_0(T) * exp[-Ea_ij(T) / (k_B T)]
    """
    nu0_T = soft_mode_prefactor(T)
    deltaE = sample_relax_penalty_for_hop(i, j, T)
    Ea = E_MIG_RIGID + deltaE
    rate = nu0_T * np.exp(-Ea / (k_B_eV * T))
    # Numerical safety: avoid zero or negative rates
    rate = max(float(rate), 0.0)
    return rate, Ea

###############################################################################
# 8. Many-particle BKL kMC with dynamic barriers (occupancy + exclusion)
###############################################################################

class KMCSimulator:
    """
    Bortz–Kalos–Lebowitz (n-fold way) kMC for Li diffusion on a fixed Li
    sublattice with configuration- and T-dependent barriers that include
    lattice-relaxation penalties.
    """

    def __init__(self, T, li_indices, neighbors, hop_vectors, lattice, volume):
        self.T = float(T)
        self.li_indices = li_indices
        self.neighbors = neighbors
        self.hop_vectors = hop_vectors
        self.lattice = lattice
        self.volume = float(volume)

        # Initialize occupancy: assume sites are either Li-occupied or empty
        # according to the chemical composition embedded in structure.
        # We approximate by occupying all Li sites initially (one per site).
        # For LLZO, this slightly overestimates Li content but is consistent
        # with a dense Li sublattice for assessing dynamical effects.
        self.num_sites = len(li_indices)
        self.occupancy = np.ones(self.num_sites, dtype=int)

        # Map from site index to particle id; each occupied site gets a particle
        self.site_to_particle = {}
        self.particles = {}  # pid -> dict(start, current)
        pid = 0
        for s_idx in range(self.num_sites):
            if self.occupancy[s_idx] == 1:
                self.site_to_particle[s_idx] = pid
                coord = li_cart_coords[s_idx]
                self.particles[pid] = {
                    "start": coord.copy(),
                    "current": coord.copy()
                }
                pid += 1

        self.num_particles = pid
        self.current_time = 0.0
        self.step_count = 0

        # Preallocate rate cache: (src, dst) -> (rate, Ea)
        self.rate_cache = {}

    def _event_list(self):
        """
        Build list of all allowed hops (src -> dst) with current occupancies
        and associated rates.
        """
        events = []
        cumulative_rates = []
        total_rate = 0.0

        for src in range(self.num_sites):
            if self.occupancy[src] == 0:
                continue
            for dst in self.neighbors[src]:
                if self.occupancy[dst] != 0:
                    continue  # destination must be vacant
                key = (src, dst)
                if key in self.rate_cache:
                    rate, Ea = self.rate_cache[key]
                else:
                    rate, Ea = hop_rate(src, dst, self.T)
                    self.rate_cache[key] = (rate, Ea)
                if rate <= 0.0:
                    continue
                total_rate += rate
                events.append((src, dst))
                cumulative_rates.append(total_rate)

        return events, cumulative_rates, total_rate

    def run_step(self):
        """
        Perform a single BKL kMC step. Returns False if no events are possible.
        """
        events, cum_rates, total_rate = self._event_list()
        if total_rate <= 0.0 or not events:
            return False  # deadlocked: no allowed hops

        # Time increment
        dt = -np.log(rng.random()) / total_rate
        self.current_time += dt
        self.step_count += 1

        # Select event
        r = rng.random() * total_rate
        idx = int(np.searchsorted(cum_rates, r))
        src, dst = events[idx]

        # Perform hop: update occupancy and particle position
        pid = self.site_to_particle.pop(src)
        self.site_to_particle[dst] = pid
        self.occupancy[src] = 0
        self.occupancy[dst] = 1

        # Update particle cartesian coordinate using precomputed hop vector
        vec = self.hop_vectors[(src, dst)]
        self.particles[pid]["current"] = self.particles[pid]["current"] + vec

        # Invalidate local rate cache entries involving src or dst
        keys_to_delete = [
            key for key in self.rate_cache.keys()
            if src in key or dst in key
        ]
        for key in keys_to_delete:
            del self.rate_cache[key]

        return True

    def compute_msd_and_sigma(self):
        """
        Compute MSD and Nernst–Einstein conductivity at current time.
        """
        if self.current_time <= 0.0 or self.num_particles == 0:
            return 0.0, 0.0, 0.0

        displacements2 = []
        for pid, pdata in self.particles.items():
            dr = pdata["current"] - pdata["start"]
            # Apply minimum image to displacement for MSD
            inv_lat = np.linalg.inv(self.lattice.matrix)
            dr_frac = dr @ inv_lat
            dr_frac -= np.round(dr_frac)
            dr_cart = dr_frac @ self.lattice.matrix
            displacements2.append(np.dot(dr_cart, dr_cart))

        msd = float(np.mean(displacements2))  # Å^2

        # Diffusion coefficient: MSD = 6 D t for 3D
        D_cm2_s = msd / (6.0 * self.current_time) * 1e-16  # Å^2 -> cm^2

        # ion concentration n (ions/cm^3)
        # number of mobile ions = num_particles
        n_ions = self.num_particles / (self.volume * 1e-24)  # Å^3 -> cm^3

        e_charge = 1.602176634e-19  # C
        k_B_J = 1.380649e-23  # J/K

        sigma = (n_ions * e_charge**2 * D_cm2_s) / (k_B_J * self.T)  # S/cm

        return msd, D_cm2_s, sigma

###############################################################################
# 9. Driver: run kMC with convergence checking and output conductivity
###############################################################################

def run_kmc_simulation(T=300.0, t_max=1e-6, log_interval=100):
    """
    Run the many-particle kMC up to time t_max (s) or until convergence of σ.
    """
    sim = KMCSimulator(
        T=T,
        li_indices=li_indices,
        neighbors=neighbors,
        hop_vectors=hop_vectors,
        lattice=lattice,
        volume=volume,
    )

    sigma_history = []
    max_history = 1000
    convergence_rsd = 0.05  # 5%
    is_converged = False

    while sim.current_time < t_max:
        if not sim.run_step():
            print("Deadlock encountered; stopping kMC.")
            break

        if sim.step_count % log_interval == 0:
            msd, D, sigma = sim.compute_msd_and_sigma()
            sigma_history.append(sigma)
            if len(sigma_history) > max_history:
                sigma_history.pop(0)

            if len(sigma_history) == max_history:
                avg_sigma = float(np.mean(sigma_history))
                std_sigma = float(np.std(sigma_history))
                rsd = std_sigma / avg_sigma if avg_sigma > 0.0 else 0.0

                print(
                    f"Step {sim.step_count}: t={sim.current_time*1e9:.2f} ns, "
                    f"MSD={msd:.3f} Å^2, D={D:.3e} cm^2/s, "
                    f"sigma={sigma:.3e} S/cm, RSD={rsd*100:.2f}%"
                )

                if rsd < convergence_rsd:
                    print(
                        f"Convergence reached (RSD < {convergence_rsd*100:.1f}%) "
                        f"at t={sim.current_time*1e9:.2f} ns."
                    )
                    is_converged = True
                    break
            else:
                print(
                    f"Step {sim.step_count}: t={sim.current_time*1e9:.2f} ns, "
                    f"MSD={msd:.3f} Å^2, D={D:.3e} cm^2/s, "
                    f"sigma={sigma:.3e} S/cm"
                )

    msd, D, sigma = sim.compute_msd_and_sigma()

    return {
        "is_success": True,
        "converged": is_converged,
        "conductivity_S_per_cm": sigma,
        "diffusivity_cm2_per_s": D,
        "msd_A2": msd,
        "simulation_time_ns": sim.current_time * 1e9,
        "temperature_K": T,
        "steps": sim.step_count,
    }

###############################################################################
# 10. Main: run kMC at 300 K and save result
###############################################################################

if __name__ == "__main__":
    T_run = 300.0  # K
    t_max = 1e-6   # s (~1 microsecond upper bound)
    log_interval = 500

    try:
        result = run_kmc_simulation(T=T_run, t_max=t_max, log_interval=log_interval)
        result["error_message"] = None
        result["execution_log"] = (
            f"Completed {result['steps']} steps in "
            f"{result['simulation_time_ns']:.2f} ns at {T_run} K."
        )
    except Exception as exc:
        result = {
            "is_success": False,
            "converged": False,
            "conductivity_S_per_cm": 0.0,
            "diffusivity_cm2_per_s": 0.0,
            "msd_A2": 0.0,
            "simulation_time_ns": 0.0,
            "temperature_K": T_run,
            "steps": 0,
            "error_message": str(exc),
            "execution_log": "kMC simulation failed before completion.",
        }
        print(f"ERROR during kMC simulation: {exc}", file=sys.stderr)

    # Save result to JSON
    out_path = os.path.join(script_dir, "kmc_llzo_lattice_relaxation_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== kMC Simulation Result ===")
    print(json.dumps(result, indent=2))
    print(f"\nSaved result to: {out_path}")