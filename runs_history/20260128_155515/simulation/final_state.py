#!/usr/bin/env python3
"""
kmc_diffusion.py

A tiny kinetic‑Monte‑Carlo (KMC) driver that reads a VASP POSCAR (or any
pymatgen‑compatible structure file), builds a simple vacancy‑mediated
diffusion model on the lattice, runs the KMC steps and reports the
self‑diffusion coefficient D.  Optionally a temperature sweep can be
performed to extract an activation energy from an Arrhenius fit.

The code is deliberately simple – it is meant as a pedagogical template
that you can extend (multiple species, site‑dependent barriers,
elastic interactions, etc.).

Author:  <your‑name>
Date:    2024‑03‑08
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_structure(file_path: Path) -> Structure:
    """
    Read a crystal structure from a VASP POSCAR/CONTCAR (or any format
    supported by pymatgen) and return a pymatgen Structure object.
    """
    # Poscar can read both POSCAR and CONTCAR files
    try:
        poscar = Poscar.from_file(str(file_path))
        struct = poscar.structure
    except Exception as e:
        raise RuntimeError(f"Failed to read structure from {file_path}: {e}")

    return struct


def build_neighbor_list(struct: Structure, cutoff: float = 5.0) -> List[List[int]]:
    """
    For each atom in the structure return a list of neighbour indices.
    The neighbour list is built once (periodic images are taken into
    account) and is used throughout the KMC simulation.

    Parameters
    ----------
    struct : Structure
        The pymatgen Structure.
    cutoff : float
        Maximum distance (Å) to consider a site a neighbour.  A value of
        ~5 Å works for most close‑packed metals; increase for more open
        frameworks.

    Returns
    -------
    neighbours : List[List[int]]
        neighbours[i] is a list of site indices that atom i can jump to.
    """
    neighbours = []
    for i, site in enumerate(struct):
        # get_neighbors returns (site, distance, image) tuples
        # we only need the neighbour indices (excluding the site itself)
        neigh = struct.get_neighbors(site, r=cutoff, include_index=True)
        neigh_indices = [n[2] for n in neigh if n[2] != i]  # exclude self
        neighbours.append(neigh_indices)

    return neighbours


def kmc_run(
    struct: Structure,
    neighbours: List[List[int]],
    n_steps: int,
    temperature: float,
    barrier: float,
    attempt_freq: float = 1e13,
    seed: int = None,
) -> Tuple[float, float]:
    """
    Run a simple vacancy‑mediated KMC simulation.

    The model assumes:
      * a single vacancy that can exchange with any neighbouring atom,
      * all jumps have the same energy barrier (user‑defined),
      * the attempt frequency is the same for all jumps.

    The algorithm:
      1. pick a random atom,
      2. pick a random neighbour of that atom,
      3. compute the jump rate k = ν·exp(−E/kBT),
      4. draw a residence time Δt = −ln(r)/k,
      5. move the atom to the neighbour site,
      6. accumulate the total elapsed time.

    The mean‑square displacement (MSD) of all atoms is tracked and
    finally converted to a diffusion coefficient D = MSD/(2·d·t) where
    d = 3 is the dimensionality.

    Parameters
    ----------
    struct : Structure
        The initial crystal structure.
    neighbours : List[List[int]]
        Pre‑computed neighbour list.
    n_steps : int
        Number of KMC steps to perform.
    temperature : float
        Temperature in Kelvin.
    barrier : float
        Energy barrier for a jump (eV).
    attempt_freq : float, optional
        Attempt frequency ν (s⁻¹).  Default = 1×10¹³ s⁻¹ (typical phonon
        frequency).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    D : float
        Self‑diffusion coefficient (cm² s⁻¹).
    total_time : float
        Total simulated physical time (seconds).
    """
    if seed is not None:
        np.random.seed(seed)

    # Boltzmann constant in eV/K
    kB = 8.617333262e-5

    # Pre‑compute the Arrhenius factor (same for all jumps)
    rate = attempt_freq * np.exp(-barrier / (kB * temperature))

    # Initialise positions (in Å) and reference positions
    positions = struct.cart_coords.copy()          # shape (N, 3)
    ref_positions = positions.copy()

    # KMC loop
    elapsed_time = 0.0
    for step in range(n_steps):
        # 1) pick a random atom
        atom_idx = np.random.randint(len(struct))

        # 2) pick a random neighbour of that atom
        possible = neighbours[atom_idx]
        if not possible:
            # isolated atom – skip this step
            continue
        target_idx = np.random.choice(possible)

        # 3) draw a residence time from exponential distribution
        #    Δt = -ln(r) / rate   (r ∈ (0,1])
        r = np.random.rand()
        dt = -np.log(r) / rate
        elapsed_time += dt

        # 4) perform the jump (swap the two atoms)
        #    For a vacancy model we could keep a vacancy index, but for
        #    a simple self‑diffusion estimate swapping works fine.
        positions[[atom_idx, target_idx]] = positions[[target_idx, atom_idx]]

    # Compute mean‑square displacement (MSD) over all atoms
    displacements = positions - ref_positions
    msd = np.mean(np.sum(displacements ** 2, axis=1))   # Å²

    # Convert MSD (Å²) → cm² (1 Å = 1e‑8 cm)
    msd_cm2 = msd * 1e-16

    # Diffusion coefficient D = MSD / (2·d·t)   (d = 3)
    D = msd_cm2 / (2 * 3 * elapsed_time)   # cm² s⁻¹

    return D, elapsed_time


def run_temperature_series(
    struct: Structure,
    neighbours: List[List[int]],
    temperatures: List[float],
    barrier: float,
    n_steps: int,
    attempt_freq: float = 1e13,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the KMC simulation at a list of temperatures and return the
    diffusion coefficients.  The result can be fitted to an Arrhenius
    expression to obtain an activation energy.

    Returns
    -------
    temps_K : np.ndarray
        Temperatures (K) used.
    D_vals : np.ndarray
        Corresponding diffusion coefficients (cm² s⁻¹).
    """
    D_vals = []
    for T in temperatures:
        D, _ = kmc_run(
            struct,
            neighbours,
            n_steps=n_steps,
            temperature=T,
            barrier=barrier,
            attempt_freq=attempt_freq,
            seed=seed,
        )
        D_vals.append(D)
        print(f"  T = {T:6.1f} K  →  D = {D:.3e} cm² s⁻¹")
    return np.array(temperatures), np.array(D_vals)


def fit_arrhenius(temps_K: np.ndarray, D_vals: np.ndarray) -> Tuple[float, float]:
    """
    Fit D(T) to the Arrhenius form  D = D0·exp(−Ea/kBT)  and return
    the activation energy Ea (eV) and the prefactor D0 (cm² s⁻¹).

    Parameters
    ----------
    temps_K : np.ndarray
        Temperatures (K).
    D_vals : np.ndarray
        Diffusion coefficients (cm² s⁻¹).

    Returns
    -------
    Ea : float
        Activation energy (eV).
    D0 : float
        Prefactor (cm² s⁻¹).
    """
    # Linearise: ln D = ln D0 – Ea/(kB·T)
    kB = 8.617333262e-5  # eV/K
    x = 1.0 / temps_K
    y = np.log(D_vals)

    # Linear regression y = a + b·x   →   b = –Ea/kB
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]

    Ea = -b * kB
    D0 = np.exp(a)
    return Ea, D0


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simple Kinetic‑Monte‑Carlo diffusion calculator. "
            "Reads a POSCAR (or any pymatgen‑compatible file), builds a "
            "lattice, runs a vacancy‑mediated KMC model and reports the "
            "self‑diffusion coefficient.  Optionally performs a temperature "
            "sweep to extract an activation energy."
        )
    )
    parser.add_argument(
        "poscar",
        type=Path,
        help="Path to the POSCAR/CONTCAR file (any format readable by pymatgen).",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=300.0,
        help="Simulation temperature in K (default: 300 K).",
    )
    parser.add_argument(
        "-b",
        "--barrier",
        type=float,
        default=0.5,
        help="Diffusion energy barrier (eV) for a single hop (default: 0.5 eV).",
    )
    parser.add_argument(
        "-n",
        "--steps",
        type=int,
        default=100_000,
        help="Number of KMC steps to perform (default: 1e5).",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=5.0,
        help="Neighbour‑search cutoff distance in Å (default: 5 Å).",
    )
    parser.add_argument(
        "--temp-sweep",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional list of temperatures (K) for a sweep.  If supplied, "
            "the script runs a separate KMC simulation at each temperature "
            "and fits an Arrhenius line to obtain the activation energy."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Read the structure
    # ------------------------------------------------------------------
    try:
        struct = read_structure(args.poscar)
    except Exception as exc:
        sys.exit(f"Error reading structure: {exc}")

    print("\n=== INPUT STRUCTURE ===")
    print(struct)
    print(f"Number of sites: {len(struct)}")
    print(f"Lattice vectors (Å):\n{struct.lattice.matrix}")

    # ------------------------------------------------------------------
    # 2. Build neighbour list
    # ------------------------------------------------------------------
    neighbours = build_neighbor_list(struct, cutoff=args.cutoff)
    # sanity check: print average coordination number
    avg_coord = np.mean([len(n) for n in neighbours])
    print(f"\nNeighbour list built with cutoff = {args.cutoff:.2f} Å")
    print(f"Average coordination number per atom: {avg_coord:.2f}")

    # ------------------------------------------------------------------
    # 3. Run KMC (single temperature or sweep)
    # ------------------------------------------------------------------
    if args.temp_sweep is None:
        # ---- single‑temperature run ------------------------------------
        print("\n=== KMC SIMULATION (single temperature) ===")
        start = time.time()
        D, total_time = kmc_run(
            struct,
            neighbours,
            n_steps=args.steps,
            temperature=args.temperature,
            barrier=args.barrier,
            attempt_freq=1e13,
            seed=args.seed,
        )
        wall = time.time() - start

        print("\n--- RESULTS ---")
        print(f"Temperature          : {args.temperature:6.1f} K")
        print(f"Barrier (E)          : {args.barrier:.3f} eV")
        print(f"KMC steps            : {args.steps:,}")
        print(f"Simulated physical time : {total_time:.3e} s")
        print(f"Mean‑square displacement  : {D*2*3*total_time:.3e} Å²")
        print(f"Diffusion coefficient D    : {D:.3e} cm² s⁻¹")
        print(f"Wall‑clock time            : {wall:.2f} s")
    else:
        # ---- temperature sweep -----------------------------------------
        temps = sorted(args.temp_sweep)
        print("\n=== KMC SIMULATION (temperature sweep) ===")
        print(f"Temperatures (K): {temps}")

        T_arr, D_arr = run_temperature_series(
            struct,
            neighbours,
            temperatures=temps,
            barrier=args.barrier,
            n_steps=args.steps,
            attempt_freq=1e13,
            seed=args.seed,
        )

        # Fit Arrhenius
        Ea, D0 = fit_arrhenius(T_arr, D_arr)
        print("\n--- ARRHENIUS FIT ---")
        print(f"Fit activation energy Ea : {Ea:.3f} eV")
        print(f"Fit prefactor D0        : {D0:.3e} cm² s⁻¹")
        print("\n(If you need a more sophisticated fit, replace the "
              "linear regression with scipy.optimize.curve_fit.)")

    # ------------------------------------------------------------------
    # 4. Optional: write a tiny report file
    # ------------------------------------------------------------------
    if args.temp_sweep is None:
        out_name = "kmc_report.txt"
        with open(out_name, "w") as f:
            f.write("KMC diffusion report (single temperature)\n")
            f.write(f"Structure file   : {args.poscar}\n")
            f.write(f"Temperature (K)  : {args.temperature}\n")
            f.write(f"Barrier (eV)     : {args.barrier}\n")
            f.write(f"KMC steps        : {args.steps}\n")
            f.write(f"Physical time (s): {total_time:.3e}\n")
            f.write(f"Diffusion coeff D: {D:.3e} cm^2/s\n")
        print(f"\nReport written to {out_name}")
    else:
        out_name = "kmc_temp_sweep.txt"
        with open(out_name, "w") as f:
            f.write("KMC diffusion report (temperature sweep)\n")
            f.write(f"Structure file   : {args.poscar}\n")
            f.write(f"Barrier (eV)     : {args.barrier}\n")
            f.write(f"KMC steps per T : {args.steps}\n")
            f.write("T(K)    D(cm^2/s)\n")
            for T, D in zip(T_arr, D_arr):
                f.write(f"{T:6.1f}  {D:.3e}\n")
            f.write("\nArrhenius fit:\n")
            f.write(f"Ea (eV) : {Ea:.3f}\n")
            f.write(f"D0 (cm^2/s) : {D0:.3e}\n")
        print(f"\nTemperature‑sweep report written to {out_name}")


if __name__ == "__main__":
    main()