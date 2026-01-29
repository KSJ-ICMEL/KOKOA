"""
KOKOA Simulation #4
Generated: 2026-01-12 23:16:13
"""
import os, sys, traceback

# Project root (pre-calculated by Simulator)
_PROJECT_ROOT = "C:/Users/sjkim/KOKOA"
_CIF_PATH = "C:/Users/sjkim/KOKOA/Li4.47La3Zr2O12.cif"

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    os.chdir('C:/Users/sjkim/KOKOA/runs/20260112_231238')
except Exception as e:
    sys.stderr.write(f"Directory Error: {e}\n")

try:
    import sys
    import math
    import random
    import numpy as np
    from pymatgen.io.cif import CifParser
    from scipy.spatial import KDTree

    # Simulation parameters
    TARGET_TIME = 1e-08  # seconds
    NU = 1e13  # attempt frequency (s^-1)
    E_A = 0.5  # activation energy (eV)
    K_B = 8.617333262e-5  # eV/K
    Q = 1.602176634e-19  # C
    ELEM_CONV = 1e-8  # Å to cm

    def main():
        try:
            # Temperature from command line or default 300 K
            T = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0

            # Load CIF
            parser = CifParser(_CIF_PATH)
            structure = parser.get_structures()[0]
            lattice = structure.lattice
            vol = lattice.volume  # Å^3

            # Extract Li+ sites
            li_sites = [site.coords for site in structure if site.specie.symbol == 'Li']
            if not li_sites:
                print("Conductivity: 0.0 S/cm")
                return
            pos = np.array(li_sites)
            n_li = len(pos)

            # Build neighbor list (within 3 Å)
            tree = KDTree(pos)
            neigh = tree.query_ball_point(pos, 3.0)
            # Remove self
            for i, lst in enumerate(neigh):
                neigh[i] = [j for j in lst if j != i]

            # Initial positions for MSD
            init_pos = pos.copy()

            # Occupancy set
            occ = set(range(n_li))

            # Precompute rate
            rate = NU * math.exp(-E_A / (K_B * T))

            t = 0.0
            while t < TARGET_TIME:
                # Build event list
                events = []
                for i in range(n_li):
                    for j in neigh[i]:
                        if j not in occ:
                            events.append((i, j))
                if not events:
                    break
                total_events = len(events)
                sum_rates = rate * total_events
                dt = -math.log(random.random()) / sum_rates
                if t + dt > TARGET_TIME:
                    break
                # Choose event
                idx = random.randint(0, total_events - 1)
                src, tgt = events[idx]
                # Move Li+ from src to tgt
                pos[src] = pos[tgt]  # tgt is empty, its old position is free
                occ.remove(src)
                occ.add(tgt)
                t += dt

            # MSD
            msd = np.mean(np.sum((pos - init_pos)**2, axis=1))  # Å^2
            D = msd / (6 * t) if t > 0 else 0.0  # Å^2/s
            D_cm2s = D * (ELEM_CONV**2)  # cm^2/s

            # Number density (cm^-3)
            n_cm3 = n_li / (vol * 1e-24)

            sigma = n_cm3 * Q**2 * D_cm2s / (K_B * 8.617333262e-5 * T)  # S/cm
            print(f"Conductivity: {sigma} S/cm")
        except Exception as e:
            print(f"Error: {e}")

    if __name__ == "__main__":
        main()
except Exception as e:
    sys.stderr.write(f"Runtime Error: {str(e)}\n")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
