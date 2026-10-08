"""Table tab:th-compare: every thermostat on the two test systems.

Run from scripts/ch13_thermostats/: python compare.py

For liquid argon (runs.py, three runs of 100 ps each unless stated): the
spread of K over its canonical value, the mean kinetic temperature with
its standard error over the runs, and the diffusion coefficient over its
value at fixed energy. For lithium on the model surface (1000 atoms at
1000 K, one thermostat each): the largest gap between the density of the
potential energy and the exact one, as a fraction of its peak, and the
hops per atom per ps. Then the cost from timing.py.
"""

import math

import numpy as np
from ch13 import (
    BARRIER,
    N_FREE,
    RUNS,
    T_LI,
    ch12,
    count_hops,
    diffusion,
    exact_energy_density,
)

from mdlab import units

CANON = math.sqrt(2 / N_FREE)
BINS = np.linspace(0, 1.125 * BARRIER, 46)
EXACT = exact_energy_density(T_LI, BINS)


def liquid(name, seeds=3):
    # Andersen and Langevin move the centre of mass: all 3N freedoms count
    n_free = 768 if name.startswith(("andersen", "langevin")) else N_FREE
    spread, temp, d = [], [], []
    for s in range(seeds):
        r = np.load(RUNS / f"{name}_s{s}.npz")
        k = r["kinetic"][50:]
        spread.append(k.std() / k.mean() / math.sqrt(2 / n_free) * CANON)
        temp.append(2 * k.mean() / (n_free * units.KB))
        d.append(diffusion(r["positions"], 500.0) * 1e4)
    err = (lambda x: np.std(x, ddof=1) / math.sqrt(seeds)) if seeds > 1 \
        else (lambda x: float("nan"))
    return np.mean(spread), np.mean(temp), err(temp), np.mean(d), err(d)


def lithium(name):
    r = np.load(RUNS / f"{name}.npz")
    hist, _ = np.histogram(r["potential"].ravel(), bins=BINS, density=True)
    gap = np.abs(hist - EXACT).max() / EXACT.max()
    path = r["positions"]
    hops = count_hops(path).mean() / ((len(path) - 1) * 10.0 / 1000)
    return gap, hops


_, _, _, d_nve, d_nve_err = liquid("nve")
rows = [
    ("fixed energy", "nve", None),
    ("rescaling", "rescale", None),
    ("Berendsen, 0.1 ps", "berendsen_100", "li_berendsen"),
    ("Berendsen, 1 ps", "berendsen_1000", None),
    ("Andersen, 0.1/ps", "andersen_0.1", None),
    ("Andersen, 1/ps", "andersen_1", "li_andersen"),
    ("Andersen, 10/ps", "andersen_10", None),
    ("Langevin, 0.1/ps", "langevin_0.1", "li_langevin_0.0001"),
    ("Langevin, 1/ps", "langevin_1", "li_langevin_0.001"),
    ("Langevin, 10/ps", "langevin_10", "li_langevin_0.01"),
    ("Nose-Hoover, 0.1 ps", "nh_100", "li_nh"),
    ("Nose-Hoover, 1 ps", "nh_1000", None),
    ("chain of 3, 0.1 ps", "nhc_100", "li_nhc"),
    ("chain of 3, 1 ps", "nhc_1000", None),
    ("CSVR, 0.1 ps", "csvr_100", "li_csvr"),
    ("CSVR, 1 ps", "csvr_1000", None),
]
print(f"canonical spread {CANON:.4f}; D at fixed energy {d_nve:.3f} e-4 Å²/fs")
for label, name, li in rows:
    seeds = 1 if name == "rescale" else 3
    spread, t, t_err, d, d_err = liquid(name, seeds)
    text = (f"{label:22s} spread/canonical {spread / CANON:5.2f}  T {t:7.2f}"
            f" ± {t_err:4.2f} K  D/D_fixed {d / d_nve:6.3f} "
            f"(± {d_err / d_nve:.3f})")
    if seeds > 1 and name != "nve":
        # standard errors of the two means combined in quadrature
        z = (d - d_nve) / math.hypot(d_err, d_nve_err)
        text += f"  {z:+5.2f} SE from fixed energy"
    if li:
        gap, hops = lithium(li)
        text += f"  Li: gap {gap:.3f}, hops {hops:.3f}/ps"
    print(text)

heat = np.load(ch12.RUNS / "heat_130.npz")
t12 = (2 * heat["kinetic"] / (N_FREE * units.KB)).mean()
d12 = diffusion(heat["positions"], 50.0) * 1e4
print(f"Chapter 12's run prepared near 130 K (fixed energy, 50 ps): T "
      f"{t12:.1f} K, D {d12:.2f} e-4 Å²/fs")

li = np.load(RUNS / "li_berendsen.npz")
site, dist = __import__("ch13").nearest_hollow(li["positions"])
late = slice(len(li["positions"]) // 2, None)
energy = (li["potential"] + li["kinetic"])[late]
print(f"Berendsen on one lithium atom, last 50 ps: distance from the hollow "
      f"{dist[late].mean():.3f} Å (spread {dist[late].std():.1e}); K/k_BT "
      f"{(li['kinetic'][late] / (units.KB * T_LI)).mean():.4f}; K + U "
      f"{energy.mean():.4f} eV")

for name in ("berendsen_100", "csvr_100", "nhc_100", "langevin_1"):
    values = []
    for s in range(3):
        r = np.load(RUNS / f"{name}_s{s}.npz")
        e = (r["potential"] + r["kinetic"])[50:]
        t = 2 * r["kinetic"][50:].mean() / (N_FREE * units.KB)
        values.append(e.var() / (units.KB * t) ** 2 / 256)
    print(f"{name}: C_V from the energy's fluctuations "
          f"{np.mean(values):.2f} N k_B (Chapter 12: 2.376)")
r = np.load(RUNS / "csvr_stride_s0.npz")
k = r["kinetic"][50:]
print(f"CSVR acting every 3 steps, tau 300 fs: spread of K "
      f"{k.std() / k.mean():.4f}; T {2 * k.mean() / (N_FREE * units.KB):.2f}"
      f" K")
r = np.load(RUNS / "csvr_wrong_s0.npz")
k = r["kinetic"][50:]
print(f"CSVR told 3N: T {2 * k.mean() / (N_FREE * units.KB):.2f} K")
