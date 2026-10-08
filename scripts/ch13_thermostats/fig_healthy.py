"""Figure fig:th-healthy: a healthy thermostatted run.

The 256 atoms of liquid argon under CSVR (τ = 100 fs, target 135 K), 100
ps with steps of 10 fs (runs.py, csvr_100_s0). (a) The kinetic temperature
every 100 fs, with the target and the canonical spread √(2/N_f) either
side (dotted). (b) The total energy per atom and the effective energy,
the total less the heat the thermostat has added, both from their
starting values. (c) The kinetic energy over the run, against the
canonical gamma density (dashed).

Prints the numbers of Section 13.8.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch13 import N_ATOMS, N_FREE, RUNS, T_LIQUID

from mdlab import statmech, units, viz
from mdlab.viz import (
    ACCENT,
    OCHRE,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

r = np.load(RUNS / "csvr_100_s0.npz")
t = r["times"] / 1000
temp = 2 * r["kinetic"] / (N_FREE * units.KB)
energy = r["potential"] + r["kinetic"]
effective = energy - r["heat"]
spread = math.sqrt(2 / N_FREE)
late = t > 5
inside = np.mean(np.abs(temp[late] / T_LIQUID - 1) < spread)
print(f"T over 5-100 ps: mean {temp[late].mean():.2f} K, relative spread "
      f"{temp[late].std() / temp[late].mean():.4f} (canonical {spread:.4f});"
      f" within one canonical spread of the target {inside:.3f} of the time "
      f"(0.683 for a Gaussian)")
print(f"total energy: range {np.ptp(energy) * 1000 / N_ATOMS:.2f} meV per "
      f"atom over the run; effective energy: range "
      f"{np.ptp(effective) * 1000 / N_ATOMS:.4f} meV per atom, drift "
      f"{np.polyfit(r['times'], effective, 1)[0] * 1000:.1e} eV/ps")
v = r["velocities"]
p = np.einsum("i,tix->tx", r["masses"], v)
print(f"largest total momentum {np.abs(p).max():.1e} amu Å/fs")

viz.use_style(notebook=False)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(viz.FULL, 2.3),
                              gridspec_kw=dict(wspace=0.55))
a.plot(t, temp, color=ACCENT, lw=0.4)
for level in (1 - spread, 1, 1 + spread):
    a.axhline(T_LIQUID * level, **THRESHOLD_STYLE)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ / K")
viz.panel_tag(a, "a")

b.plot(t, (energy - energy[0]) * 1000 / N_ATOMS, color=OCHRE, lw=0.6,
       label=r"$E$")
b.plot(t, (effective - effective[0]) * 1000 / N_ATOMS, color=ACCENT, lw=0.8,
       label=r"$E$ $-$ heat")
b.set_xlabel("time / ps")
b.set_ylabel("change / meV per atom")
b.legend(fontsize=6, loc="lower left")
viz.panel_tag(b, "b")

k = r["kinetic"][late]
c.hist(k / k.mean(), bins=np.linspace(0.8, 1.2, 33), density=True,
       color=ACCENT, alpha=0.35)
grid = np.linspace(0.8, 1.2, 300)
c.plot(grid, 0.5 * N_FREE * statmech.kinetic_energy_density(
    grid * 0.5 * N_FREE, N_FREE, 1.0, kb=1.0), **REFERENCE_STYLE, lw=1.0)
c.set_xlabel(r"$K/\overline{K}$")
c.set_ylabel("density")
viz.panel_tag(c, "c")

print("wrote", viz.save(fig, figure_path("ch13_thermostats", "healthy.pdf")))
