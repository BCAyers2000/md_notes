"""Figure fig:sm-heat: the heat capacity of liquid argon, two ways.

Five runs of 256 atoms at fixed energy, prepared near 100, 115, 130, 145
and 160 K (runs.py), 50 ps each. (a) The mean total energy against the
mean kinetic temperature, with the straight line fitted through them,
whose slope is the heat capacity C_V. (b) The heat capacity per atom,
in units of k_B, from the spread of the kinetic energy of each run by the
formula of Lebowitz, Percus and Verlet (circles), and of the four runs of
different sizes near 130 K (squares), against the slope of (a) (dashed).

Prints the numbers of Section 12.7.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch12 import RUNS

from mdlab import statmech, units, viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, figure_path

N = 256
n_free = statmech.degrees_of_freedom(N)
temps, energies, from_spread = [], [], []
for t in (100, 115, 130, 145, 160):
    run = np.load(RUNS / f"heat_{t}.npz")
    k, u = run["kinetic"], run["potential"]
    temps.append((2 * k / (n_free * units.KB)).mean())
    energies.append((k + u).mean())
    from_spread.append(statmech.heat_capacity_nve(k, n_free) / (N * units.KB))
    print(f"prepared near {t} K: T = {temps[-1]:.2f} K, E = "
          f"{energies[-1]:.4f} eV, C_V from the spread "
          f"{from_spread[-1]:.3f} N k_B")
temps, energies = np.array(temps), np.array(energies)
slope, intercept = np.polyfit(temps, energies, 1)
cv_slope = slope / (N * units.KB)
residual = energies - (slope * temps + intercept)
print(f"slope dE/dT = {slope:.5f} eV/K = {cv_slope:.3f} N k_B; largest "
      f"distance of a point from the line {np.abs(residual).max():.4f} eV")
print(f"from the spread: mean {np.mean(from_spread):.3f}, range "
      f"{min(from_spread):.3f} to {max(from_spread):.3f} N k_B")
sizes = (108, 256, 500, 864)
size_cv = []
for n in sizes:
    k = np.load(RUNS / f"size_{n}.npz")["kinetic"]
    size_cv.append(statmech.heat_capacity_nve(
        k, statmech.degrees_of_freedom(n)) / (n * units.KB))
    print(f"N = {n}: C_V from the spread {size_cv[-1]:.3f} N k_B")
print("an ideal gas would have 1.5 N k_B, a harmonic crystal 3 N k_B")
every = np.array(from_spread + size_cv)
print(f"largest departure of a single-run estimate from the slope "
      f"{np.abs(every / cv_slope - 1).max():.3f}")
print(f"canonical spread of the kinetic temperature at 135 K: "
      f"{np.sqrt(2 / n_free) * 135:.1f} K")
c_kin = n_free / (2 * N)  # the kinetic part, N_f k_B/2, per N k_B
print(f"spread at fixed energy against canonical, predicted from the slope "
      f"for N = {N}: sqrt(1 - {c_kin:.4f}/{cv_slope:.3f}) = "
      f"{np.sqrt(1 - c_kin / cv_slope):.3f}")

viz.use_style(notebook=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.4),
                           gridspec_kw=dict(wspace=0.4))
a.plot(temps, energies, "o", color=ACCENT, ms=4)
tt = np.array([95, 165])
a.plot(tt, slope * tt + intercept, **REFERENCE_STYLE, lw=1.0)
a.set_xlabel(r"$T$ / K")
a.set_ylabel(r"$E$ / eV")
viz.panel_tag(a, "a")
b.plot(temps, from_spread, "o", color=ACCENT, ms=4, label="five energies")
size_temps = [(2 * np.load(RUNS / f"size_{n}.npz")["kinetic"]
               / (statmech.degrees_of_freedom(n) * units.KB)).mean()
              for n in sizes]
b.plot(size_temps, size_cv, "s", color=OCHRE, ms=3.5, label="four sizes")
b.axhline(cv_slope, **REFERENCE_STYLE, lw=1.0)
b.set_xlabel(r"$T$ / K")
b.set_ylabel(r"$C_V / Nk_\mathrm{B}$")
b.set_ylim(1.5, 3.0)
b.legend(fontsize=7, loc="lower right")
viz.panel_tag(b, "b")

print("wrote", viz.save(fig, figure_path("ch12_ensembles", "heat.pdf")))
