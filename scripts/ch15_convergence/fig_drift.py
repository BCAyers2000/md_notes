"""Figure fig:cv-drift: a slow drift.

The liquid at fixed energy from the third start of Chapter 13, 1 ns at
each step δt (runs.py, step_*). (a) The total energy per atom, less its
starting value, for δt = 20, 30, 35 and 40 fs; the run at 40 fs blew up
at 557 ps. (b) The drift per atom per ns from the least-squares slope,
with its error, against δt, and the largest drift that moves the mean
temperature of a 1 ns run by less than its own error (dotted). (c) The
spread of the total energy over that of the kinetic energy against δt,
against δt² through the point at 5 fs (dashed).

Prints the tyre of the toolbox; how often the test finds a drift in
series with none; for each δt the drift, its error and their ratio, the
fluctuation ratio and its growth from the step before, and the drift of
the temperature it implies; and the limit, from the heat capacity of
Chapter 12 and the error of T in 1 ns at fixed energy.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch15 import N_FREE, RUNS, ou

from mdlab import units, viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE, THRESHOLD_STYLE

rng = np.random.default_rng(4)
days = np.arange(30.0)
readings = 2.5 - 0.01 * days + 0.05 * rng.standard_normal(30)
slope, error, ratio = stats.drift_test(days, readings, g=1.0)
misses = readings - readings.mean() - slope * (days - days.mean())
print(f"tyre: the misses about the fitted line scatter by "
      f"{math.sqrt(np.sum(misses**2) / 28):.3f} bar (n − 2 = 28)")
s_tt = np.sum((days - days.mean()) ** 2)
print(f"tyre: 30 daily readings, noise 0.05 bar, leak 0.01 bar/day: slope "
      f"{slope:.4f} ± {error:.4f} bar/day ({ratio:.1f} errors); Σ(t − t̄)² "
      f"= {s_tt:.1f} day², 0.05/√ of it = {0.05 / math.sqrt(s_tt):.5f}")

alarms = []
for _ in range(1000):
    x = ou(5000, 0.95, rng)
    alarms.append(abs(stats.drift_test(np.arange(5000.0), x)[2]) >= 2)
print(f"series of 5000 with g = 39 and no drift: the test finds a drift in "
      f"{np.mean(alarms):.3f} of 1000")

C_V = 2.347 * units.KB  # per atom, Chapter 12
nve = np.load(RUNS / "long_nve.npz")
t_series = 2 * nve["kinetic"] / (N_FREE * units.KB)
error_2ns = stats.standard_error(t_series)[1]
error_1ns = error_2ns * math.sqrt(2)
limit = 2 * error_1ns * C_V / 1.0  # eV per atom per ns, for a 1 ns run
print(f"error of T at fixed energy: {error_2ns:.4f} K in 2 ns, so "
      f"{error_1ns:.4f} K in 1 ns; with C_V/N = 2.347 k_B, a drift moves the "
      f"mean of a 1 ns run by less than that if below "
      f"{1e6 * limit:.1f} μeV per atom per ns")

steps = (5, 10, 20, 30, 35, 40)
rows, energies = [], {}
previous = None
for dt in steps:
    r = np.load(RUNS / f"step_{dt}.npz")
    t, e, k = r["times"], r["potential"] + r["kinetic"], r["kinetic"]
    finite = np.isfinite(e) & (np.abs(e - e[0]) < 1.0)
    end = len(e) if finite.all() else int(np.argmax(~finite))
    t, e, k = t[:end], e[:end], k[:end]
    every = max(1, len(t) // 20000)
    b, err, rat = stats.drift_test(t[::every], e[::every])
    per_atom = b * 1e6 / 256  # eV per atom per ns
    per_error = err * 1e6 / 256
    fluct = np.std(e) / np.std(k)
    growth = "" if previous is None else (
        f", {fluct / previous[1]:.2f} times that at {previous[0]} fs "
        f"(δt² gives {(dt / previous[0]) ** 2:.2f})")
    previous = (dt, fluct)
    rows.append((dt, per_atom, per_error, fluct))
    energies[dt] = (t, (e - e[0]) / 256)
    print(f"δt = {dt} fs, {t[-1] / 1000:.0f} ps: drift "
          f"{1e6 * per_atom:+.2f} ± {1e6 * per_error:.2f} μeV per atom per ns"
          f" ({rat:+.1f} errors), temperature {per_atom / C_V:+.4f} K per ns;"
          f" fluctuation ratio {fluct:.2e}{growth}; T "
          f"{2 * k.mean() / (N_FREE * units.KB):.2f} K")
    if not np.isfinite(r["potential"]).all():
        bad = int(np.argmax(~np.isfinite(r["potential"])))
        print(f"    the energy is no longer finite from "
              f"{r['times'][bad] / 1000:.0f} ps")

viz.use_style(notebook=False)
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(viz.FULL, 2.0), gridspec_kw=dict(wspace=0.55))
for dt, colour in ((20, ACCENT), (30, OCHRE), (35, OXBLOOD),
                   (40, REFERENCE)):
    t, de = energies[dt]
    ax_a.plot(t / 1000, 1000 * de, color=colour, lw=0.6, label=f"{dt} fs")
ax_a.set_ylim(-3, 6)
ax_a.set_xlabel("time / ps")
ax_a.set_ylabel(r"$(E - E_0)/N$ / meV")
ax_a.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax_a, "a")

dts = np.array([row[0] for row in rows], float)
drift = np.array([abs(row[1]) for row in rows]) * 1e6
drift_err = np.array([row[2] for row in rows]) * 1e6
ax_b.errorbar(dts, drift, yerr=drift_err, fmt="o", ms=3, color=ACCENT,
              lw=0.8, capsize=0)
ax_b.axhline(1e6 * limit, **THRESHOLD_STYLE)
ax_b.set_yscale("log")
ax_b.set_ylim(1e-3, 3e4)
ax_b.set_xlabel(r"$\delta t$ / fs")
ax_b.set_ylabel(r"$|\dot E|/N$ / $\mu$eV ns$^{-1}$")
viz.panel_tag(ax_b, "b")

fluct = np.array([row[3] for row in rows])
ax_c.loglog(dts, fluct, "o", ms=3, color=ACCENT)
fine = np.geomspace(5, 40, 50)
ax_c.loglog(fine, fluct[0] * (fine / 5) ** 2, color=REFERENCE, ls="--",
            lw=0.8)
ax_c.set_xticks([5, 10, 20, 40])
ax_c.set_xticks([], minor=True)
ax_c.xaxis.set_major_formatter(plt.ScalarFormatter())
ax_c.set_xlabel(r"$\delta t$ / fs")
ax_c.set_ylabel(r"$\sigma_E/\sigma_K$")
viz.panel_tag(ax_c, "c")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "drift.pdf")))
