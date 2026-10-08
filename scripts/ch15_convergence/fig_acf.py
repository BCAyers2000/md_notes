"""Figure fig:cv-acf: the memory of the liquid.

The 2 ns runs of liquid argon at 135 K, every 10 fs (runs.py, long_csvr
and long_nve). (a) Under CSVR with τ_T = 1 ps: the autocorrelation
functions of U, K, P and P_xy against the lag. (b) The same at fixed
energy, where U and K, whose sum is fixed, have the same function. (c)
The estimate of g for U summed up to each lag, under CSVR (teal) and
at fixed energy (ochre), each with the lag at which the rule of Section
15.2 stops (dotted).

Prints g, τ_int, the spread, the mean and its error, and the naive error,
for each quantity in each run; the temperature with its error; and the
relaxation time of Chapter 13, τ_T C_V/C_K.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, GPA, N_FREE, RUNS

from mdlab import units, viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, GREEN, OCHRE, OXBLOOD, THRESHOLD_STYLE

QUANTITIES = (("U", "potential", 1.0, ACCENT), ("K", "kinetic", 1.0, OCHRE),
              ("P", "pressure", GPA, OXBLOOD), ("P_xy", "shear", GPA, GREEN))
LAGS = 800  # 8 ps


def series(run, key, scale):
    a = run[key][:, 0] if key == "shear" else run[key]
    return a * scale


runs = {"CSVR": np.load(RUNS / "long_csvr.npz"),
        "fixed energy": np.load(RUNS / "long_nve.npz")}
acfs, cumulative = {}, {}
for label, run in runs.items():
    for name, key, scale, _ in QUANTITIES:
        a = series(run, key, scale)
        g = stats.statistical_inefficiency(a)
        mean, error, _ = stats.standard_error(a, g)
        naive = a.std(ddof=1) / math.sqrt(len(a))
        acf = stats.autocorrelation(a)
        acfs[label, name] = acf[: LAGS + 1]
        print(f"{label}, {name}: corr at 0.1, 0.2 and 1 ps "
              f"{acf[10]:.2f}, {acf[20]:.2f}, {acf[100]:.2f}")
        print(f"{label}, {name}: g {g:.1f}, τ_int {g * DT / 2:.0f} fs, "
              f"spread {a.std(ddof=1):.5f}, mean {mean:.5f} ± {error:.2g} "
              f"(naive {naive:.2g}, {error / naive:.1f} times smaller)")
        if name == "U":
            k = np.arange(1, len(acf) - 1)
            c = acf[1:-1]
            running = 1 + 2 * np.cumsum(c * (1 - k / len(a)))
            cut = np.nonzero((c <= 0) & (k > 3))[0][0]
            cumulative[label] = (running[:LAGS], cut + 1)
            print(f"    the rule stops at the lag {cut + 1}, "
                  f"{(cut + 1) * DT / 1000:.2f} ps")
    t = 2 * run["kinetic"] / (N_FREE * units.KB)
    mean_t, error_t, _ = stats.standard_error(t)
    print(f"{label}: T = {mean_t:.2f} ± {error_t:.2f} K over "
          f"{run['times'][-1] / 1e6:.0f} ns")
total = runs["CSVR"]["potential"] + runs["CSVR"]["kinetic"]
print(f"Chapter 13's relaxation time τ_T C_V/C_K = 1.59 ps; here τ_int of "
      f"K + U under CSVR is "
      f"{stats.statistical_inefficiency(total) * DT / 2 / 1000:.2f} ps")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.45))
lag_ps = np.arange(LAGS + 1) * DT / 1000
for ax, label, tag in zip(axes[:2], runs, "ab", strict=True):
    for name, _, _, colour in QUANTITIES:
        sub = name.replace("P_xy", r"\mathsf{P}_{xy}")
        ax.plot(lag_ps, acfs[label, name], color=colour, lw=0.9,
                label=f"${sub}$")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.15, 1.02)
    ax.set_xlabel("lag / ps")
    ax.set_ylabel(r"$\mathrm{corr}$")
    viz.panel_tag(ax, tag)
axes[0].legend(fontsize=7, loc="upper right", ncols=2)

ax = axes[2]
for label, colour in (("CSVR", ACCENT), ("fixed energy", OCHRE)):
    running, cut = cumulative[label]
    ax.plot(lag_ps[1:], running, color=colour, lw=0.9)
    ax.axvline(cut * DT / 1000, **THRESHOLD_STYLE)
ax.set_xlim(0, 8)
ax.set_xlabel("last lag summed / ps")
ax.set_ylabel(r"$g$ of $U$")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence", "acf.pdf")))
