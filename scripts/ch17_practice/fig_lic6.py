"""Figure fig:ip-lic6: LiC₆ at 300 K and zero pressure with MACE-MP-0.

runs_mace.py's npt_aniso (lattice lengths free one by one, 20 ps, steps of
1 fs), its continuation npt_long (80 ps more, steps of 2 fs) and npt_iso
(the three lengths scaled together, 10 ps); stress_lic6.py's pressures.
(a) The spacing of the sheets, c, against time in both runs, with the
relaxed value at 0 K and the measured 3.70 Å (dotted). (b) The running
mean of c over the continuation from the start found, with its error
band. (c) The diagonal of the pressure tensor under each coupling, with
its standard error.

Prints the relaxed constants, the convergence report of a and c in each
stretch, and the mean pressures.
"""

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ch17 import C_LIC6, DATA, RUNS

from mdlab import viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE

relaxed = read(DATA / "structures" / "lic6.extxyz")
a0 = relaxed.cell.lengths()[0] / np.sqrt(3)
c0 = relaxed.cell.lengths()[2]
print(f"relaxed at 0 K: a = {a0:.4f} Å, c = {c0:.4f} Å")


def lengths(run, cells=3):
    """A (per hexagon side) and c from the cells of a 3×3×1 run."""
    norms = np.linalg.norm(run["cell"], axis=2)
    return norms[:, 0] / cells / np.sqrt(3), norms[:, 2]


runs = {name: np.load(RUNS / f"{name}.npz")
        for name in ("npt_aniso", "npt_iso", "npt_long")
        if (RUNS / f"{name}.npz").exists()}
for name, run in runs.items():
    a, c = lengths(run)
    dt = run["times"][1] - run["times"][0]
    for label, x in (("a", a), ("c", c)):
        rep = stats.convergence_report(x, dt)
        failed = [k for k, ok in rep.checks.items() if not ok]
        print(f"{name} {label}: {rep.mean:.4f} ± {rep.error:.4f} Å from "
              f"{rep.start / 1000:.1f} ps, {rep.effective:.0f} effective "
              f"samples; quarters " + ", ".join(
                  f"{q.mean():.4f}" for q in np.array_split(x, 4))
              + ("; passes" if rep.converged else f"; fails: {failed}"))
pressures = {}
for name in ("npt_aniso", "npt_iso"):
    p = np.load(RUNS / f"stress_{name}.npz")["pressure"]
    pressures[name] = [stats.standard_error(p[:, k])[:2] for k in range(3)]
    print(f"{name}: P_xx, P_yy, P_zz = " + ", ".join(
        f"{m:.3f} ± {e:.3f}" for m, e in pressures[name]) + " GPa")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.5))
ax = axes[0]
offset = 0.0
for name, colour, label in (("npt_aniso", ACCENT, "lengths apart"),
                            ("npt_long", ACCENT, None),
                            ("npt_iso", OCHRE, "isotropic")):
    if name not in runs:
        continue
    run = runs[name]
    _, c = lengths(run)
    shift = offset if name == "npt_long" else 0.0
    ax.plot((run["times"] + shift) / 1000, c, color=colour, lw=0.4,
            label=label)
    if name == "npt_aniso":
        offset = run["times"][-1]
ax.axhline(c0, **THRESHOLD_STYLE)
ax.axhline(C_LIC6, color=OXBLOOD, ls=":", lw=0.8)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$c$ / \AA")
ax.set_ylim(3.55, 4.04)
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "a")

ax = axes[1]
source = runs.get("npt_long", runs["npt_aniso"])
_, c = lengths(source)
dt = source["times"][1] - source["times"][0]
rep = stats.convergence_report(c, dt)
k0 = int(round(rep.start / dt))
rest = c[k0:]
n = np.arange(1, len(rest) + 1)
running = np.cumsum(rest) / n
t = (source["times"][k0:] - source["times"][k0]) / 1000
band = np.sqrt(rep.inefficiency * np.var(rest) / n)
ax.fill_between(t[10:], (running - 2 * band)[10:], (running + 2 * band)[10:],
                color=viz.tint(ACCENT, 0.3), lw=0)
ax.plot(t, running, color=ACCENT, lw=1.0)
ax.set_xlabel(r"time after discard / ps")
ax.set_ylabel(r"running mean of $c$ / \AA")
viz.panel_tag(ax, "b")

ax = axes[2]
for j, (name, colour, label) in enumerate((("npt_aniso", ACCENT,
                                            "lengths apart"),
                                           ("npt_iso", OCHRE, "isotropic"))):
    m = [x[0] for x in pressures[name]]
    e = [x[1] for x in pressures[name]]
    ax.errorbar(np.arange(3) + 0.15 * (2 * j - 1), m, e, fmt="o", ms=3,
                color=colour, label=label, capsize=0, elinewidth=0.8)
ax.axhline(0, color="black", lw=0.4)
ax.set_xticks(range(3), [r"$\mathsf{P}_{xx}$", r"$\mathsf{P}_{yy}$",
                         r"$\mathsf{P}_{zz}$"])
ax.set_ylabel("pressure / GPa")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch17_practice", "lic6.pdf")))
