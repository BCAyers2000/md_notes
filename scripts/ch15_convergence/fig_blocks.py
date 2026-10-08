"""Figure fig:cv-blocks: block averages.

(a) The series of Section 15.2 (c = 0.95, g = 39): the error of the mean
from blocks of b samples, for 2¹⁷ samples (teal) and 2¹⁰ (ochre), each
with the error of its own estimate, against the exact errors (dashed).
(b) The liquid under CSVR, 2 ns (runs.py, long_csvr): the error of the
mean of U, K and P from blocks of growing length, divided by √(g s²/n).
(c) The 2 ns run cut into twenty runs of 100 ps: the mean of U in each
with its own error bar, √(g s²/n) from that piece alone, against the mean
of the whole run (dashed), in meV from it.

Prints the plateau values, the shortfall of short blocks against g/4b,
and, for U, K and P, the spread of the twenty means against the average
error bar of one piece and how many bars hold the mean of the whole.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, GPA, RUNS, ou

from mdlab import viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

C = 0.95
G = (1 + C) / (1 - C)
rng = np.random.default_rng(2)


def g_finite(n):
    k = np.arange(1, n)
    return 1 + 2 * np.sum((1 - k / n) * C**k)


synthetic = {}
for n, colour in ((2**17, ACCENT), (2**10, OCHRE)):
    blocks = stats.block_average(ou(n, C, rng))
    exact = math.sqrt(g_finite(n) / n)
    synthetic[n] = (blocks, exact, colour)
    print(f"{n} samples: exact error {exact:.4f}; from blocks of "
          + ", ".join(f"{b} {e:.4f}" for b, e in
                      zip(blocks["length"], blocks["error"], strict=True)))
long_blocks, long_exact, _ = synthetic[2**17]
for b in (100, 195, 400, 1000):
    print(f"  b = {b}: exact shortfall of the error, 1 − √(g_b/g), "
          f"{1 - math.sqrt(g_finite(b) / G):.3f}; g/4b {G / (4 * b):.3f}")

run = np.load(RUNS / "long_csvr.npz")
quantities = {"U": run["potential"], "K": run["kinetic"],
              "P": run["pressure"] * GPA}
liquid = {}
for name, a in quantities.items():
    mean, error, g = stats.standard_error(a)
    acf = stats.autocorrelation(a)
    lag = np.arange(1, len(acf) - 1)
    stop = np.nonzero((acf[1:-1] <= 0) & (lag > 3))[0][0]
    corr, lag = acf[1:stop + 1], lag[:stop]
    b = 5 * g
    g_b = 1 + 2 * np.sum(np.where(lag < b, (1 - lag / b) * corr, 0.0))
    print(f"{name}: blocks of 5g = {b:.0f} samples miss "
          f"{1 - math.sqrt(g_b / g):.3f} of the error, from the run's own "
          f"autocorrelation; Σk corr(k) / (g/2)² = "
          f"{np.sum(lag * corr) / (g / 2) ** 2:.2f}")
    blocks = stats.block_average(a, min_blocks=8)
    liquid[name] = (blocks, error)
    top = blocks["error"][-4:] / error
    print(f"{name}: error {error:.3g}; blocks of the four longest lengths "
          f"{blocks['length'][-4:] * DT / 1000} ps give "
          f"{np.round(top, 2)} of it")

pieces = 20
segment = len(run["potential"]) // pieces
counts = {}
for name, a in quantities.items():
    whole = a.mean()
    means, errors, gs = [], [], []
    for j in range(pieces):
        part = a[j * segment:(j + 1) * segment]
        m, e, g_piece = stats.standard_error(part)
        means.append(m)
        errors.append(e)
        gs.append(g_piece)
    g_whole = stats.statistical_inefficiency(a)
    print(f"{name}: g from the whole run {g_whole:.0f}, from one piece "
          f"{np.mean(gs):.0f} on average (range "
          f"{min(gs):.0f} to {max(gs):.0f}); effective samples in a piece "
          f"{segment / g_whole:.0f}")
    means, errors = np.array(means), np.array(errors)
    inside = np.sum(np.abs(means - whole) <= errors)
    print(f"{name}: the twenty error bars spread by "
          f"{errors.std(ddof=1) / errors.mean():.2f} of their mean")
    counts[name] = (means, errors, whole)
    print(f"{name}, twenty pieces of 100 ps: spread of the means "
          f"{means.std(ddof=1):.3g}, average error bar of one piece "
          f"{errors.mean():.3g} (range {errors.min():.3g} to "
          f"{errors.max():.3g}); {inside} of 20 bars hold the whole run's "
          f"mean {whole:.5g}; correlation of successive means "
          f"{np.corrcoef(means[:-1], means[1:])[0, 1]:+.2f}")

viz.use_style(notebook=False)
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(viz.FULL, 2.0), gridspec_kw=dict(wspace=0.55))
for blocks, exact, colour in synthetic.values():
    ax_a.errorbar(blocks["length"], blocks["error"],
                  yerr=blocks["error_error"], fmt="o", ms=2.5, lw=0.8,
                  color=colour, capsize=0)
    ax_a.axhline(exact, **REFERENCE_STYLE, lw=0.8)
ax_a.set_xscale("log", base=2)
ax_a.set_xticks([2**0, 2**5, 2**10, 2**15])
ax_a.set_yscale("log")
ax_a.set_xlabel(r"block length $b$ / samples")
ax_a.set_ylabel("error of the mean")
viz.panel_tag(ax_a, "a")

for (name, (blocks, error)), colour in zip(liquid.items(),
                                           (ACCENT, OCHRE, OXBLOOD),
                                           strict=True):
    ax_b.errorbar(blocks["length"] * DT / 1000, blocks["error"] / error,
                  yerr=blocks["error_error"] / error, fmt="o-", ms=2.5,
                  lw=0.7, color=colour, capsize=0, label=f"${name}$")
ax_b.axhline(1, **REFERENCE_STYLE, lw=0.8)
ax_b.set_xscale("log")
ax_b.set_xlabel("block length / ps")
ax_b.set_ylabel(r"ratio to the error from $g$")
ax_b.legend(fontsize=7, loc="lower right")
viz.panel_tag(ax_b, "b")

means, errors, whole = counts["U"]
ax_c.errorbar(np.arange(1, pieces + 1), 1000 * (means - whole),
              yerr=1000 * errors, fmt="o", ms=2.5, lw=0.8, color=ACCENT,
              capsize=0)
ax_c.axhline(0, **REFERENCE_STYLE, lw=0.8)
ax_c.set_xlabel("piece of 100 ps")
ax_c.set_ylabel(r"$\bar U-\bar U_{2\,\mathrm{ns}}$ / meV")
viz.panel_tag(ax_c, "c")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "blocks.pdf")))
