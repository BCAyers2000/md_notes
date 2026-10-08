"""Figure fig:cv-ensemble: whether a run samples its ensemble.

The liquid at 130 and 140 K, 1 ns each, under CSVR (τ_T = 1 ps) and
Berendsen coupling (τ_T = 0.1 ps) (runs.py, ensemble_*), each after the
start found from its total energy. (a) The densities of the total energy
K + U at the two temperatures under each. (b) The logarithm of the ratio
of the two densities, ln[P₁₄₀(E)/P₁₃₀(E)], against E, with the slope
β₁₃₀ − β₁₄₀ that the canonical ensemble requires (dashed, through each
set's middle).

Prints the slope of the ratio and its error from the block bootstrap for
each thermostat, against the required slope; the same test on energies
drawn from the exact gamma density at the two temperatures; and the
bootstrap's error of a mean against √(g s²/n) on the 2 ns run.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch15 import RUNS

from mdlab import units, viz
from mdlab.analysis import stats
from mdlab.viz import REFERENCE_STYLE, THERMOSTAT

KB = units.KB
TEMPS = (130.0, 140.0)
REQUIRED = 1 / (KB * TEMPS[0]) - 1 / (KB * TEMPS[1])
BINS = 40


def ratio_points(e1, e2):
    """Bin centres, ln of the ratio of the densities, and weights.

    Only bins where both runs put more than ten samples are kept; the
    weights are 1/(1/n₁ + 1/n₂).
    """
    lo = max(np.percentile(e1, 0.5), np.percentile(e2, 0.5))
    hi = min(np.percentile(e1, 99.5), np.percentile(e2, 99.5))
    edges = np.linspace(lo, hi, BINS + 1)
    n1, _ = np.histogram(e1, edges)
    n2, _ = np.histogram(e2, edges)
    ok = (n1 > 10) & (n2 > 10)
    x = 0.5 * (edges[1:] + edges[:-1])[ok]
    y = np.log(n2[ok] / len(e2)) - np.log(n1[ok] / len(e1))
    return x, y, 1 / (1 / n1[ok] + 1 / n2[ok])


def slope(e1, e2):
    """The weighted least-squares slope of ln[P₂(E)/P₁(E)] against E."""
    x, y, w = ratio_points(e1, e2)
    xm = np.sum(w * x) / np.sum(w)
    ym = np.sum(w * y) / np.sum(w)
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)


rng = np.random.default_rng(6)
print(f"required slope β₁₃₀ − β₁₄₀ = {REQUIRED:.3f} per eV")
shape = 400
exact = [rng.gamma(shape, KB * t, 100_000) for t in TEMPS]
boot = stats.block_bootstrap(exact, slope, 1, 200, rng)
print(f"gamma density of shape {shape}, 100 000 independent samples at each "
      f"temperature: slope {slope(*exact):.3f} ± {boot.std():.3f} "
      f"({(slope(*exact) - REQUIRED) / boot.std():+.1f} errors)")

energies = {}
for kind, label in (("csvr", "CSVR"), ("berendsen", "Berendsen")):
    runs = []
    for t in TEMPS:
        r = np.load(RUNS / f"ensemble_{kind}_{t:g}.npz")
        e = r["potential"] + r["kinetic"]
        start = 10 * stats.detect_equilibration(e[::10], nskip=10)[0]
        runs.append(e[start:])
        mean_e, error_e, _ = stats.standard_error(e[start:])
        print(f"{label} at {t:g} K: start {start * 0.01:.0f} ps; mean "
              f"{mean_e:.4f} ± {error_e:.4f} eV, spread "
              f"{e[start:].std():.4f} eV")
    energies[label] = runs
    fitted = slope(*runs)
    boot = stats.block_bootstrap(runs, slope, 2000, 200, rng)
    print(f"{label}: slope {fitted:.2f} ± {boot.std():.2f} per eV, "
          f"{fitted / REQUIRED:.2f} ± {boot.std() / REQUIRED:.2f} of the "
          f"required ({(fitted - REQUIRED) / boot.std():+.1f} errors)")

long_run = np.load(RUNS / "long_csvr.npz")["potential"]
mean, error, g = stats.standard_error(long_run)
boot = stats.block_bootstrap([long_run], np.mean, 2000, 400, rng)
print(f"2 ns run, U: √(g s²/n) = {error:.4f} eV; block bootstrap with "
      f"blocks of 20 ps {boot.std():.4f} eV")
for t in TEMPS:
    print(f"canonical spread of E at {t:g} K, √(k_B T² C_V) with C_V = 2.347 "
          f"N k_B: {math.sqrt(KB * t**2 * 2.347 * 256 * KB):.4f} eV")
print(f"C_V ΔT for 10 K: {2.347 * 256 * KB * 10:.3f} eV")
gap = []
for t, csvr_e, ber_e in zip(TEMPS, energies["CSVR"], energies["Berendsen"],
                            strict=True):
    m1, s1, _ = stats.standard_error(csvr_e)
    m2, s2, _ = stats.standard_error(ber_e)
    gap.append(f"{t:g} K: {(m1 - m2) / math.hypot(s1, s2):+.2f}")
print("mean energies, CSVR less Berendsen, in combined errors: "
      + ", ".join(gap))
spread_ratio = np.mean([energies["CSVR"][k].std()
                        / energies["Berendsen"][k].std() for k in (0, 1)])
print(f"spread under CSVR over that under Berendsen {spread_ratio:.2f}; its "
      f"square {spread_ratio**2:.1f}")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.2),
                                 gridspec_kw=dict(wspace=0.35))
for label, (e1, e2) in energies.items():
    colour = THERMOSTAT[label]
    edges = np.linspace(-9.6, -7.2, 80)
    for e, strength in ((e1, 0.5), (e2, 1.0)):
        ax_a.hist(e, bins=edges, density=True, histtype="step",
                  color=viz.tint(colour, strength), lw=0.9)
ax_a.set_xlabel(r"$E = K + U$ / eV")
ax_a.set_ylabel(r"density / eV$^{-1}$")
viz.panel_tag(ax_a, "a")

for label, (e1, e2) in energies.items():
    x, y, _ = ratio_points(e1, e2)
    ax_b.plot(x, y, "o", ms=2.5, color=THERMOSTAT[label], label=label)
    xm, ym = x.mean(), y.mean()
    ax_b.plot(x, ym + REQUIRED * (x - xm), **REFERENCE_STYLE, lw=0.8)
ax_b.set_xlabel(r"$E$ / eV")
ax_b.set_ylabel("log of the ratio of densities")
ax_b.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "ensemble.pdf")))
