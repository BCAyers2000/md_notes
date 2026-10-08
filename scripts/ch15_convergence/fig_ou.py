"""Figure fig:cv-ou: samples that remember.

The Ornstein-Uhlenbeck step x' = cx + √(1 − c²)ξ with c = 0.95, read as
samples 10 fs apart (correlation time −δt/ln c = 195 fs). (a) 400
samples of it against 400 independent ones (grey). (b) Its
autocorrelation estimated from 10 000 samples against the exact c^k
(dashed), with the first lag beyond 3 at which the estimate reaches zero
(dotted). (c) The spread of the mean of n samples, 4000 series over,
against σ/√n (dotted) and the exact σ√(g_n/n), g_n = 1 + 2Σ(1 − k/n)c^k
(dashed), which tends to σ√(g/n) with g = (1 + c)/(1 − c).

Prints the numbers of Section 15.2.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, ou

from mdlab import viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, REFERENCE, REFERENCE_STYLE, THRESHOLD_STYLE

C = 0.95
G_EXACT = (1 + C) / (1 - C)
TAU = -DT / math.log(C)
print(f"c = {C}: correlation time −δt/ln c = {TAU:.1f} fs; g = (1 + c)/(1 − c)"
      f" = {G_EXACT:.1f}; gδt/2 = {G_EXACT * DT / 2:.1f} fs; √g = "
      f"{math.sqrt(G_EXACT):.2f}")

rng = np.random.default_rng(1)
series = ou(10_000, C, rng)
acf = stats.autocorrelation(series)
lags = np.arange(len(acf))
cut = np.nonzero((acf[1:] <= 0) & (lags[1:] > 3))[0][0] + 1
g_est = stats.statistical_inefficiency(series)
full = 1 + 2 * np.sum((1 - lags[1:] / len(series)) * acf[1:])
print(f"one series of 10 000: C first ≤ 0 at lag {cut} ({cut * DT:.0f} fs); "
      f"g estimated {g_est:.1f}; summed over every lag {full:.1e}")
print(f"  spread of C beyond the decay (lags 300 to 1000): "
      f"{acf[300:1000].std():.3f}; 1/√n = {1 / math.sqrt(len(series)):.3f}")
mean, error, _ = stats.standard_error(series, g_est)
print(f"  mean {mean:.3f}, error √(g s²/n) {error:.3f}, naive s/√n "
      f"{series.std(ddof=1) / math.sqrt(len(series)):.4f}")

ests = [stats.statistical_inefficiency(ou(10_000, C, rng)) for _ in range(200)]
print(f"g estimated from 200 series of 10 000: mean {np.mean(ests):.1f}, "
      f"spread {np.std(ests):.1f}")


def g_finite(n):
    k = np.arange(1, n)
    return 1 + 2 * np.sum((1 - k / n) * C**k)


sizes = np.array([10, 30, 100, 300, 1000, 3000])
many = ou(int(sizes[-1]), C, rng, shape=(4000,))
spread = [many[:n].mean(axis=0).std() for n in sizes]
for n, s in zip(sizes, spread, strict=True):
    print(f"n = {n:5d}: spread of the mean {s:.4f}; σ/√n "
          f"{1 / math.sqrt(n):.4f}; σ√(g_n/n) "
          f"{math.sqrt(g_finite(n) / n):.4f} (g_n "
          f"{g_finite(n):.1f}); σ√(g/n) {math.sqrt(G_EXACT / n):.4f}")

viz.use_style(notebook=False)
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(viz.FULL, 2.0), gridspec_kw=dict(wspace=0.6))
t = np.arange(400) * DT / 1000
independent = rng.standard_normal(400)


def stretches(x):
    """Lengths of the runs of samples on one side of zero."""
    sign = np.sign(x)
    cuts = np.nonzero(sign[1:] != sign[:-1])[0] + 1
    return np.diff(np.concatenate([[0], cuts, [len(x)]]))


print(f"panel (a): the series stays on one side of zero for up to "
      f"{stretches(series[:400]).max() * DT / 1000:.2f} ps; the independent "
      f"samples change sign on average every "
      f"{stretches(independent).mean():.1f} samples")
ax_a.plot(t, independent, color=REFERENCE, lw=0.4)
ax_a.plot(t, series[:400], color=ACCENT, lw=0.9)
ax_a.set_xlabel("time / ps")
ax_a.set_ylabel(r"$x$")
viz.panel_tag(ax_a, "a")

show = lags[:160]
ax_b.plot(show * DT, acf[:160], color=ACCENT, lw=1.0)
ax_b.plot(show * DT, C**show, **REFERENCE_STYLE, lw=0.8)
ax_b.axvline(cut * DT, **THRESHOLD_STYLE)
ax_b.axhline(0, color="black", lw=0.4)
ax_b.set_xlabel(r"lag $k\,\delta t$ / fs")
ax_b.set_ylabel(r"$\mathrm{corr}(k)$")
viz.panel_tag(ax_b, "b")

ax_c.loglog(sizes, spread, "o", color=ACCENT, ms=3)
fine = np.unique(np.geomspace(sizes[0], sizes[-1], 50).astype(int))
ax_c.loglog(fine, 1 / np.sqrt(fine), **THRESHOLD_STYLE)
ax_c.loglog(fine, [math.sqrt(g_finite(n) / n) for n in fine],
            **REFERENCE_STYLE, lw=0.8)
ax_c.set_xlabel(r"$n$")
ax_c.set_ylabel("spread of the mean")
viz.panel_tag(ax_c, "c")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence", "ou.pdf")))
