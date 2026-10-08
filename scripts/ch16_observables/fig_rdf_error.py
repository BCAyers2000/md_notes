"""Figure fig:ob-rdf-error: how long the radial distribution function takes.

Chapter 15's 2 ns CSVR run of liquid argon, frames every 500 fs. (a) The
radial distribution function from the first 10 ps, less that of the
whole run, with twice its standard error from the 20 frames (band),
each bin's statistical inefficiency taken from the whole run. (b)
The standard error of g at its first peak against the length of run used,
from the series of single-frame values and its integrated correlation
time (line), against the scatter of independent pieces of that length
(points).

Prints the single-frame spread and τ_int at the peak, the error after
10 ps, 100 ps and 2 ns, the run needed for an error of 0.01, and the
fraction of bins whose 10 ps value lies within two errors of the 2 ns
one.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import ch15

from mdlab import viz
from mdlab.analysis import stats, structure
from mdlab.viz import ACCENT, REFERENCE_STYLE

FRAME_DT = 500.0  # fs
liquid = np.load(ch15.RUNS / "long_csvr.npz")
h = liquid["cell"]
frames = liquid["positions"][1:].astype(float)
x, per_frame = structure.rdf(frames, h, h[0, 0] / 2 - 1e-6, 232,
                             per_frame=True)
whole = per_frame.mean(0)
peak = np.argmax(whole)
series = per_frame[:, peak]
g_peak = stats.statistical_inefficiency(series)
tau = 0.5 * g_peak * FRAME_DT
spread = series.std(ddof=1)
print(f"first peak at {x[peak]:.3f} Å: g = {whole[peak]:.4f}; one frame "
      f"spreads by {spread:.4f}; frames {FRAME_DT:.0f} fs apart have a "
      f"statistical inefficiency of {g_peak:.2f} (τ_int = {tau:.0f} fs)")
for t_ps in (10, 100, 2000):
    n = int(t_ps * 1000 / FRAME_DT)
    err = spread * np.sqrt(g_peak / n)
    print(f"error at the peak after {t_ps} ps: {err:.4f} "
          f"({100 * err / whole[peak]:.2f}%)")
need = 2 * tau * spread**2 / 0.01**2
print(f"run for an error of 0.01 at the peak: {need / 1000:.1f} ps")

# Each bin's error from 20 frames: their spread, with the statistical
# inefficiency of that bin measured over the whole run.
first = per_frame[:20]
short = first.mean(0)
varies = per_frame.std(0) > 0
ineff = np.ones(len(x))
ineff[varies] = [stats.statistical_inefficiency(per_frame[:, k])
                 for k in np.flatnonzero(varies)]
short_err = first.std(0, ddof=1) * np.sqrt(ineff / len(first))
nonzero = whole > 0.05
within = np.abs(short - whole)[nonzero] <= 2 * short_err[nonzero]
print(f"10 ps against 2 ns: {within.mean():.2f} of the bins within two "
      f"errors; largest error {short_err.max():.3f}")

lengths = np.array([5, 10, 20, 50, 100, 200, 500])  # ps
scatter = []
for t_ps in lengths:
    n = int(t_ps * 1000 / FRAME_DT)
    pieces = [series[k:k + n].mean() for k in range(0, len(series) - n + 1, n)]
    scatter.append(np.std(pieces, ddof=1))
    print(f"{t_ps} ps: {len(pieces)} pieces spread by {scatter[-1]:.4f}; "
          f"predicted {spread * np.sqrt(g_peak / n):.4f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.35))
ax = axes[0]
ax.fill_between(x, -2 * short_err, 2 * short_err, color=viz.tint(ACCENT, 0.25),
                lw=0)
ax.plot(x, short - whole, color=ACCENT, lw=0.8)
ax.axhline(0, color="black", lw=0.4)
ax.set_xlim(0, h[0, 0] / 2)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$g_{10\,\mathrm{ps}}(r) - g_{2\,\mathrm{ns}}(r)$")
viz.panel_tag(ax, "a")

ax = axes[1]
t = np.logspace(np.log10(3), np.log10(2000), 50)
ax.loglog(t, spread * np.sqrt(g_peak * FRAME_DT / (t * 1000)),
          **REFERENCE_STYLE, lw=0.8)
ax.loglog(lengths, scatter, "o", color=ACCENT, ms=3)
ax.set_xlabel("length of run / ps")
ax.set_ylabel(r"error of $g(r)$ at its peak")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables",
                                             "rdf_error.pdf")))
