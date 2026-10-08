"""Figure fig:cv-faults and Table tab:cv-faults: faults of convergence.

Each panel sets a fault (teal) against a healthy run (grey dashed).
(a) Averaged from the start: the running mean of U/N of a melting run
(runs.py, melt_s7) from t = 0 (teal), against the same from the start
found (grey), and the 2 ns mean (dotted). (b) The naive error bar: the twenty
pieces of 100 ps of the 2 ns run, mean of U less that of 2 ns, with
s/√n (teal) and √(g s²/n) (grey). (c) Too short for a plateau: the error
of U from blocks for the first 20 ps of the run (teal) and for all 2 ns
(grey), each divided by its √(g s²/n). (d) A drifting energy: the total
energy per atom at δt = 35 fs (teal) and 10 fs (grey). (e) A thermostat
that fails the ensemble test: ln[P₁₄₀/P₁₃₀] under Berendsen coupling
(teal) and CSVR (grey), each less its mean, against E less its mean. (f)
A box too small: the squared displacement over 6t for 256 atoms (teal)
and 2048 (grey), against t.

Prints the numbers of the table.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, RUNS

from mdlab import viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, REFERENCE, THRESHOLD_STYLE

HEALTHY = {"color": REFERENCE, "linestyle": "--", "lw": 0.8}
long_u = np.load(RUNS / "long_csvr.npz")["potential"]
truth_u = long_u.mean()

# (a) averaged from the start
own_start = stats.detect_equilibration(long_u, nskip=1000)[0]
print(f"the 2 ns run's own start for U: {own_start * DT / 1000:.1f} ps")
melts = [np.load(RUNS / f"melt_s{s}.npz")["potential"] / 256
         for s in range(8)]
starts = [stats.detect_equilibration(u, nskip=20)[0] for u in melts]
from_start = np.mean([u.mean() for u in melts])
after = np.mean([u[s:].mean() for u, s in zip(melts, starts, strict=True)])
err_after = np.mean([stats.standard_error(u[s:])[1]
                     for u, s in zip(melts, starts, strict=True)])
print(f"(a) eight melting runs: U/N from the start {1000 * from_start:.3f} "
      f"meV, after the starts {1000 * after:.3f} meV; shift "
      f"{1000 * (after - from_start):.3f} meV, "
      f"{(after - from_start) / err_after:.1f} of the mean error bar "
      f"{1000 * err_after:.3f} meV")
shown, shown_start = melts[7], starts[7]
running = np.cumsum(shown) / np.arange(1, len(shown) + 1)
kept = shown[shown_start:]
running_kept = np.cumsum(kept) / np.arange(1, len(kept) + 1)
print(f"    run s7: start {shown_start * DT / 1000:.1f} ps; mean from t = 0 "
      f"{1000 * shown.mean():.3f} meV, from the start {1000 * kept.mean():.3f}"
      f" ± {1000 * stats.standard_error(kept)[1]:.3f} meV; 2 ns "
      f"{1000 * truth_u / 256:.3f} meV")

# (b) the naive error bar
pieces = long_u[:200_000].reshape(20, -1)
held_naive = held_g = 0
naive_bars, g_bars, piece_means = [], [], []
for piece in pieces:
    m, e, _ = stats.standard_error(piece)
    naive = piece.std(ddof=1) / math.sqrt(len(piece))
    held_naive += abs(m - truth_u) <= naive
    held_g += abs(m - truth_u) <= e
    piece_means.append(m)
    naive_bars.append(naive)
    g_bars.append(e)
print(f"(b) twenty pieces of 100 ps: {held_naive} naive bars and {held_g} "
      f"bars with √g hold the 2 ns mean")

# (c) too short for a plateau
short = long_u[:2000]
g_short = stats.statistical_inefficiency(short)
g_long = stats.statistical_inefficiency(long_u)
blocks_short = stats.block_average(short)
blocks_long = stats.block_average(long_u, min_blocks=8)
print(f"(c) first 20 ps: {len(short) / g_short:.1f} effective samples; "
      f"2 ns: {len(long_u) / g_long:.0f}")
print("    20 ps report:")
print(stats.convergence_report(short, DT))

# (d) a drifting energy
drift = {}
for dt in (10, 35):
    r = np.load(RUNS / f"step_{dt}.npz")
    e = (r["potential"] + r["kinetic"]) / 256
    drift[dt] = (r["times"] / 1000, e - e[0])
    every = max(1, len(e) // 20000)
    ratio = stats.drift_test(r["times"][::every], e[::every])[2]
    print(f"(d) δt = {dt} fs: drift test {ratio:+.1f} errors")

# (e) the ensemble test
ratios = {}
for kind in ("berendsen", "csvr"):
    e = []
    for t in (130, 140):
        r = np.load(RUNS / f"ensemble_{kind}_{t}.npz")
        total = r["potential"] + r["kinetic"]
        start = 10 * stats.detect_equilibration(total[::10], nskip=10)[0]
        e.append(total[start:])
    lo = max(np.percentile(e[0], 0.5), np.percentile(e[1], 0.5))
    hi = min(np.percentile(e[0], 99.5), np.percentile(e[1], 99.5))
    edges = np.linspace(lo, hi, 41)
    n1, _ = np.histogram(e[0], edges)
    n2, _ = np.histogram(e[1], edges)
    ok = (n1 > 10) & (n2 > 10)
    x = 0.5 * (edges[1:] + edges[:-1])[ok]
    y = np.log(n2[ok] / len(e[1])) - np.log(n1[ok] / len(e[0]))
    ratios[kind] = (x - x.mean(), y - y.mean())

# (f) a box too small
msd = {}
for n in (256, 2048):
    pos = np.load(RUNS / f"size_{n}.npz")["positions"].astype(float)
    pos = pos - pos.mean(axis=1, keepdims=True)
    lags = np.arange(1, 31)
    msd[n] = (lags, np.array([np.mean(np.sum((pos[k:] - pos[:-k]) ** 2, -1))
                              for k in lags]) / (6 * lags * 1000))
    print(f"(f) N = {n}: squared displacement over 6t at 20 ps "
          f"{1e4 * msd[n][1][19]:.3f} e-4 Å²/fs")
print(f"    ratio at 20 ps {msd[256][1][19] / msd[2048][1][19]:.3f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(2, 3, figsize=(viz.FULL, 4.3),
                         gridspec_kw=dict(wspace=0.55, hspace=0.75))
ax = axes[0, 0]
t = np.arange(len(running)) * DT / 1000
ax.plot(t, 1000 * running, color=ACCENT, lw=0.9)
ax.plot(t[shown_start:], 1000 * running_kept, **HEALTHY)
ax.axhline(1000 * truth_u / 256, **THRESHOLD_STYLE)
ax.set_ylim(-50.8, -49.6)
ax.set_xlabel("time / ps")
ax.set_ylabel(r"running $\bar U/N$ / meV")
ax.set_title("(a) transient retained", fontsize=7, pad=9, loc="left")

ax = axes[0, 1]
k = np.arange(1, 21)
dev = 1000 * (np.array(piece_means) - truth_u)
ax.errorbar(k - 0.15, dev, yerr=1000 * np.array(g_bars), fmt="none",
            ecolor=REFERENCE, lw=0.8)
ax.errorbar(k + 0.15, dev, yerr=1000 * np.array(naive_bars), fmt="o",
            ms=2, color=ACCENT, lw=0.8, capsize=0)
ax.axhline(0, color="black", lw=0.4)
ax.set_xlabel("piece of 100 ps")
ax.set_ylabel(r"$\bar U-\bar U_{2\,\mathrm{ns}}$ / meV")
ax.set_title("(b) correlation omitted", fontsize=7, pad=9, loc="left")

ax = axes[0, 2]
ax.plot(blocks_short["length"] * DT / 1000,
        blocks_short["error"] / math.sqrt(g_short * short.var() / len(short)),
        "o-", ms=2.5, lw=0.8, color=ACCENT)
ax.plot(blocks_long["length"] * DT / 1000,
        blocks_long["error"] / math.sqrt(g_long * long_u.var() / len(long_u)),
        **HEALTHY)
ax.axhline(1, **THRESHOLD_STYLE)
ax.set_xscale("log")
ax.set_xlabel("block / ps")
ax.set_ylabel(r"ratio to $g$'s error")
ax.set_title("(c) short run", fontsize=7, pad=9, loc="left")

ax = axes[1, 0]
for dt, style in ((10, HEALTHY), (35, {"color": ACCENT, "lw": 0.6})):
    tt, de = drift[dt]
    ax.plot(tt, 1000 * de, **style)
ax.set_xlabel("time / ps")
ax.set_ylabel(r"$(E - E_0)/N$ / meV")
ax.set_title("(d) long time step", fontsize=7, pad=9, loc="left")

ax = axes[1, 1]
x, y = ratios["csvr"]
ax.plot(x, y, **HEALTHY)
x, y = ratios["berendsen"]
ax.plot(x, y, "o", ms=2.5, color=ACCENT)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel(r"$E - \bar E$ / eV")
ax.set_ylabel("log ratio")
ax.set_title("(e) wrong ensemble", fontsize=7, pad=9, loc="left")

ax = axes[1, 2]
for n, style in ((2048, HEALTHY), (256, {"color": ACCENT, "lw": 0.9})):
    lags, value = msd[n]
    ax.plot(lags, 1e4 * value, **style)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$\langle\Delta r^2\rangle/6t$ / $10^{-4}$ \AA$^2$ fs$^{-1}$")
ax.set_title("(f) small cell", fontsize=7, pad=9, loc="left")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "faults.pdf")))
