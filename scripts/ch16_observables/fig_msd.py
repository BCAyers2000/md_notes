"""Figure fig:ob-msd: how far atoms wander.

Liquid argon at fixed energy (runs.py, argon_nve: 100 ps, velocities
every 10 fs, from which the paths are summed by the trapezium rule). (a)
The mean squared displacement against the lag, with the ballistic
3k_BT t²/m (dashed) and 6Dt (dotted). (b) Three estimates of D against
the time they reach: MSD/6t, the slope of the MSD over a short window
divided by 6, and the Green-Kubo integral ⅓∫C_vv (line); the dotted
line is D from the 2-20 ps fit.

Prints the paths' largest difference from the saved positions, T, D
from the fit and from Green-Kubo, D from Chapter 15's 2 ns run with its
error from 20 blocks of 100 ps (all origins) against the scatter of 20
single-origin fits, and the agreement with ASE's DiffusionCoefficient.
"""

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.md.analysis import DiffusionCoefficient
from ch16 import RUNS, ch15

from mdlab import units, viz
from mdlab.analysis import transport
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, THRESHOLD_STYLE

DT = 10.0
run = np.load(RUNS / "argon_nve.npz")
v = run["velocities"].astype(float)
m = run["masses"]
n = v.shape[1]
temperature = 2 * run["kinetic"].mean() / ((3 * n - 3) * units.KB)
start = run["positions"][0].astype(float)
paths = start + np.concatenate(
    [np.zeros((1, n, 3)), np.cumsum(0.5 * DT * (v[1:] + v[:-1]), axis=0)])
saved = run["positions"].astype(float)
print(f"T = {temperature:.2f} K; paths differ from the saved positions by "
      f"at most {np.abs(paths[::50] - saved).max():.1e} Å over 100 ps")

lags = np.unique(np.round(np.logspace(0, np.log10(5000), 60)).astype(int))
t = lags * DT
msd = transport.msd(paths, masses=m, origin_step=10, lags=lags).sum(1)
full = transport.msd(paths, 2000, masses=m, origin_step=10).sum(1)
t_full = DT * np.arange(2001)
d_fit = transport.diffusion_coefficient(t_full, full, 2000.0, 20000.0)
c = transport.velocity_autocorrelation(v, 2000)
gk = transport.running_integral(c, DT) / 3
plateau = gk[200:1001].mean()
print(f"D from the slope over 2-20 ps: {d_fit * 1e4:.3f}e-4 Å²/fs; "
      f"Green-Kubo ⅓∫C over 2-10 ps: {plateau * 1e4:.3f}e-4 "
      f"({100 * (plateau / d_fit - 1):.1f}%)")
top = np.argmax(gk)
print(f"the running integral peaks at {gk[top] / plateau:.2f} of its plateau "
      f"at {top * DT:.0f} fs; from 1 to 10 ps it strays from the plateau by "
      f"at most {100 * np.abs(gk[100:1001] / plateau - 1).max():.1f}%")
kt_m = units.KB * temperature / (m[0] * units.MV2_TO_EV)
print(f"ballistic: MSD = 3k_BT t²/m = {3 * kt_m:.3e} t² (Å², fs); at 100 fs "
      f"{3 * kt_m * 1e4:.3f} Å² against {full[10]:.3f}")
cross = 2 * d_fit / kt_m
print(f"3k_BTt²/m = 6Dt at t = 2Dm/k_BT = {cross:.0f} fs")
window = 50  # frames, 0.5 ps
slope = np.gradient(full, DT) / 6
reach = t_full[np.argmax(full[1:] / (6 * t_full[1:]) > 0.9 * d_fit) + 1]
print(f"MSD/6t reaches 0.9 D only at "
      f"{reach / 1000:.1f} ps")

# Chapter 15's 2 ns run at fixed energy, positions every 500 fs.
long = np.load(ch15.RUNS / "long_nve.npz")
pos = long["positions"][1:].astype(float)
t_long = 2 * long["kinetic"].mean() / ((3 * n - 3) * units.KB)
blocks = np.array_split(pos, 20)
lag_block = np.arange(41)  # 0 to 20 ps
d_all, d_one = [], []
for b in blocks:
    a = transport.msd(b, 40, masses=m).sum(1)
    d_all.append(transport.diffusion_coefficient(500 * lag_block, a, 2000.0,
                                                 20000.0))
    o = transport.msd(b, 40, masses=m, origin_step=len(b)).sum(1)
    d_one.append(transport.diffusion_coefficient(500 * lag_block, o, 2000.0,
                                                 20000.0))
mean_all, err_all = np.mean(d_all), np.std(d_all, ddof=1) / np.sqrt(20)
spread_ratio = np.std(d_one, ddof=1) / np.std(d_all, ddof=1)
print(f"Chapter 15's 2 ns at {t_long:.2f} K: D = {mean_all * 1e4:.3f} ± "
      f"{err_all * 1e4:.3f}e-4 Å²/fs from 20 blocks, every origin; one "
      f"origin per block spreads {spread_ratio:.1f}"
      f" times as widely")

# Checked against ASE: one origin per segment, every lag fitted.
ase_dt = 500.0 * np.sqrt(units.FORCE_TO_ACCEL)  # 500 fs in ASE's time unit
images = [Atoms(f"Ar{n}", positions=x, cell=long["cell"].T) for x in pos[:400]]
theirs = DiffusionCoefficient(images, ase_dt)
theirs.calculate(number_of_segments=4)
theirs_d = np.mean(theirs.slopes[0], axis=1) * np.sqrt(units.FORCE_TO_ACCEL)
ours_d = []
for seg in np.array_split(pos[:400], 4):
    o = transport.msd(seg, remove_drift=False, origin_step=len(seg)).sum(1)
    ours_d.append(transport.diffusion_coefficient(
        500 * np.arange(len(seg)), o, 0.0, 500 * (len(seg) - 1)))
print(f"ASE's DiffusionCoefficient, 4 segments of 50 ps: largest relative "
      f"difference {np.max(np.abs(np.array(ours_d) / theirs_d - 1)):.1e}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.32))
ax = axes[0]
ax.loglog(t / 1000, 3 * kt_m * t**2, **REFERENCE_STYLE, lw=0.8)
ax.loglog(t / 1000, 6 * d_fit * t, **THRESHOLD_STYLE)
ax.loglog(t / 1000, msd, color=ACCENT, lw=1.0)
ax.set_ylim(1e-3, 300)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"MSD / \AA$^2$")
viz.panel_tag(ax, "a")

ax = axes[1]
ts = t_full[1:] / 1000
ax.semilogx(ts, full[1:] / (6 * t_full[1:]) * 1e4, color=OCHRE, lw=0.9,
            label=r"MSD$/6t$")
ax.semilogx(ts, slope[1:] * 1e4, color=ACCENT, lw=1.6, alpha=0.5,
            label="slope/6")
ax.semilogx(ts, gk[1:] * 1e4, color="black", lw=0.6,
            label=r"$\frac13\int C_{vv}$")
ax.axhline(d_fit * 1e4, **THRESHOLD_STYLE)
ax.set_ylim(0, 6)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$D$ / $10^{-4}$ \AA$^2$\,fs$^{-1}$")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "msd.pdf")))
