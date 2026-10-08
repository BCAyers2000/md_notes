"""Figure fig:ob-faults: six ways to get an observable wrong.

Liquid argon (runs.py, argon_nve and argon_langevin; Chapter 15's runs).
(a) The MSD from positions wrapped into the cell (oxblood) against the
unwrapped paths (teal). (b) A run whose starting velocities kept their
drift: the MSD of the same paths with the centre of mass moving at the
speed such velocities give, √(3k_BT/Nm) (oxblood), with the drift
removed (teal). (c) D fitted to windows of the MSD that start at t₁ and
end at 10t₁ against t₁, with the 2-20 ps value (dotted). (d) g(r) taken
past half the cell with the minimum image (oxblood) against the true
g(r) from a 2048-atom run (teal). (e) The VACF under Langevin friction
of 10 ps⁻¹ (oxblood) against fixed energy (teal). (f) The Green-Kubo
integral of the shear stress from one 1 ns run (256 atoms) out to 40 ps,
against Section 16.6's 14 ns value (dotted).

Prints the size of each error.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import RUNS, ch15

from mdlab import cell, units, viz
from mdlab.analysis import structure, transport
from mdlab.viz import ACCENT, OXBLOOD, THRESHOLD_STYLE

DT = 10.0
PA_S = units.ELEMENTARY_CHARGE / units.ANGSTROM**3 * units.FEMTOSECOND
run = np.load(RUNS / "argon_nve.npz")
v = run["velocities"].astype(float)
m = run["masses"]
h = run["cell"]
n = v.shape[1]
temperature = 2 * run["kinetic"].mean() / ((3 * n - 3) * units.KB)
start = run["positions"][0].astype(float)
paths = start + np.concatenate(
    [np.zeros((1, n, 3)), np.cumsum(0.5 * DT * (v[1:] + v[:-1]), axis=0)])
sparse = paths[::10]  # every 100 fs
lag = np.arange(1001)
t = 100.0 * lag
good = transport.msd(sparse, 1000, masses=m, origin_step=5).sum(1)
d_good = transport.diffusion_coefficient(t, good, 2000.0, 20000.0)

wrapped = np.array([cell.wrap(x, h) for x in sparse])
bad = transport.msd(wrapped, 1000, masses=m, origin_step=5).sum(1)
print(f"(a) wrapped: MSD at 20, 50, 100 ps {bad[200]:.1f}, {bad[500]:.1f}, "
      f"{bad[1000]:.1f} Å² against {good[200]:.1f}, {good[500]:.1f}, "
      f"{good[1000]:.1f}; two points at random in the cell are L²/2 = "
      f"{h[0, 0] ** 2 / 2:.0f} Å² apart on average, squared")

kt_m = units.KB * temperature / (m[0] * units.MV2_TO_EV)
v_drift = np.sqrt(3 * kt_m / n)
drift = np.array([v_drift, 0.0, 0.0])
drifting = sparse + drift * (100.0 * np.arange(len(sparse)))[:, None, None]
naive = transport.msd(drifting, 1000, remove_drift=False,
                      origin_step=5).sum(1)
d_naive = transport.diffusion_coefficient(t, naive, 2000.0, 20000.0)
removed = transport.msd(drifting, 1000, masses=m, origin_step=5).sum(1)
d_removed = transport.diffusion_coefficient(t, removed, 2000.0, 20000.0)
print(f"(b) drift √(3k_BT/Nm) = {v_drift:.2e} Å/fs: D from 2-20 ps "
      f"{d_naive * 1e4:.3f}e-4 against {d_good * 1e4:.3f}e-4 "
      f"({100 * (d_naive / d_good - 1):.0f}% high); removed, "
      f"{d_removed * 1e4:.3f}e-4")

dense = transport.msd(paths, 2000, masses=m, origin_step=10).sum(1)
t_dense = DT * np.arange(2001)
starts = np.array([20.0, 50, 100, 200, 500, 1000, 2000])
windows = [transport.diffusion_coefficient(t_dense, dense, s, 10 * s)
           for s in starts]
for s, d in zip(starts, windows, strict=True):
    print(f"(c) window {s / 1000:.2f}-{10 * s / 1000:.1f} ps: D = "
          f"{d * 1e4:.3f}e-4 ({100 * (d / d_good - 1):+.0f}%)")

big = np.load(ch15.RUNS / "size_2048.npz")
hb = big["cell"]
xb, gb = structure.rdf(big["positions"][1::10].astype(float), hb, 16.0, 320)
small = np.load(ch15.RUNS / "long_csvr.npz")
hs = small["cell"]
frames = small["positions"][1::20].astype(float)
reach = np.sqrt(3) * hs[0, 0] / 2
edges = np.linspace(0, 16.0, 321)
counts = np.zeros(320)
for f in frames:
    i, j = np.triu_indices(len(f), 1)
    d = np.linalg.norm(cell.minimum_image(f[j] - f[i], hs), axis=1)
    counts += 2 * np.histogram(d, edges)[0]
ideal = len(frames) * 256 * 256 / np.linalg.det(hs) * structure.shell_volumes(
    edges)
g_bad = counts / ideal
xs = 0.5 * (edges[1:] + edges[:-1])
beyond = (xs > hs[0, 0] / 2 + 0.5) & (xs < 15.5)
print(f"(d) half the 256-atom cell is {hs[0, 0] / 2:.2f} Å; past it the "
      f"minimum-image g falls to {g_bad[np.argmin(abs(xs - 14))]:.2f} at "
      f"14 Å, where the 2048-atom g is {gb[np.argmin(abs(xb - 14))]:.3f}")

lang = np.load(RUNS / "argon_langevin.npz")
c_n = transport.velocity_autocorrelation(v, 200)
c_l = transport.velocity_autocorrelation(lang["velocities"].astype(float),
                                         200)
t_l = 2 * lang["kinetic"].mean() / ((3 * n - 3) * units.KB)
d_n = transport.running_integral(transport.velocity_autocorrelation(
    v, 1000), DT)[200:].mean() / 3
d_l = transport.running_integral(transport.velocity_autocorrelation(
    lang["velocities"].astype(float), 1000), DT)[200:].mean() / 3
for name, c in (("fixed energy", c_n), ("Langevin", c_l)):
    print(f"(e) {name}: C_vv first falls below half of C_vv(0) at "
          f"{DT * np.argmax(c / c[0] < 0.5):.0f} fs and first reaches zero "
          f"at {DT * np.argmax(c < 0):.0f} fs")
print(f"(e) Langevin 10 ps⁻¹ at {t_l:.1f} K: D = {d_l * 1e4:.3f}e-4 against "
      f"{d_n * 1e4:.3f}e-4 at fixed energy ({100 * (d_l / d_n - 1):.0f}%)")

one = np.load(ch15.RUNS / "size_256.npz")
corr = transport.correlation(one["shear"], max_lag=4000) / 3
t_one = 2 * one["kinetic"].mean() / ((3 * 256 - 3) * units.KB)
eta_one = (transport.running_integral(corr, DT) * one["volume"].mean()
           / (units.KB * t_one) * PA_S)
eta_all = 1.853e-4
for upper in (2, 10, 20, 40):
    print(f"(f) one 1 ns run, integral to {upper} ps: "
          f"{eta_one[upper * 100] * 1e4:.2f}e-4 Pa s")

viz.use_style(notebook=False)
fig, axes = plt.subplots(2, 3, figsize=(viz.FULL, 4.3),
                         gridspec_kw=dict(wspace=0.45, hspace=0.75))
ax = axes[0, 0]
ax.plot(t / 1000, good, color=ACCENT, lw=0.9)
ax.plot(t / 1000, bad, color=OXBLOOD, lw=0.9)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"MSD / \AA$^2$")
ax.set_title("(a) wrapped positions", fontsize=7, pad=9, loc="left")

ax = axes[0, 1]
ax.plot(t / 1000, good, color=ACCENT, lw=0.9)
ax.plot(t / 1000, naive, color=OXBLOOD, lw=0.9)
ax.set_xlim(0, 30)
ax.set_ylim(0, 150)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"MSD / \AA$^2$")
ax.set_title("(b) centre-of-mass drift", fontsize=7, pad=9, loc="left")

ax = axes[0, 2]
ax.semilogx(starts / 1000, np.array(windows) * 1e4, "o-", color=OXBLOOD,
            ms=3, lw=0.8)
ax.axhline(d_good * 1e4, **THRESHOLD_STYLE)
ax.set_ylim(0, 4.5)
ax.set_xlabel(r"window start $t_1$ / ps")
ax.set_ylabel(r"$D$ / $10^{-4}$ \AA$^2$\,fs$^{-1}$")
ax.set_title("(c) fit window", fontsize=7, pad=9, loc="left")

ax = axes[1, 0]
ax.plot(xb, gb, color=ACCENT, lw=0.9)
ax.plot(xs[xs < 15.9], g_bad[xs < 15.9], color=OXBLOOD, lw=0.9)
ax.axvline(hs[0, 0] / 2, **THRESHOLD_STYLE)
ax.set_xlim(8, 16)
ax.set_ylim(0, 1.3)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$g(r)$")
ax.set_title("(d) shell beyond half-cell", fontsize=7, pad=9, loc="left")

ax = axes[1, 1]
lag_v = DT * np.arange(201)
ax.plot(lag_v, c_n / c_n[0], color=ACCENT, lw=0.9)
ax.plot(lag_v, c_l / c_l[0], color=OXBLOOD, lw=0.9)
ax.axhline(0, color="black", lw=0.4)
ax.set_xlabel(r"$t$ / fs")
ax.set_ylabel(r"$C_{vv}(t)/C_{vv}(0)$")
ax.set_title("(e) thermostat damping", fontsize=7, pad=9, loc="left")

ax = axes[1, 2]
ax.plot(DT * np.arange(len(eta_one)) / 1000, eta_one * 1e4, color=OXBLOOD,
        lw=0.8)
ax.axhline(eta_all * 1e4, **THRESHOLD_STYLE)
ax.set_xlabel(r"upper limit / ps")
ax.set_ylabel(r"$\eta_{\mathrm s}$ / $10^{-4}$ Pa$\,$s")
ax.set_title("(f) integration limit", fontsize=7, pad=9, loc="left")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables",
                                             "faults.pdf")))
