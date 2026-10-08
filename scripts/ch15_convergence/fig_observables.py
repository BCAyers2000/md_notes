"""Figure fig:cv-observables and Table tab:cv-observables: run lengths.

How long each quantity needs.

(a) The error of the mean of U, T and P against the length of run: the
2 ns CSVR run (runs.py, long_csvr) cut into pieces of each length, the
error bar each piece finds for itself averaged over the pieces (filled),
and the spread of the pieces' means (open) where there are at least five,
each divided by the error of the whole run, against T^{−1/2} (dashed).
(b) The diffusion coefficient against the temperature, each with its
error from blocks of 100 ps: CSVR at three temperatures (teal),
Berendsen with τ_T = 0.1 ps at two (ochre), fixed energy (grey), and the
straight line through the CSVR runs at 130 and 140 K (dashed).

Prints the table: the spread of one sample, τ_int under CSVR and at fixed
energy, the error of a 1 ns run, and the run needed for a stated
precision; and the numbers that settle the questions of Chapters 12 to
14: D under the thermostats, the gap between two species' temperatures,
the error of a single run's C_V, and the error of a pressure Chapter 14
took from 50 frames.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, GPA, N_FREE, RUNS, ch12, diffusion

from mdlab import statmech, units, viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE, REFERENCE_STYLE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "ch14_pressure"))
import ch14  # noqa: E402

KB = units.KB


def temperature(kinetic, n_free=N_FREE):
    return 2 * kinetic / (n_free * KB)


def d_blocks(name, blocks):
    """D in 1e-4 Å²/fs from independent blocks, with T; each with errors."""
    r = np.load(RUNS / f"{name}.npz")
    pos = r["positions"].astype(float)
    size = (len(pos) - 1) // blocks
    ds = 1e4 * np.array([diffusion(pos[j * size:(j + 1) * size + 1], 500.0)
                         for j in range(blocks)])
    t_mean, t_err, _ = stats.standard_error(temperature(r["kinetic"]))
    return ds.mean(), ds.std(ddof=1) / math.sqrt(blocks), t_mean, t_err


# (a) and the table
csvr = np.load(RUNS / "long_csvr.npz")
nve = np.load(RUNS / "long_nve.npz")
quantities = {
    "U": (csvr["potential"], nve["potential"], "eV"),
    "T": (temperature(csvr["kinetic"]), temperature(nve["kinetic"]), "K"),
    "P": (csvr["pressure"] * GPA, nve["pressure"] * GPA, "GPa"),
}
targets = {"U": 0.256 * 0.01, "T": 0.1, "P": 0.001}  # 0.01 meV per atom
lengths = np.array([50, 100, 200, 500, 1000, 2000]) * 100  # samples
curves = {}
print("quantity: spread of one sample; τ_int CSVR, fixed energy; error of "
      "1 ns; run for the target")
for name, (a, b, unit) in quantities.items():
    g_a = stats.statistical_inefficiency(a)
    g_b = stats.statistical_inefficiency(b)
    err_ns = math.sqrt(g_a * a.var() / 100_000)
    err_ns_b = math.sqrt(g_b * b.var() / 100_000)
    need = 2 * g_a * DT / 2 * a.var() / targets[name] ** 2 / 1e6
    need_b = 2 * g_b * DT / 2 * b.var() / targets[name] ** 2 / 1e6
    print(f"{name}: σ {a.std():.4g} {unit} (fixed energy {b.std():.4g}); "
          f"τ_int {g_a * DT / 2000:.2f} ps, {g_b * DT / 2000:.3f} ps; error "
          f"of 1 ns {err_ns:.2g}, {err_ns_b:.2g} {unit}; for ±"
          f"{targets[name]:g} {unit}: {need:.2g} ns under CSVR, {need_b:.2g}"
          f" ns at fixed energy")
    print(f"    ratios: τ_int CSVR/fixed {g_a / g_b:.1f}; run CSVR/fixed "
          f"{need / need_b:.1f}; error of 1 ns relative to the mean "
          f"{err_ns / abs(a.mean()):.4f}; spread over mean "
          f"{a.std() / abs(a.mean()):.3f}")
    full = stats.standard_error(a)[1]
    own, spread = [], []
    for n in lengths:
        pieces = a[: len(a) // n * n].reshape(-1, n)
        own.append(np.mean([stats.standard_error(x)[1] for x in pieces]))
        spread.append(pieces.mean(axis=1).std(ddof=1) if len(pieces) >= 5
                      else np.nan)
    curves[name] = (np.array(own) / full, np.array(spread) / full)
    print(f"  {name}: pieces of " + ", ".join(
        f"{n * DT / 1000:g} ps {o:.2f} (spread {sp:.2f})"
        for n, o, sp in zip(lengths, *curves[name], strict=True))
        + " times the error of 2 ns")
print("U per atom: target 0.01 meV")

scr_runs = [np.load(ch14.RUNS / f"npt_scr_s{s}.npz") for s in range(3)]
taus, sigmas = [], []
for r in scr_runs:
    v = r["volume"][r["times"] >= 10000]
    g = stats.statistical_inefficiency(v)
    taus.append(g * (r["times"][1] - r["times"][0]) / 2)
    sigmas.append(v.std())
tau_v = np.mean(taus)
sig_v = np.mean(sigmas)
print(f"V under cell rescaling (Chapter 14, three runs of 300 ps): σ "
      f"{sig_v:.0f} Å³, τ_int {min(taus) / 1000:.1f} to "
      f"{max(taus) / 1000:.1f} ps (mean {tau_v / 1000:.1f}); error of 1 ns "
      f"{sig_v * math.sqrt(2 * tau_v / 1e6):.1f} Å³; for ±10 Å³ "
      f"{2 * tau_v * sig_v**2 / 100 / 1e6:.2g} ns")

shear = csvr["shear"][:, 0] * GPA
print(f"P_xy: σ {shear.std():.4f} GPa, τ_int "
      f"{stats.statistical_inefficiency(shear) * DT / 2:.0f} fs")

# (b) diffusion against temperature
runs_b = {"CSVR": [("long_csvr", 20), ("ensemble_csvr_130", 10),
                   ("ensemble_csvr_140", 10)],
          "Berendsen": [("ensemble_berendsen_130", 10),
                        ("ensemble_berendsen_140", 10)],
          "fixed energy": [("long_nve", 20)]}
points = {}
for label, names in runs_b.items():
    points[label] = [d_blocks(name, blocks) for name, blocks in names]
    for (name, blocks), (d, e, t, te) in zip(names, points[label],
                                             strict=True):
        print(f"{name}: D = {d:.3f} ± {e:.3f} e-4 Å²/fs ({blocks} blocks); "
              f"T = {t:.2f} ± {te:.2f} K")
(d1, e1, t1, _), (d2, e2, t2, _) = points["CSVR"][1:]
slope = (d2 - d1) / (t2 - t1)
slope_err = math.hypot(e1, e2) / (t2 - t1)
print(f"dD/dT from CSVR at 130 and 140 K: {slope:.4f} ± {slope_err:.4f} "
      f"e-4 Å²/fs per K")
d_c, e_c, t_c, _ = points["CSVR"][0]
d_n, e_n, t_n, _ = points["fixed energy"][0]
shift = slope * (t_c - t_n)
d_n_at = d_n + shift
e_n_at = math.hypot(e_n, slope_err * (t_c - t_n))
diff = d_c - d_n_at
diff_err = math.hypot(e_c, e_n_at)
energy_nve = nve["potential"] + nve["kinetic"]
sigma_e = np.load(RUNS / "long_csvr.npz")["potential"]
e_csvr = csvr["potential"] + csvr["kinetic"]
g_e = stats.statistical_inefficiency(e_csvr)
c_v = 2.347 * 256 * KB
t_err_40 = e_csvr.std() / c_v * math.sqrt(g_e * DT / 40000)
print(f"fixed energy run: T {t_n:.2f} K, {t_c - t_n:.2f} K below CSVR; the "
      f"mean energy of 40 ps of CSVR, with τ_int(E) "
      f"{g_e * DT / 2000:.2f} ps and spread {e_csvr.std():.3f} eV, is "
      f"uncertain by {t_err_40:.1f} K of temperature (C_V = 2.347 N k_B)")
d_blocks_err = math.sqrt(20) * e_c / d_c
print(f"D: one block of 100 ps fixes it to {100 * d_blocks_err:.1f}%; 1 ns to "
      f"{100 * e_c * math.sqrt(2) / d_c:.2f}%; a 1% target needs "
      f"{2 * (e_c * math.sqrt(2) / d_c / 0.01) ** 2 / 2:.2f} ns")
print(f"fixed energy moved to {t_c:.2f} K: {d_n_at:.3f} ± {e_n_at:.3f}; "
      f"CSVR less that {diff:+.3f} ± {diff_err:.3f} ({diff / diff_err:+.1f}"
      f" errors); ratio {d_c / d_n_at:.3f} ± "
      f"{d_c / d_n_at * math.hypot(e_c / d_c, e_n_at / d_n_at):.3f}")
for d, e, t, _ in points["Berendsen"]:
    line = d1 + slope * (t - t1)
    line_err = math.hypot(e1, slope_err * (t - t1))
    off = (d - line) / math.hypot(e, line_err)
    print(f"Berendsen at {t:.2f} K: {d:.3f} ± {e:.3f} against CSVR's line "
          f"{line:.3f} ± {line_err:.3f}: {off:+.1f} errors")

# Chapter 12: the gap between the temperatures of two species
mix = np.load(Path(ch12.RUNS, "mixture_masses.npz"))
m = mix["masses"]
v = mix["velocities"]
late = mix["frame_times"] >= mix["frame_times"][-1] - 10000
light = m < m.max()
gaps = []
for part in (light, ~light):
    k = statmech.kinetic_energy(m[part], v[late][:, part])
    gaps.append(2 * k / (3 * part.sum() * KB))
gap = gaps[0] - gaps[1]
g_mean, g_err, g_g = stats.standard_error(gap)
e_light = stats.standard_error(gaps[0])[1]
e_heavy = stats.standard_error(gaps[1])[1]
print(f"  correlation of the two temperatures "
      f"{np.corrcoef(gaps[0], gaps[1])[0, 1]:+.2f}; variances "
      f"{gaps[0].var():.1f} and {gaps[1].var():.1f} K², of the gap "
      f"{gap.var():.1f} K²; the two errors combined as if independent "
      f"{math.hypot(e_light, e_heavy):.2f} K")
print(f"Chapter 12's mixture, last 10 ps: light {gaps[0].mean():.2f} K, "
      f"heavy {gaps[1].mean():.2f} K; gap {g_mean:.2f} ± {g_err:.2f} K "
      f"(g = {g_g:.1f}), {g_mean / g_err:.1f} errors")

# Chapter 12: the error of a single run's heat capacity at fixed energy
rng = np.random.default_rng(5)
for t_name in (100, 115, 130, 145, 160):
    r = np.load(Path(ch12.RUNS, f"heat_{t_name}.npz"))
    k = r["kinetic"]

    def c_v(series):
        return statmech.heat_capacity_nve(series, N_FREE) / (256 * KB)

    whole = c_v(k)
    boot = stats.block_bootstrap([k], c_v, 100, 2000, rng)
    print(f"    distance from the slope 2.376: "
          f"{100 * abs(whole - 2.376) / 2.376:.1f}%")
    print(f"Chapter 12, run near {t_name} K: C_V = {whole:.3f} ± "
          f"{boot.std():.3f} N k_B ({100 * boot.std() / whole:.1f}%), "
          f"against the slope 2.376")

# Chapter 14: a pressure from 50 frames of one 100 ps run
print(f"Chapter 14's 0.0877 GPa: 50 frames 2 ps apart; a frame's spread "
      f"{csvr['pressure'].std() * GPA:.4f} GPa, 2 ps beyond τ_int, so the "
      f"error is about {csvr['pressure'].std() * GPA / math.sqrt(50):.4f} "
      f"GPa; the 2 ns run gives "
      f"{csvr['pressure'].mean() * GPA:.4f} ± "
      f"{stats.standard_error(csvr['pressure'] * GPA)[1]:.4f} GPa")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.2),
                                 gridspec_kw=dict(wspace=0.4))
ps = lengths * DT / 1000
for (name, (own, spread)), colour in zip(curves.items(),
                                          (ACCENT, OCHRE, OXBLOOD),
                                          strict=True):
    ax_a.loglog(ps, own, "o", ms=3, color=colour, label=f"${name}$")
    ax_a.loglog(ps, spread, "o", ms=3, mfc="none", color=colour, mew=0.7)
ax_a.loglog(ps, np.sqrt(2000 / ps), **REFERENCE_STYLE, lw=0.8)
ax_a.set_yticks([1, 2, 4, 8])
ax_a.set_yticks([], minor=True)
ax_a.yaxis.set_major_formatter(plt.ScalarFormatter())
ax_a.set_xlabel("length of run / ps")
ax_a.set_ylabel("error, relative to 2 ns")
ax_a.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax_a, "a")

for label, colour, marker in (("CSVR", ACCENT, "o"),
                              ("Berendsen", OCHRE, "s"),
                              ("fixed energy", REFERENCE, "D")):
    d, e, t, te = np.array(points[label]).T
    ax_b.errorbar(t, d, yerr=e, xerr=te, fmt=marker, ms=3, lw=0.8,
                  color=colour, capsize=0, label=label)
grid = np.linspace(129, 141, 10)
ax_b.plot(grid, d1 + slope * (grid - t1), **REFERENCE_STYLE, lw=0.8)
ax_b.set_xlabel(r"$T$ / K")
ax_b.set_ylabel(r"$D$ / $10^{-4}$ \AA$^2$ fs$^{-1}$")
ax_b.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "observables.pdf")))
