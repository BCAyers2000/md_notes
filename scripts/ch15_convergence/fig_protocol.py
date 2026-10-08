"""Figure fig:cv-protocol: a protocol, stage by stage.

256 argon atoms placed at random in the liquid's cell (runs.py,
protocol). (a) Steepest descent: the largest force on any atom after each
accepted move, against the tolerance 0.05 eV/Å (dotted). (b) 50 ps of
CSVR at fixed volume from velocities drawn at 135 K: the temperature
(thin) and its running mean (thick), against 135 K (dotted). (c) 1 ns
of stochastic cell rescaling at 0.1 GPa: the volume, the start found
(dotted), and the mean after it (dashed). (d) The same for the potential
energy per atom.

Prints the minimisation, the report of convergence_report for V, P, T
and U of the stage at constant pressure after 300 ps and after 1 ns, and
three tests of the report: how often its error bar holds the true mean of
series of known mean with 25 to 200 effective samples; on
the twenty pieces of 100 ps of the 2 ns run, how often a piece declared
converged holds the 2 ns mean within one and two of its errors; and the
same on 200 series of known mean with a decaying offset.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, GPA, N_FREE, RUNS, T_LIQUID, ou

from mdlab import units, viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, REFERENCE, REFERENCE_STYLE, THRESHOLD_STYLE

KB = units.KB
run = np.load(RUNS / "protocol.npz")
energies, forces = run["min_energy"], run["min_force"]
print(f"steepest descent: {len(energies) - 1} accepted moves; energy from "
      f"{run['start_energy']:.3g} eV to {energies[-1]:.3f} eV; largest force "
      f"from {forces[0]:.3g} to {forces[-1]:.3f} eV/Å")

t_nvt = run["nvt_times"] / 1000
temp_nvt = 2 * run["nvt_kinetic"] / (N_FREE * KB)
window = np.convolve(temp_nvt, np.ones(100) / 100, mode="valid")
print(f"fixed volume: T starts at {temp_nvt[0]:.1f} K, lowest "
      f"{temp_nvt[t_nvt <= 1].min():.1f} K in the first ps; its mean over "
      f"1 ps first reaches 130 K by "
      f"{t_nvt[np.argmax(window > 130) + 99]:.1f} ps;"
      f" mean over the last 25 ps {temp_nvt[t_nvt >= 25].mean():.2f} ± "
      f"{stats.standard_error(temp_nvt[t_nvt >= 25])[1]:.2f} K")
accel = forces[0] * units.FORCE_TO_ACCEL / 39.948
print(f"the largest force at the start would move its atom by about "
      f"{0.5 * accel * DT**2:.1e} Å in one step of {DT:g} fs")

t_npt = run["npt_times"] / 1000
series = {
    "V": (run["npt_volume"], "Å³"),
    "P": (run["npt_pressure"] * GPA, "GPa"),
    "T": (2 * run["npt_kinetic"] / (N_FREE * KB), "K"),
    "U": (run["npt_potential"], "eV"),
}
reports = {}
for length in (300, 1000):
    keep = t_npt <= length
    for name, (a, unit) in series.items():
        report = stats.convergence_report(a[keep], DT)
        reports[name] = report
        print(f"--- {name} ({unit}), constant pressure, first {length} ps:")
        print(report)
        if not report.converged:
            need = stats.MIN_EFFECTIVE * report.inefficiency * DT / 1000
            print(f"    {stats.MIN_EFFECTIVE} effective samples need "
                  f"{need:.0f} ps after the start")

rng = np.random.default_rng(7)
for effective in (25, 50, 100, 200):
    n = effective * 39
    trials = ou(n, 0.95, rng, shape=(300,))
    held, held_plain = 0, 0
    for k in range(300):
        report = stats.convergence_report(trials[:, k], DT,
                                          nskip=max(1, n // 100))
        held += abs(report.mean) <= report.error
        held_plain += (abs(trials[:, k].mean())
                       <= stats.standard_error(trials[:, k])[1])
    print(f"series with {effective} effective samples and no offset, 300 "
          f"of them: one error bar of the report holds the true mean in "
          f"{held / 300:.2f}; without the search for a start, in "
          f"{held_plain / 300:.2f}")

long_run = np.load(RUNS / "long_csvr.npz")
whole = {"U": long_run["potential"],
         "T": 2 * long_run["kinetic"] / (N_FREE * KB),
         "P": long_run["pressure"] * GPA}
for name, a in whole.items():
    truth = a.mean()
    pieces = a[: len(a) // 20 * 20].reshape(20, -1)
    verdicts, within1, within2 = [], [], []
    for piece in pieces:
        report = stats.convergence_report(piece, DT)
        verdicts.append(report.converged)
        if report.converged:
            within1.append(abs(report.mean - truth) <= report.error)
            within2.append(abs(report.mean - truth) <= 2 * report.error)
    print(f"{name}, twenty pieces of 100 ps: {sum(verdicts)} declared "
          f"converged; of those {sum(within1)} hold the 2 ns mean within one "
          f"error and {sum(within2)} within two")

offset = 3 * np.exp(-np.arange(20_000) / 200)
counts = np.zeros(3, int)
for _ in range(200):
    report = stats.convergence_report(ou(20_000, 0.95, rng) + offset, DT,
                                      nskip=50)
    if report.converged:
        counts += [1, abs(report.mean) <= report.error,
                   abs(report.mean) <= 2 * report.error]
print(f"200 series of 200 ps with the offset 3e^(−t/2 ps): {counts[0]} "
      f"declared converged; of those {counts[1]} ({counts[1] / counts[0]:.2f})"
      f" hold the true mean within one error and {counts[2]} "
      f"({counts[2] / counts[0]:.2f}) within two")

viz.use_style(notebook=False)
fig, axes = plt.subplots(2, 2, figsize=(viz.FULL, 3.6),
                         gridspec_kw=dict(wspace=0.4, hspace=0.75))
ax = axes[0, 0]
ax.semilogy(np.arange(len(forces)), forces, color=ACCENT, lw=0.9)
ax.axhline(0.05, **THRESHOLD_STYLE)
ax.set_xlabel("accepted move")
ax.set_ylabel(r"largest force / eV \AA$^{-1}$")
viz.panel_tag(ax, "a")

ax = axes[0, 1]
running = np.cumsum(temp_nvt) / np.arange(1, len(temp_nvt) + 1)
ax.plot(t_nvt, temp_nvt, color=REFERENCE, lw=0.3)
ax.plot(t_nvt, running, color=ACCENT, lw=1.0)
ax.axhline(T_LIQUID, **THRESHOLD_STYLE)
ax.set_xlabel("time / ps")
ax.set_ylabel(r"$T$ / K")
viz.panel_tag(ax, "b")

for ax, name, scale, label, tag in (
        (axes[1, 0], "V", 1.0, r"$V$ / \AA$^3$", "c"),
        (axes[1, 1], "U", 1000 / 256, r"$U/N$ / meV", "d")):
    a = series[name][0] * scale
    report = reports[name]
    ax.plot(t_npt, a, color=ACCENT, lw=0.3)
    ax.axvline(report.start / 1000, **THRESHOLD_STYLE)
    ax.axhline(report.mean * scale, **REFERENCE_STYLE, lw=0.8)
    ax.set_xlabel("time / ps")
    ax.set_ylabel(label)
    viz.panel_tag(ax, tag)

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "protocol.pdf")))
