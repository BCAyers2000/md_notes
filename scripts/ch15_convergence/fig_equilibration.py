"""Figure fig:cv-equilibration: where a run begins.

(a) The series of Section 15.2 (c = 0.95, samples 10 fs apart, 50 ps)
with the decaying offset 3e^{−t/2 ps} added (dashed), and the start that
maximises the effective number of samples (dotted). (b) That number,
(n − t₀)/g(t₀), against the start tried. (c) Eight runs of the liquid
melting from the face-centred cubic lattice at 135 K under CSVR (runs.py,
melt_s0 to melt_s7): the potential energy per atom (grey), their mean
(teal), the mean of the 2 ns run (dashed), and each run's start found
from its own U (ticks).

Prints the bias of the mean with and without the discard, against
Δτ_r/T, and over 200 such series where the start falls, the offset it
leaves and how often one error bar holds the true mean; for each
melting run the start found from U, T and P and the means; the start
found in Chapter 14's runs at constant pressure from V; and the start
and mean of Chapter 11's healthy water run.
"""

import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, GPA, N_FREE, RUNS, ou

from mdlab import units, viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, REFERENCE, REFERENCE_STYLE, THRESHOLD_STYLE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "ch14_pressure"))
import ch14  # noqa: E402

rng = np.random.default_rng(3)
n, delta, tau_r = 5000, 3.0, 200  # samples; offset in units of σ; 2 ps
t = np.arange(n) * DT
offset = delta * np.exp(-np.arange(n) / tau_r)
series = ou(n, 0.95, rng) + offset
t0, g0, neff0 = stats.detect_equilibration(series)
tried = np.arange(0, n - 1, 10)
neff = [(n - s) / stats.statistical_inefficiency(series[s:]) for s in tried]
mean_all, error_all, _ = stats.standard_error(series)
mean_cut, error_cut, _ = stats.standard_error(series[t0:])
print(f"offset 3σ decaying over 2 ps in a 50 ps run: start found at "
      f"{t0 * DT / 1000:.2f} ps ({t0 / tau_r:.1f} decay times), g "
      f"{g0:.1f}, {neff0:.0f} effective samples")
print(f"  mean from the start {mean_all:.3f} ± {error_all:.3f}; after the "
      f"discard {mean_cut:.3f} ± {error_cut:.3f}; the offset's own share "
      f"Δτ_r/T = {delta * tau_r / n:.3f} from the start, "
      f"{offset[t0:].mean():.4f} after the discard")

stops, left, covered, covered_all = [], [], [], []
for _ in range(200):
    trial = ou(n, 0.95, rng) + offset
    s, _, _ = stats.detect_equilibration(trial, nskip=10)
    m, e, _ = stats.standard_error(trial[s:])
    m_all, e_all, _ = stats.standard_error(trial)
    stops.append(s / tau_r)
    left.append(offset[s:].mean() / e)
    covered.append(abs(m) <= e)
    covered_all.append(abs(m_all) <= e_all)
print(f"  200 trials: start found between {min(stops):.1f} and "
      f"{max(stops):.1f} decay times (median {np.median(stops):.1f}); "
      f"offset left at most {max(left):.2f} errors (median "
      f"{np.median(left):.2f}); one error bar holds the true mean in "
      f"{np.mean(covered):.2f} of trials after the discard, "
      f"{np.mean(covered_all):.2f} without it")

long_u = np.load(RUNS / "long_csvr.npz")["potential"] / 256
mean_long, error_long, _ = stats.standard_error(long_u)
print(f"2 ns run: U/N = {1000 * mean_long:.3f} ± {1000 * error_long:.3f} "
      f"meV")
with (contextlib.redirect_stdout(io.StringIO()),
      contextlib.redirect_stderr(io.StringIO())):
    from pymbar import timeseries
    default = timeseries.detect_equilibration(series)[0]
    careful = timeseries.detect_equilibration(series, fast=False)[0]
print(f"pymbar on the series of (a): start {careful} samples with "
      f"fast=False (mdlab {t0}), {default} with its default")
melts, starts, from_start, after = [], [], [], []
g_after = []
for seed in range(8):
    r = np.load(RUNS / f"melt_s{seed}.npz")
    u = r["potential"] / 256
    temp = 2 * r["kinetic"] / (N_FREE * units.KB)
    p = r["pressure"] * GPA
    s_u = stats.detect_equilibration(u, nskip=20)[0]
    s_t = stats.detect_equilibration(temp, nskip=20)[0]
    s_p = stats.detect_equilibration(p, nskip=20)[0]
    m_all, e_all, _ = stats.standard_error(u)
    m_cut, e_cut, g_cut = stats.standard_error(u[s_u:])
    g_after.append(g_cut)
    melts.append(u)
    starts.append(s_u)
    from_start.append(m_all)
    after.append((m_cut, e_cut))
    print(f"melt s{seed}: start from U {s_u * DT / 1000:.2f} ps, from T "
          f"{s_t * DT / 1000:.2f} ps, from P {s_p * DT / 1000:.2f} ps; U/N "
          f"from the start {1000 * m_all:.3f} ± {1000 * e_all:.3f} meV, "
          f"after {1000 * m_cut:.3f} ± {1000 * e_cut:.3f} meV")
print(f"  U/N on the lattice {1000 * melts[0][0]:.3f} meV; starts from U "
      f"{min(starts) * DT / 1000:.2f} to {max(starts) * DT / 1000:.2f} ps")
after = np.array(after)
print(f"  means from the start average {1000 * np.mean(from_start):.3f} meV;"
      f" after the discards {1000 * after[:, 0].mean():.3f} meV, scattering "
      f"by {1000 * after[:, 0].std(ddof=1):.3f} meV against error bars of "
      f"{1000 * after[:, 1].min():.3f} to {1000 * after[:, 1].max():.3f} "
      f"(mean {1000 * after[:, 1].mean():.3f}) meV")
print(f"  g after the starts {min(g_after):.0f} to {max(g_after):.0f}; "
      f"the 2 ns run's {stats.statistical_inefficiency(long_u):.0f}; "
      f"independent samples in 100 ps of U "
      f"{10000 / stats.statistical_inefficiency(long_u):.0f}")
mean_curve = np.mean(melts, axis=0)
print(f"  mean of the eight within 1 meV of the 2 ns mean from "
      f"{np.argmax(np.abs(mean_curve - mean_long) < 1e-3) * DT / 1000:.2f}"
      f" ps")

for kind in ("scr", "berendsen", "mtk"):
    found = []
    for seed in range(3):
        r = np.load(ch14.RUNS / f"npt_{kind}_s{seed}.npz")
        dt_record = r["times"][1] - r["times"][0]
        s = stats.detect_equilibration(r["volume"],
                                       nskip=max(1, len(r["volume"]) // 300))
        if kind == "scr" and seed == 0:
            v = r["volume"]
            candidates = np.arange(0, len(v) - 1, 10)
            curve = np.array([(len(v) - x) / stats.statistical_inefficiency(
                v[x:]) for x in candidates])
            at = {ps: curve[np.searchsorted(candidates * dt_record,
                                            ps * 1000)]
                  for ps in (0, 100, 195)}
            peak = candidates[curve.argmax()] * dt_record / 1000
            print(f"Chapter 14, scr s0: n_eff(t0) {at[0]:.1f} at 0, "
                  f"{at[100]:.1f} at 100 ps, {at[195]:.1f} at 195 ps; largest "
                  f"{curve.max():.1f} at {peak:.0f} ps")
        found.append((s[0] * dt_record / 1000,
                      len(r["volume"]) / stats.statistical_inefficiency(
                          r["volume"])))
    print(f"Chapter 14, {kind}: start found from V at "
          + ", ".join(f"{x:.1f} ps ({ne:.0f} effective samples in the run)"
                      for x, ne in found) + "; 10 ps discarded there")

water = np.load(Path(viz.THEORY, "data", "ch11_constraints", "runs",
                     "rigid_healthy.npz"))
dt_w = water["times"][1] - water["times"][0]
for name in ("kinetic", "potential"):
    report = stats.convergence_report(water[name], dt_w)
    print(f"Chapter 11's water, {name}: start {report.start:.0f} fs, mean "
          f"{report.mean:.4f} ± {report.error:.4f} eV, converged "
          f"{report.converged}; whole-run mean {water[name].mean():.4f}")
per_molecule = stats.convergence_report(water["potential"], dt_w)
print(f"  per molecule U {per_molecule.mean / 64:.5f} ± "
      f"{per_molecule.error / 64:.5f} eV")

viz.use_style(notebook=False)
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(viz.FULL, 2.0), gridspec_kw=dict(wspace=0.5))
ax_a.plot(t / 1000, series, color=ACCENT, lw=0.5)
ax_a.plot(t / 1000, offset, **REFERENCE_STYLE, lw=0.8)
ax_a.axvline(t0 * DT / 1000, **THRESHOLD_STYLE)
ax_a.set_xlabel("time / ps")
ax_a.set_ylabel(r"$x$")
viz.panel_tag(ax_a, "a")

ax_b.plot(tried * DT / 1000, neff, color=ACCENT, lw=0.9)
ax_b.axvline(t0 * DT / 1000, **THRESHOLD_STYLE)
ax_b.set_xlabel(r"start $t_0$ / ps")
ax_b.set_ylabel(r"$n_{\mathrm{eff}}$ after $t_0$")
viz.panel_tag(ax_b, "b")

time_m = np.arange(len(melts[0])) * DT / 1000
for u in melts:
    ax_c.plot(time_m, 1000 * u, color=REFERENCE, lw=0.3, alpha=0.6)
ax_c.plot(time_m, 1000 * mean_curve, color=ACCENT, lw=0.9)
ax_c.axhline(1000 * mean_long, **REFERENCE_STYLE, lw=0.8)
top = 1000 * max(u.max() for u in melts)
for s in starts:
    ax_c.plot([s * DT / 1000] * 2, [top - 0.8, top], color="black", lw=0.6)
ax_c.set_xlim(0, 30)
ax_c.set_xlabel("time / ps")
ax_c.set_ylabel(r"$U/N$ / meV")
viz.panel_tag(ax_c, "c")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "equilibration.pdf")))
