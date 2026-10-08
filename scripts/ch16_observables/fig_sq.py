"""Figure fig:ob-sq: the structure factor of liquid argon.

(a) S(G) of the 256-atom liquid at 135 K from the positions, on every
wave vector of the cell up to 12 reciprocal vectors along each axis
(Chapter 15's long_csvr run, 250 frames 8 ps apart), against the
transform of g(r) cut at half the cell (line). (b) S(G) at the smallest
wave vectors of the 2048- and 1372-atom runs of Chapter 15 (size_*, two
runs each, frames every ps), each with the standard error of its frames'
values, and a straight line in G² fitted below 0.6 Å⁻¹ and extrapolated
to G = 0, against ρk_BTκ_T from Chapter 14's
compressibility (band, its error).

Prints the first peak of S(G), the two routes at it and at the smallest
G, the fitted S(0) with its error, and ρk_BTκ_T.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import KT, ch15

from mdlab import units, viz
from mdlab.analysis import stats, structure
from mdlab.viz import ACCENT, OCHRE, REFERENCE

KAPPA_T, KAPPA_ERR = 1.46, 0.11  # GPa⁻¹, Chapter 14 from three volumes
V_KAPPA = 12306.0  # Å³, the mean volume at which Chapter 14 took dP/dV
N = 256

liquid = np.load(ch15.RUNS / "long_csvr.npz")
h = liquid["cell"]
rho = N / np.linalg.det(h)
frames = liquid["positions"][1::16].astype(float)
q, s = structure.structure_factor(frames, h, 12)
keep = q < 12 * 2 * np.pi / h[0, 0] + 1e-9
q, s = q[keep], s[keep]
x, g = structure.rdf(liquid["positions"][1::4].astype(float), h,
                     h[0, 0] / 2 - 1e-6, 232)
crest = [x[(x > lo) & (x < hi)][np.argmax(g[(x > lo) & (x < hi)])]
         for lo, hi in ((3.0, 5.0), (5.5, 8.5), (8.5, 11.6))]
print("peaks of g(r) at " + ", ".join(f"{c:.2f}" for c in crest) + " Å; "
      f"mean spacing {np.mean(np.diff(crest)):.2f} Å, 2π over it "
      f"{2 * np.pi / np.mean(np.diff(crest)):.3f} Å⁻¹")
q_line = np.linspace(0.2, q.max(), 400)
s_line = structure.structure_factor_from_rdf(x, g, rho, q_line)
s_at = structure.structure_factor_from_rdf(x, g, rho, q)
top = np.argmax(s)
print(f"{len(frames)} frames; first peak S = {s[top]:.3f} at G = "
      f"{q[top]:.3f} Å⁻¹ ({2 * np.pi / q[top]:.3f} Å); from g(r) "
      f"{s_at[top]:.3f}")
print(f"smallest G {q[0]:.4f} Å⁻¹: from positions {s[0]:.4f}, from g(r) "
      f"{s_at[0]:.4f}")
big = q > 1.0
print(f"above 1 Å⁻¹ the two routes differ by at most "
      f"{np.abs(s - s_at)[big].max():.3f}")
dip = q_line < 1.0
lowest = np.argmin(s_line[dip])
print(f"below 1 Å⁻¹ the transform of g(r) cut at L/2 falls to "
      f"{s_line[dip][lowest]:.3f} at G = {q_line[dip][lowest]:.2f} Å⁻¹")

# The 2048-atom g(r), cut at the 256-atom cell's L/2 and at its own.
big_run = np.load(ch15.RUNS / "size_2048.npz")
h_big = big_run["cell"]
x_big, g_big = structure.rdf(big_run["positions"][1::5].astype(float), h_big,
                             h_big[0, 0] / 2 - 1e-6, 464)
rho_big = 2048 / np.linalg.det(h_big)
for cut in (h[0, 0] / 2, h_big[0, 0] / 2):
    kept = x_big < cut
    at_peak = structure.structure_factor_from_rdf(
        x_big[kept], g_big[kept], rho_big, np.array([q[top]]))[0]
    print(f"2048-atom g(r) cut at {cut:.1f} Å: S at G = {q[top]:.3f} Å⁻¹ is "
          f"{at_peak:.3f}")

kt_kappa = rho * KT * KAPPA_T * units.EV_PER_A3_TO_GPA  # κ_T in Å³/eV
kt_err = kt_kappa * KAPPA_ERR / KAPPA_T
print(f"this run's volume {np.linalg.det(h):.0f} Å³ against Chapter 14's "
      f"{V_KAPPA:.0f}: {100 * (np.linalg.det(h) / V_KAPPA - 1):.1f}% larger")
print(f"ρ k_BT κ_T = {kt_kappa:.4f} ± {kt_err:.4f} (κ_T = {KAPPA_T} ± "
      f"{KAPPA_ERR} GPa⁻¹)")

small = {}
for n_atoms in (1372, 2048):
    per_frame = []
    for tag in ("", "_b"):
        run = np.load(ch15.RUNS / f"size_{n_atoms}{tag}.npz")
        hs = run["cell"]
        for frame in run["positions"][1:].astype(float):
            qs, ss = structure.structure_factor(frame, hs, 3)
            per_frame.append(ss)
    per_frame = np.array(per_frame)
    half = len(per_frame) // 2
    means, errors = [], []
    for k in range(len(qs)):
        runs_k = [stats.standard_error(per_frame[:half, k]),
                  stats.standard_error(per_frame[half:, k])]
        means.append(0.5 * (runs_k[0][0] + runs_k[1][0]))
        errors.append(0.5 * np.hypot(runs_k[0][1], runs_k[1][1]))
    small[n_atoms] = (qs, np.array(means), np.array(errors))
xs = np.concatenate([small[n][0] for n in small])
ys = np.concatenate([small[n][1] for n in small])
es = np.concatenate([small[n][2] for n in small])
fit = xs < 0.6
coef, cov = np.polyfit(xs[fit] ** 2, ys[fit], 1, w=1 / es[fit], cov="unscaled")
s0, s0_err = coef[1], np.sqrt(cov[1, 1])
chi2 = np.sum(((np.polyval(coef, xs[fit] ** 2) - ys[fit]) / es[fit]) ** 2)
print(f"S(0) fitted to {fit.sum()} values below 0.6 Å⁻¹: {s0:.4f} ± "
      f"{s0_err:.4f}, slope {coef[0]:.4f} Å²; χ² = {chi2:.1f} for "
      f"{fit.sum() - 2} degrees of freedom; "
      f"{(s0 - kt_kappa) / np.hypot(s0_err, kt_err):.1f} combined errors "
      f"from ρk_BTκ_T")
scale = np.sqrt(chi2 / (fit.sum() - 2))
print(f"error scaled by √(χ²/dof) = {scale:.2f}: S(0) = {s0:.4f} ± "
      f"{s0_err * scale:.4f}, "
      f"{(s0 - kt_kappa) / np.hypot(s0_err * scale, kt_err):.1f} combined "
      f"errors")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.32))
ax = axes[0]
ax.plot(q_line, s_line, color=REFERENCE, lw=0.8, ls="--",
        label=r"from $g(r)$")
ax.plot(q, s, "o", color=ACCENT, ms=1.8, label="from positions")
ax.set_xlim(0, q.max())
ax.set_xlabel(r"$G$ / \AA$^{-1}$")
ax.set_ylabel(r"$S(G)$")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.axhspan(kt_kappa - kt_err, kt_kappa + kt_err,
           color=viz.tint(REFERENCE, 0.25), lw=0)
grid = np.linspace(0, 0.6, 50)
ax.plot(grid, np.polyval(coef, grid**2), color=REFERENCE, lw=0.8, ls="--")
for n_atoms, colour in ((1372, OCHRE), (2048, ACCENT)):
    qs, ss, es_ = small[n_atoms]
    ax.errorbar(qs, ss, es_, fmt="o", ms=2.5, color=colour, lw=0.7,
                label=f"{n_atoms} atoms")
ax.set_xlim(0, 0.62)
ax.set_ylim(0, 0.12)
ax.set_xlabel(r"$G$ / \AA$^{-1}$")
ax.set_ylabel(r"$S(G)$")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "sq.pdf")))
