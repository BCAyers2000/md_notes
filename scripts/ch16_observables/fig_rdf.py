"""Figure fig:ob-rdf: how atoms arrange themselves.

(a) The radial distribution function of liquid argon at 135 K (Chapter
15's 2 ns CSVR run, every 2 ps) against an ideal gas of the same number
in the same cell (dashed, from 1.5 Å, where its bins hold enough
pairs). (b) The fcc crystal at 21 K (runs.py,
crystal_nve), with the distances of its first shells (dotted). (c) The
running coordination number of the liquid and the crystal, with the
liquid's first minimum (dotted).

Prints the first peak and minimum, the coordination numbers, the far
value against 1 − 1/N, the partner density in the corners of the
256- and 2048-atom cells, the energy and the pressure from g(r) against
the run, the crystal's shells, and the largest difference from ASE's
get_rdf on the same frames.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.geometry.rdf import get_rdf
from ch16 import GPA, RUNS, ch12, ch15

from mdlab import units, viz
from mdlab.analysis import stats, structure
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, THRESHOLD_STYLE

N = 256
liquid = np.load(ch15.RUNS / "long_csvr.npz")
h = liquid["cell"]
side = h[0, 0]
volume = np.linalg.det(h)
rho = N / volume
r_max = side / 2 - 1e-6
frames = liquid["positions"][1::4].astype(float)  # every 2 ps
x, g = structure.rdf(frames, h, r_max, 232)
dr = x[1] - x[0]
peak = np.argmax(g)
inside = (x > 4.5) & (x < 6.5)
low = np.flatnonzero(inside)[np.argmin(g[inside])]
n_liquid = structure.running_coordination(x, g, rho)
print(f"liquid: L = {side:.3f} Å, ρ = {rho:.5f} Å⁻³, {len(frames)} frames, "
      f"bins of {dr:.4f} Å")
print(f"first peak g = {g[peak]:.3f} at {x[peak]:.3f} Å; first minimum "
      f"g = {g[low]:.3f} at {x[low]:.3f} Å; coordination there "
      f"{n_liquid[low]:.2f}")
far = x > 10.0
print(f"mean g beyond 10 Å {g[far].mean():.4f}; 1 − 1/N = {1 - 1 / N:.4f}")


def corner_density(names, n_atoms, n_bins):
    """Partners beyond L/2, in the cube's corners, over ρ times their volume.

    Each atom has N − 1 partners in the cell; those not within the sphere
    of radius L/2 lie in the corners. Returns the ratio frame by frame.
    """
    ratios = []
    for name in names:
        run = np.load(ch15.RUNS / name)
        cell = run["cell"]
        length, vol = cell[0, 0], np.linalg.det(cell)
        density = n_atoms / vol
        r, per_frame = structure.rdf(run["positions"][1:].astype(float), cell,
                                     length / 2 - 1e-6, n_bins,
                                     per_frame=True)
        within = np.array([structure.running_coordination(r, f, density)[-1]
                           for f in per_frame])
        corners = vol - 4 / 3 * np.pi * (length / 2) ** 3
        ratios.append((n_atoms - 1 - within) / (density * corners))
    return np.concatenate(ratios)


# The corner-density comparison uses every saved frame of the large cells.
# Run it explicitly when checking the finite-size discussion.
if "--check-corners" in sys.argv:
    for n_atoms, n_bins in ((256, 232), (2048, 464)):
        names = [f"size_{n_atoms}.npz", f"size_{n_atoms}_b.npz"]
        ratio, ratio_err, _ = stats.standard_error(corner_density(names, n_atoms,
                                                                  n_bins))
        print(f"{n_atoms} atoms: partner density in the corners {ratio:.6f} ± "
              f"{ratio_err:.6f} of ρ; 1 − 1/N = {1 - 1 / n_atoms:.6f}; the "
              f"deficit times N, {(1 - ratio) * n_atoms:.3f} ± "
              f"{ratio_err * n_atoms:.3f}, against S(0) (fig_sq.py)")

# The energy and the pressure from g(r), on finer bins.
xf, gf = structure.rdf(frames, h, r_max, 1163)
edges = np.append(xf - 0.5 * (xf[1] - xf[0]), xf[-1] + 0.5 * (xf[1] - xf[0]))
shell = structure.shell_volumes(edges)
phi, dphi = ch12.model(h).pair(xf)
partners = rho * gf * shell  # partners per atom in each bin
u_rdf = 0.5 * np.sum(partners * phi)
u_run, u_err, _ = stats.standard_error(liquid["potential"] / N)
w_rdf = -0.5 * N * np.sum(partners * xf * dphi)
p_rdf = (2 * liquid["kinetic"].mean() + w_rdf) / (3 * volume) * GPA
p_run, p_err, _ = stats.standard_error(liquid["pressure"] * GPA)
x5 = x
edges5 = np.append(x5 - 0.5 * dr, x5[-1] + 0.5 * dr)
u_coarse = 0.5 * np.sum(rho * g * structure.shell_volumes(edges5)
                        * ch12.model(h).pair(x5)[0])
print(f"on bins of {dr:.3f} Å instead: U/N = {u_coarse * 1000:.3f} meV, "
      f"{(u_coarse - u_run) / u_err:.1f} errors from the run")
print(f"U/N from g(r) {u_rdf * 1000:.3f} meV against the run's "
      f"{u_run * 1000:.3f} ± {u_err * 1000:.3f} meV "
      f"({(u_rdf - u_run) / u_err:.1f} errors)")
print(f"P from g(r) {p_rdf:.4f} GPa against the run's {p_run:.4f} ± "
      f"{p_err:.4f} GPa ({(p_rdf - p_run) / p_err:.1f} errors)")

# Checked against ASE on 50 of the frames.
images = [Atoms(f"Ar{N}", positions=f, cell=h.T, pbc=True)
          for f in frames[:50]]
ours = structure.rdf(frames[:50], h, r_max, 232)[1]
theirs = get_rdf(images, r_max, 232, no_dists=True)
print(f"largest difference from ASE's get_rdf: "
      f"{np.abs(ours - theirs).max():.1e}")

# The crystal at 20 K.
crystal = np.load(RUNS / "crystal_nve.npz")
hc = crystal["cell"]
rho_c = N / np.linalg.det(hc)
xc, gc = structure.rdf(crystal["positions"].astype(float), hc,
                       hc[0, 0] / 2 - 1e-6, 400)
n_crystal = structure.running_coordination(xc, gc, rho_c)
a = hc[0, 0] / 4
shells = a / np.sqrt(2) * np.sqrt(np.arange(1, 6))
t_c = 2 * crystal["kinetic"].mean() / (3 * (N - 1) * units.KB)
cpos = crystal["positions"].astype(float)
wander = cpos - cpos.mean(axis=0)
print(f"crystal atoms wander from their mean positions by "
      f"{np.sqrt(np.mean(wander**2)):.3f} Å along each axis (root mean "
      f"square)")
print(f"crystal density {N / np.linalg.det(hc):.5f} Å⁻³ against the "
      f"liquid's {rho:.5f}")
print(f"crystal: a = {a:.5f} Å, T = {t_c:.2f} K; shells at "
      f"{', '.join(f'{s:.3f}' for s in shells)} Å")
gaps = 0.5 * (shells[:-1] + shells[1:])
counts = n_crystal[np.searchsorted(xc, gaps)]
print("crystal coordination out to the gaps between shells: "
      + ", ".join(f"{c:.2f}" for c in counts)
      + "; shell sizes " + ", ".join(
          f"{c:.2f}" for c in np.diff(np.concatenate([[0], counts]))))
print(f"crystal first peak height {gc.max():.2f}")

# An ideal gas of N atoms in the liquid's cell.
rng = np.random.default_rng(0)
gas = rng.uniform(0, side, (400, N, 3))
xg, gg = structure.rdf(gas, h, r_max, 232)
print(f"ideal gas: mean g {gg[xg > 2].mean():.4f} ± "
      f"{gg[xg > 2].std() / np.sqrt((xg > 2).sum()):.4f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.42))
ax = axes[0]
shown = xg > 1.5  # nearer bins hold too few pairs to show
ax.plot(xg[shown], gg[shown], **REFERENCE_STYLE, lw=0.8, label="ideal gas")
ax.plot(x, g, color=ACCENT, lw=1.0, label="liquid")
ax.set_xlim(0, side / 2)
ax.set_ylim(0, 2.9)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$g(r)$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
for s in shells:
    ax.axvline(s, **THRESHOLD_STYLE)
ax.plot(xc, gc, color=OCHRE, lw=0.8)
ax.set_xlim(0, hc[0, 0] / 2)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$g(r)$")
viz.panel_tag(ax, "b")

ax = axes[2]
ax.plot(x, n_liquid, color=ACCENT, lw=1.0, label="liquid")
ax.plot(xc, n_crystal, color=OCHRE, lw=0.8, label="crystal")
ax.axvline(x[low], **THRESHOLD_STYLE)
ax.set_xlim(0, 8)
ax.set_ylim(0, 60)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$n_{\mathrm c}(r)$")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "rdf.pdf")))
