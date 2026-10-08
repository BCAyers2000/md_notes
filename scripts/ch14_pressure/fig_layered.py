"""Figure fig:pr-layered: shape matters for a layered solid.

The layered model solid of ch14.py at 40 K under Berendsen's barostat at
P₀ = 0 (runs.py, layered_*): guests placed between the sheets, their
reach grown from 1.6 to 2.5 Å over the first 10 ps and then held for 30
ps. (a) The change of the height c (solid) and of the in-plane width a
(dashed) of the cell, under isotropic, semi-isotropic and anisotropic
coupling. (b) The pressure in the plane, ½(P_xx + P_yy), and across it,
P_zz, averaged over the last 10 ps, for each coupling.

Prints the strains and stresses, and C₁₁ and C₃₃ of the perfect sheets
at T = 0 with the cell of the relaxed solid, the stiffness within and
across the sheets.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch14 import GPA, NX, R_CUT, RUNS, SHEETS, layered_sheets, layered_table

from mdlab import cell, md, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

COUPLE = (("isotropic", OXBLOOD), ("semi-isotropic", ACCENT),
          ("anisotropic", OCHRE))
PIECE = 0.5  # ps

relaxed = np.load(RUNS / "layered_relaxed.npz")
h_relaxed = relaxed["cell"]
print(f"relaxed sheets: cell {np.diag(h_relaxed).round(3)} Å, sheet spacing "
      f"{h_relaxed[2, 2] / SHEETS:.3f} Å")

# a guest in a hollow, midway between sheets, to its nearest sheet atoms
reach = math.hypot(h_relaxed[0, 0] / NX / math.sqrt(3),
                   h_relaxed[2, 2] / SHEETS / 2)
pulling = [s for s in np.arange(1.6, 2.51, 0.1) if reach > 2 ** (1 / 6) * s]
print(f"guest to nearest sheet atoms {reach:.2f} Å, beyond the pair minimum "
      f"2^(1/6)σ (attracting) for σ up to {max(pulling):.1f} Å")

# the stiffness of the perfect sheets in the relaxed cell, at T = 0
r0, types, h0 = layered_sheets()
stretch = h_relaxed @ np.linalg.inv(h0)
r0, h0 = r0 @ stretch.T, h_relaxed


def tensor(axis, e):
    strain = np.eye(3)
    strain[axis, axis] += e
    model = md.PairModel(layered_table(), strain @ h0, R_CUT, 0.5,
                         types=types)
    model(r0 @ strain.T)
    return model.virial / cell.cell_volume(strain @ h0)


step = 1e-5
c11 = -(tensor(0, step)[0, 0] - tensor(0, -step)[0, 0]) / (2 * step)
c33 = -(tensor(2, step)[2, 2] - tensor(2, -step)[2, 2]) / (2 * step)
print(f"C11 = {c11 * GPA:.2f} GPa within the sheets, C33 = {c33 * GPA:.2f} "
      f"GPa across them; ratio {c11 / c33:.1f}")

results = {}
for couple, _ in COUPLE:
    r = np.load(RUNS / f"layered_{couple}.npz")
    start = np.diag(r["start_cell"])
    strain = r["cells"] / start - 1
    late = slice(-20, None)
    plane = 0.5 * (r["tensors"][late, 0, 0] + r["tensors"][late, 1, 1])
    across = r["tensors"][late, 2, 2]
    results[couple] = (strain, plane.mean(), across.mean())
    print(f"{couple}: over the last 10 ps, a changed by "
          f"{strain[late, 0].mean() * 100:+.2f}% and "
          f"{strain[late, 1].mean() * 100:+.2f}%, c by "
          f"{strain[late, 2].mean() * 100:+.2f}%; pressure in the plane "
          f"{plane.mean() * GPA:+.3f} GPa, across {across.mean() * GPA:+.3f}"
          f" GPa, mean {(2 * plane.mean() + across.mean()) / 3 * GPA:+.3f}"
          f" GPa")
    pxx = r["tensors"][late, 0, 0].mean() * GPA
    pyy = r["tensors"][late, 1, 1].mean() * GPA
    blocks = [strain[k:k + 10, 2].mean() * 100
              for k in range(20, len(strain), 10)]
    print(f"    P_xx {pxx:+.3f}, P_yy {pyy:+.3f} GPa; c in 5-ps blocks of the"
          f" hold {', '.join(f'{b:.2f}' for b in blocks)}%; early dip of c "
          f"{strain[:20, 2].min() * 100:+.2f}%")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                                 gridspec_kw=dict(wspace=0.4,
                                                  width_ratios=(1.5, 1)))
for couple, colour in COUPLE:
    strain = results[couple][0]
    t = (np.arange(len(strain)) + 1) * PIECE
    ax_a.plot(t, strain[:, 2] * 100, color=colour, lw=1.0, label=couple)
    ax_a.plot(t, strain[:, 0] * 100, color=colour, lw=0.8, ls="--")
ax_a.axvline(10.0, **THRESHOLD_STYLE)
ax_a.set_xlabel("time / ps")
ax_a.set_ylabel("change of length / %")
ax_a.set_ylim(-2, 10)
ax_a.legend(fontsize=7, loc="center right")
ax_a.text(0.02, 0.95, "solid: across; dashed: in-plane",
          transform=ax_a.transAxes, va="top", fontsize=6.5)
viz.panel_tag(ax_a, "a")

x = np.arange(3)
for k, (couple, colour) in enumerate(COUPLE):
    _, plane, across = results[couple]
    ax_b.bar(k - 0.18, plane * GPA, width=0.34, facecolor="white",
             edgecolor=colour, hatch="///", linewidth=0.8)
    ax_b.bar(k + 0.18, across * GPA, width=0.34, color=colour)
ax_b.axhline(0.0, color="black", lw=0.5)
ax_b.set_xticks(x, ["iso", "semi", "aniso"])
ax_b.set_ylabel("pressure / GPa")
ax_b.text(0.97, 0.05, "hatched: in-plane\nsolid: across", fontsize=6,
          transform=ax_b.transAxes, va="bottom", ha="right")
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, figure_path("ch14_pressure", "layered.pdf")))
