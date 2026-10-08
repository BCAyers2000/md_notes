"""Figure fig:en-li: a lithium atom on the model surface.

A lithium atom leaves a hollow of the model surface with 0.33 eV of
kinetic energy, 10% more than the 0.3 eV barrier at the bridges, moving
at 37.5° to the x axis. The motion is the reference solution of
``dynamics.solve_newton``. (a) The potential energy U, shaded from the
hollows (light) to the tops (dark), the boundary U = E of the small
forbidden islands about the carbon atoms, and the path over 1.5 ps. (b)
Kinetic, potential and total energy over the first 600 fs.

Prints the numbers that Section 3.13 (sec:en-surface) quotes.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, energy, units, viz
from mdlab.viz import ACCENT, ELEMENT, OCHRE, OXBLOOD, figure_path

MASS_LI = 6.94  # amu
K0 = 0.33  # eV
ANGLE = np.radians(37.5)
DIRECTION = np.array([np.cos(ANGLE), np.sin(ANGLE)])
DURATION = 1500.0  # fs
SPACING = energy.SURFACE_SPACING


def surface_force(r):
    return energy.hexagonal_surface(r)[1]


speed = math.sqrt(2 * K0 / (MASS_LI * units.MV2_TO_EV))
v0 = speed * DIRECTION / np.linalg.norm(DIRECTION)
t = np.linspace(0.0, DURATION, 6001)
r, v = dynamics.solve_newton(
    surface_force,
    MASS_LI,
    energy.surface_sites()["hollow"],
    v0,
    t,
    force_to_accel=units.FORCE_TO_ACCEL,
)
kinetic = 0.5 * MASS_LI * np.sum(v**2, axis=-1) * units.MV2_TO_EV
potential = energy.hexagonal_surface(r)[0]
total = kinetic + potential

# which hollow the atom is nearest, to count the hops between hollows
a1 = np.array([SPACING, 0.0])
a2 = np.array([SPACING / 2, SPACING * np.sqrt(3) / 2])
indices = np.array([(i, j) for i in range(-8, 9) for j in range(-8, 9)])
hollows = indices @ np.array([a1, a2])
nearest = np.argmin(
    np.linalg.norm(r[:, None, :] - hollows[None, :, :], axis=-1), axis=1
)
visited = set(nearest.tolist())
hops = int(np.sum(np.diff(nearest) != 0))
print(
    f"starting speed {speed:.5f} Å/fs; E = {total[0]:.4f} eV; "
    f"largest |E - E(0)| = {np.max(np.abs(total - total[0])):.1e} eV"
)
print(
    f"K ranges {kinetic.min():.4f} to {kinetic.max():.4f} eV; "
    f"U ranges {potential.min():.4f} to {potential.max():.4f} eV; "
    f"{hops} changes of nearest hollow, {len(visited)} hollows visited"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1,
    2,
    figsize=(viz.FULL, 3.2),
    gridspec_kw=dict(width_ratios=(1.0, 1.0), wspace=0.3),
)
lo = r.min(axis=0) - 1.2
hi = r.max(axis=0) + 1.2
x = np.linspace(lo[0], hi[0], 400)
y = np.linspace(lo[1], hi[1], 400)
X, Y = np.meshgrid(x, y)
U = energy.hexagonal_surface(np.stack([X, Y], axis=-1))[0]
shading = left.contourf(
    X, Y, U, levels=np.linspace(0.0, 0.3375, 10), cmap="cividis_r", alpha=0.45
)
left.contour(X, Y, U, levels=[total[0]], colors="black", linewidths=0.6)
pts = np.array([i * a1 + j * a2 for i in range(-6, 8) for j in range(-6, 7)])
tops = np.vstack(
    [
        pts + energy.surface_sites()["top"],
        pts - energy.surface_sites()["top"] + a1,
    ]
)
left.plot(*tops.T, "o", color=ELEMENT["C"]["colour"], ms=2.0)
left.plot(*r.T, color="white", lw=2.2)
left.plot(*r.T, color=ACCENT, lw=1.0)
left.plot(*r[0], "o", color="black", ms=3.5)
left.set_aspect("equal")
left.set_xlim(lo[0], hi[0])
left.set_ylim(lo[1], hi[1])
left.set_xlabel(r"$x$ / \AA")
left.set_ylabel(r"$y$ / \AA")
viz.panel_tag(left, "a")
colourbar = fig.colorbar(
    shading, ax=left, orientation="horizontal", fraction=0.07, pad=0.24,
    ticks=[0.0, 0.15, 0.3],
)
colourbar.set_label(r"potential energy $U$ / eV")

WINDOW = 600.0  # fs shown in (b)
right.plot(t, kinetic, color=OCHRE)
right.plot(t, potential, color=OXBLOOD, ls="--")
right.plot(t, total, color=ACCENT)
right.text(
    WINDOW, total[0] + 0.012, r"$E$", color=ACCENT, ha="right", va="bottom"
)
right.set_xlim(0, WINDOW)
right.legend(
    ["$K$", "$U$"],
    loc="upper center",
    ncols=2,
    bbox_to_anchor=(0.45, 1.02),
    handlelength=1.2,
)
right.set_ylim(0, 0.4)
right.set_xlabel(r"time $t$ / fs")
right.set_ylabel(r"energy / eV")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch03_energy", "li_energy.pdf")))
