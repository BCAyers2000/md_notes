"""Figure fig:ro-water: the principal axes of a water molecule.

Water at its measured equilibrium geometry (O-H 0.958 Å, angle 104.48°,
CCCBDB), drawn with one O-H bond along x so that its inertia tensor about
the centre of mass has an off-diagonal entry. The eigenvectors of the
tensor, its principal axes, are drawn through the centre of mass: one
along the bisector of the angle, one at right angles to it in the plane,
and the third, out of the plane, marked by its moment only.

Prints the tensor, the principal moments and the angle of the bisector,
quoted in Section 5.5.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from mdlab import dynamics, rotation, viz
from mdlab.viz import ACCENT, ELEMENT, OCHRE, figure_path

R_OH, ANGLE = 0.958, math.radians(104.4776)
POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [R_OH, 0.0, 0.0],
        [R_OH * math.cos(ANGLE), R_OH * math.sin(ANGLE), 0.0],
    ]
)
MASSES = np.array([15.999, 1.008, 1.008])
SYMBOLS = ("O", "H", "H")

centre = dynamics.centre_of_mass(MASSES, POSITIONS)
tensor = rotation.inertia_tensor(MASSES, POSITIONS, centre)
moments, axes = np.linalg.eigh(tensor)
print(f"centre of mass {centre.round(4)} Å")
print("tensor about it (amu Å²):")
print(tensor.round(4))
print(f"principal moments {moments.round(4)} amu Å²")
for k in range(3):
    print(f"  axis {k}: {axes[:, k].round(4)}")
page = tensor.round(4)  # the entries as printed in the book
trace = page[0, 0] + page[1, 1]
det = page[0, 0] * page[1, 1] - page[0, 1] ** 2
root = math.sqrt(trace**2 - 4 * det)
print(
    f"in-plane block: lambda² - {trace:.4f} lambda + {det:.4f} = 0; "
    f"root {root:.4f}; lambda = {(trace - root) / 2:.5f}, "
    f"{(trace + root) / 2:.5f} (from the four-figure entries)"
)
angle_i2 = math.degrees(math.atan2(axes[1, 1], axes[0, 1]))
print(f"axis of I_2 at {angle_i2:.2f}° to x")
bisector = math.degrees(ANGLE / 2)
print(f"bisector at {bisector:.2f}° to the x axis")

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF, 2.6))
labels = ((-0.12, -0.2), (0.0, -0.22), (-0.2, 0.12))  # offsets, Å
for (x, y, _), symbol, (dx, dy) in zip(
    POSITIONS, SYMBOLS, labels, strict=True
):
    style = ELEMENT[symbol]
    ax.add_patch(
        Circle(
            (x, y),
            0.25 * float(style["radius"]),
            facecolor=str(style["colour"]),
            edgecolor="black",
            lw=0.5,
            zorder=3,
        )
    )
    ax.text(x + dx, y + dy, symbol, ha="center", va="center", fontsize=8)
for k, colour in ((0, ACCENT), (1, OCHRE)):
    direction = axes[:2, k] / np.linalg.norm(axes[:2, k])
    if direction[1] < 0:
        direction = -direction  # label the upper end of each axis
    ends = centre[:2] + np.outer([-1.1, 1.1], direction)
    ax.plot(ends[:, 0], ends[:, 1], color=colour, lw=1.0, zorder=2)
    tip = centre[:2] + 1.12 * direction
    ax.text(
        tip[0],
        tip[1] + 0.04,
        rf"$I_{k + 1} = {moments[k]:.3f}$",
        color=colour,
        fontsize=8,
        ha="center" if direction[0] < 0 else "left",
        va="bottom",
    )
ax.plot(*centre[:2], "+", color="black", ms=7, mew=1.0, zorder=5)
ax.set_aspect("equal")
ax.set_xlim(-1.4, 1.6)
ax.set_ylim(-0.9, 1.6)
ax.set_xlabel(r"$x$ / \AA")
ax.set_ylabel(r"$y$ / \AA")
ax.set_title(r"moments in amu \AA$^2$", fontsize=8, pad=8)
ax.text(0.98, 0.98, rf"$I_3 = {moments[2]:.3f}$" + "\n(out of plane)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)

print("wrote", viz.save(fig, figure_path("ch05_rotation", "water.pdf")))
