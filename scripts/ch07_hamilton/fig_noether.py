"""Figure fig:ha-noether: forces that are not pairwise still cancel.

(a) Three atoms in a plane whose energy depends only on the angle at the
middle one, U = ½k(θ − θ₀)² with k = 2 eV/rad² and θ₀ = 104.5°, held at
θ = 130° with bonds of 1.0 and 1.3 Å, tilted by 20°. The forces, from
central differences of U, are not along the lines joining the atoms
(dotted), yet they add to zero and give no torque, because U is
unchanged when the three are moved or turned together. (b) Two atoms in
a periodic box 10 Å wide, joined by a spring (k = 1 eV/Å², natural
length 2 Å) to the nearest copy of each other. Moving both changes
nothing, so the forces add to zero; turning both while the box stays
fixed changes U, and the forces give a torque.

Prints the numbers of Section 7.5.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, ARROW, OCHRE, THRESHOLD_STYLE, figure_path

K_ANGLE, THETA0 = 2.0, math.radians(104.5)  # eV/rad², rad
HALF, TILT = math.radians(65.0), math.radians(20.0)  # 130° held, tilted
DOWN = 1.5 * math.pi + TILT
ATOMS = np.array(
    [
        [1.0 * math.cos(DOWN - HALF), 1.0 * math.sin(DOWN - HALF)],
        [0.0, 0.0],
        [1.3 * math.cos(DOWN + HALF), 1.3 * math.sin(DOWN + HALF)],
    ]
)


def angle_energy(r):
    a, b = r[0] - r[1], r[2] - r[1]
    cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    return 0.5 * K_ANGLE * (math.acos(cos) - THETA0) ** 2


def forces_of(energy, r, h=1e-6):
    out = np.zeros_like(r)
    for i in range(len(r)):
        for a in range(r.shape[1]):
            up, down = r.copy(), r.copy()
            up[i, a] += h
            down[i, a] -= h
            out[i, a] = -(energy(up) - energy(down)) / (2 * h)
    return out


def torque(r, f):
    return float(np.sum(r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]))


f3 = forces_of(angle_energy, ATOMS)
print(f"angle term at 130 deg: U = {angle_energy(ATOMS):.4f} eV")
for i, f in enumerate(f3):
    print(
        f"  force on atom {i}: ({f[0]:+.4f}, {f[1]:+.4f}) eV/Å, "
        f"|F| = {np.linalg.norm(f):.4f}"
    )
print(
    f"  sum ({f3.sum(axis=0)[0]:.1e}, {f3.sum(axis=0)[1]:.1e}), "
    "torque about the origin "
    f"{torque(ATOMS, f3):.1e} eV, about (3, 2) Å "
    f"{torque(ATOMS - [3.0, 2.0], f3):.1e} eV"
)
bond = ATOMS[0] - ATOMS[1]
print(f"  force on atom 0 along its bond: {f3[0] @ bond:.1e} (perpendicular)")

BOX, K_BOND, R0 = 10.0, 1.0, 2.0
PAIR = np.array([[1.0, 1.0], [9.0, 2.0]])


def box_energy(r):
    d = r[1] - r[0]
    d -= BOX * np.round(d / BOX)  # the nearest copy
    return 0.5 * K_BOND * (np.linalg.norm(d) - R0) ** 2


f2 = forces_of(box_energy, PAIR)
d = PAIR[1] - PAIR[0]
d -= BOX * np.round(d / BOX)
print(f"box: nearest-copy separation {d}, length {np.linalg.norm(d):.4f} Å")
print(
    f"  force on atom 1 ({f2[0, 0]:+.4f}, {f2[0, 1]:+.4f}) eV/Å, "
    f"on atom 2 ({f2[1, 0]:+.4f}, {f2[1, 1]:+.4f})"
)
print(
    f"  sum ({f2.sum(axis=0)[0]:.1e}, {f2.sum(axis=0)[1]:.1e}), "
    "torque about the origin "
    f"{torque(PAIR, f2):.4f} eV"
)
for shift in ([0.3, -0.7], [5.0, 5.0]):
    print(
        f"  U after a shift by {shift}: {box_energy(PAIR + shift):.6f} eV "
        f"(before {box_energy(PAIR):.6f})"
    )
alpha = 1e-6
turn = np.array(
    [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]
)
du = (box_energy(PAIR @ turn.T) - box_energy(PAIR @ turn.T.T)) / (2 * alpha)
print(f"  dU/d(alpha) for a turn about the origin {du:.4f} eV")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.7), gridspec_kw=dict(wspace=0.3)
)
scale = 0.5  # Å of arrow per eV/Å
for i, j in [(0, 1), (1, 2), (0, 2)]:
    left.plot(*ATOMS[[i, j]].T, **THRESHOLD_STYLE)
left.plot(*ATOMS.T, "o", color="black", ms=6, zorder=4)
left.quiver(*ATOMS.T, *(f3.T), color=ACCENT, scale=1 / scale, **ARROW)
for i, (x, y) in enumerate(ATOMS):
    left.text(x + 0.1, y + 0.1, f"{i + 1}")
left.set_aspect("equal")
left.set_xlim(-1.4, 1.6)
left.set_ylim(-1.5, 1.2)
left.set_xlabel(r"$x$ / Å")
left.set_ylabel(r"$y$ / Å")
viz.panel_tag(left, "a")

right.add_patch(plt.Rectangle((0, 0), BOX, BOX, fill=False, lw=0.8))
image = PAIR[1] - [BOX, 0.0]
right.plot(*np.array([PAIR[0], image]).T, color=OCHRE, lw=2.4, alpha=0.6)
right.plot(*PAIR.T, **THRESHOLD_STYLE)
right.plot(*PAIR.T, "o", color="black", ms=5, zorder=4)
right.plot(*image, "o", mfc="white", mec="black", ms=5, zorder=4)
arrow_scale = 6.0
right.quiver(*PAIR.T, *(f2.T), color=ACCENT, scale=1 / arrow_scale, **ARROW)
right.text(1.25, 0.15, "1")
right.text(8.6, 2.45, "2")
right.text(-0.3, 2.6, "copy of 2", fontsize=7, ha="right")
right.set_aspect("equal")
right.set_xlim(-4.0, 11.5)
right.set_ylim(-1, 11)
right.set_xlabel(r"$x$ / Å")
right.set_ylabel(r"$y$ / Å")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch07_hamilton", "noether.pdf")))
