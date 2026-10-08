"""Figure fig:en-surface: work along two paths on the model surface.

The model surface of Section 3.13 (``energy.hexagonal_surface``): three
cosine ripples make hollows on a triangular lattice, with the carbon
atoms at the tops between them. (a) Contours of U, the force −∇U (grey
arrows) and two paths from the hollow A to the top B. (b) The same paths
after a swirl c ẑ × r is added to the force; no potential energy exists
to contour, and the two works differ.

Prints every work quoted in Section 3.13.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import energy, viz
from mdlab.viz import ACCENT, ELEMENT, OCHRE, REFERENCE, figure_path

SWIRL = 0.005  # c, eV/Å²
BULGE = 1.2  # Å, height of the arc above the straight path
XLIM, YLIM = (-1.4, 5.1), (-1.7, 2.7)
SPACING = energy.SURFACE_SPACING

sites = energy.surface_sites()
A = sites["hollow"]
B = sites["top"] + np.array([SPACING, 0.0])


def conservative(r):
    return energy.hexagonal_surface(r)[1]


def swirled(r):
    return conservative(r) + energy.swirl(r, SWIRL)


s = np.linspace(0.0, 1.0, 4001)[:, None]
normal = np.array([-(B - A)[1], (B - A)[0]]) / np.linalg.norm(B - A)
PATHS = {
    "straight": A + s * (B - A),
    "arc": A + s * (B - A) + BULGE * np.sin(np.pi * s) * normal,
}
COLOURS = {"straight": ACCENT, "arc": OCHRE}

u_a, u_b = energy.hexagonal_surface(np.array([A, B]))[0]
area = 2 * BULGE * np.linalg.norm(B - A) / np.pi
print(f"A = {A} Å, U(A) = {u_a:.4f} eV")
print(f"B = {B.round(4)} Å, U(B) = {u_b:.4f} eV")
print(f"U(A) - U(B) = {u_a - u_b:.4f} eV")
print(
    f"|AB| = {np.linalg.norm(B - A):.4f} Å; area between the paths "
    f"{area:.4f} Å², 2 c area = {2 * SWIRL * area:.4f} eV"
)
works = {}
for field, f in (("conservative", conservative), ("swirled", swirled)):
    for name, path in PATHS.items():
        works[field, name] = energy.work(f, path)
        print(f"{field:12s} {name:8s} W = {works[field, name]:+.4f} eV")
print(
    "swirled: straight - arc = "
    f"{works['swirled', 'straight'] - works['swirled', 'arc']:.4f} eV"
)


def lattice_points(offset):
    a1 = np.array([SPACING, 0.0])
    a2 = np.array([SPACING / 2, SPACING * np.sqrt(3) / 2])
    pts = np.array(
        [offset + i * a1 + j * a2 for i in range(-4, 6) for j in range(-3, 4)]
    )
    inside = (
        (pts[:, 0] > XLIM[0])
        & (pts[:, 0] < XLIM[1])
        & (pts[:, 1] > YLIM[0])
        & (pts[:, 1] < YLIM[1])
    )
    return pts[inside]


hollows = lattice_points(sites["hollow"])
# the carbon atoms sit on the two kinds of top, a/√3 from a hollow
other_top = -sites["top"] + [SPACING, 0.0]
tops = np.vstack([lattice_points(sites["top"]), lattice_points(other_top)])

viz.use_style(notebook=False)
fig, axes = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.75), sharey=True, gridspec_kw=dict(wspace=0.08)
)
x = np.linspace(*XLIM, 281)
y = np.linspace(*YLIM, 193)
X, Y = np.meshgrid(x, y)
U = energy.hexagonal_surface(np.stack([X, Y], axis=-1))[0]
xs, ys = np.meshgrid(np.arange(-1.1, 5.0, 0.6), np.arange(-1.4, 2.6, 0.6))
grid = np.stack([xs, ys], axis=-1)

for ax, field, f in (
    (axes[0], "conservative", conservative),
    (axes[1], "swirled", swirled),
):
    if field == "conservative":
        ax.contour(
            X,
            Y,
            U,
            levels=(0.05, 0.1, 0.15, 0.2, 0.25, 0.32),
            colors="black",
            linewidths=0.4,
        )
    forces = f(grid)
    ax.quiver(
        xs,
        ys,
        forces[..., 0],
        forces[..., 1],
        color=REFERENCE,
        scale=1.3,
        **viz.ARROW,
    )
    ax.plot(*hollows.T, "o", mfc="white", mec="black", ms=3.2, mew=0.6)
    ax.plot(*tops.T, "o", color=ELEMENT["C"]["colour"], ms=3.2)
    for name, path in PATHS.items():
        ax.plot(*path.T, color=COLOURS[name], lw=1.6)
    ax.plot(*A, "o", color="black", ms=4)
    ax.plot(*B, "o", color="black", ms=4)
    ax.text(
        A[0] - 0.15,
        A[1] - 0.2,
        r"$A$",
        ha="right",
        va="top",
        bbox=dict(fc="white", ec="none", pad=0.5),
    )
    ax.text(
        B[0] + 0.15,
        B[1] - 0.2,
        r"$B$",
        ha="left",
        va="top",
        bbox=dict(fc="white", ec="none", pad=0.5),
    )
    ax.text(
        2.6,
        -1.35,
        rf"$W = {works[field, 'straight']:+.4f}$ eV",
        color=ACCENT,
        ha="center",
        bbox=dict(fc="white", ec="none", pad=1),
    )
    ax.text(
        0.9,
        2.3,
        rf"$W = {works[field, 'arc']:+.4f}$ eV",
        color=OCHRE,
        ha="center",
        bbox=dict(fc="white", ec="none", pad=1),
    )
    ax.set_aspect("equal")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"$x$ / \AA")
axes[0].set_ylabel(r"$y$ / \AA")
viz.panel_tag(axes[0], "a")
viz.panel_tag(axes[1], "b")

print("wrote", viz.save(fig, figure_path("ch03_energy", "surface.pdf")))
