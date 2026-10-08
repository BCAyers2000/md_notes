"""Figure fig:ha-blob: a blob of pendulums keeps its area.

The pendulum of Section 7.2 (0.5 kg on a 1.2 m rod). Every point on the
edge of an elliptical blob of starting states, centred on θ = 1.6 rad at
rest, is carried along by Hamilton's equations. (a) Without drag the
blob is sheared, because swings of different size take different times,
but its area stays the same. (b) With the drag −γp, γ = 0.5 per second,
added to dp/dt, it shrinks as e^{−γt} while it spirals towards the
bottom.

Prints the areas quoted in Section 7.4.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import hamiltonian, viz
from mdlab.viz import figure_path

MASS, LENGTH = 0.5, 1.2
INERTIA, MGL = MASS * LENGTH**2, MASS * G * LENGTH
CENTRE, HALF_WIDTHS = (1.6, 0.0), (0.4, 0.8)  # rad, kg m²/s
GAMMA = 0.5  # per second
UNDAMPED_TIMES = [0.0, 1.0, 2.0, 3.0]
DAMPED_TIMES = [0.0, 1.0, 2.0, 3.0]


def dh_dq(q, p):
    return MGL * np.sin(q)


def dh_dp(q, p):
    return p / INERTIA


s = np.linspace(0.0, 2 * np.pi, 6000, endpoint=False)
q0 = CENTRE[0] + HALF_WIDTHS[0] * np.cos(s)
p0 = CENTRE[1] + HALF_WIDTHS[1] * np.sin(s)
area0 = math.pi * HALF_WIDTHS[0] * HALF_WIDTHS[1]
print(
    f"starting area pi a b = {area0:.5f} J s "
    f"(shoelace {hamiltonian.polygon_area(q0, p0):.5f})"
)
energies = (p0**2 / (2 * INERTIA) + MGL * (1 - np.cos(q0))) / MGL
print(
    f"energies from {energies.min():.3f} to {energies.max():.3f} mgL "
    "(separatrix 2)"
)

qu, pu = hamiltonian.flow(dh_dq, dh_dp, q0, p0, UNDAMPED_TIMES)
for t, q, p in zip(UNDAMPED_TIMES, qu, pu, strict=True):
    print(f"no drag, t = {t} s: area {hamiltonian.polygon_area(q, p):.5f} J s")
change = max(
    abs(hamiltonian.polygon_area(q, p) / area0 - 1)
    for q, p in zip(qu, pu, strict=True)
)
print(f"largest relative change of the area without drag {change:.1e}")
qd, pd = hamiltonian.flow(dh_dq, dh_dp, q0, p0, DAMPED_TIMES, gamma=GAMMA)
for t, q, p in zip(DAMPED_TIMES, qd, pd, strict=True):
    ratio = hamiltonian.polygon_area(q, p) / area0
    print(
        f"drag, t = {t} s: area ratio {ratio:.5f}, "
        f"exp(-gamma t) = {math.exp(-GAMMA * t):.5f}"
    )
print(f"halving time ln2/gamma = {math.log(2) / GAMMA:.4f} s")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), sharey=True, gridspec_kw=dict(wspace=0.22)
)
th = np.linspace(-np.pi, np.pi, 400)
separatrix = 2 * math.sqrt(INERTIA * MGL) * np.cos(th / 2)
for ax, times, qs, ps, tag in [
    (left, UNDAMPED_TIMES, qu, pu, "a"),
    (right, DAMPED_TIMES, qd, pd, "b"),
]:
    ax.plot(th, separatrix, color="0.8", lw=0.7)
    ax.plot(th, -separatrix, color="0.8", lw=0.7)
    shades = plt.get_cmap("cividis")(np.linspace(0.0, 0.85, len(times)))
    for t, q, p, shade in zip(times, qs, ps, shades, strict=True):
        ax.fill(
            q,
            p,
            facecolor=(*shade[:3], 0.35),
            edgecolor=shade,
            lw=0.9,
            label=f"{t:g} s",
        )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xticks(
        [-np.pi / 2, 0, np.pi / 2],
        [r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$"],
    )
    ax.set_xlabel(r"angle $\theta$ / rad")
    viz.panel_tag(ax, tag)
left.set_ylabel(r"$p$ / kg m$^2$ s$^{-1}$")
left.legend(loc="lower left", fontsize=7, handlelength=1.0, borderaxespad=0.2)

print("wrote", viz.save(fig, figure_path("ch07_hamilton", "blob.pdf")))
