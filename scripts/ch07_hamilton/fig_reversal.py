"""Figure fig:ha-reversal: reversing the momenta retraces the path.

A puck of 0.2 kg on smooth ice, held near the origin by springs of
2 N/m east-west and 5 N/m north-south, so U = ½k_x x² + ½k_y y². It
starts at A, (0.3, 0) m, with the velocity (0.1, 0.5) m/s, and is
followed for 4 s to B (teal). There every momentum is reversed, and the
puck is followed for another 4 s (ochre, dashed). (a) Without drag it
runs back along its own path and arrives at A with its starting velocity
reversed. (b) With the drag −γp, γ = 0.3 per second, the forward path
shrinks, and the reversed one, which loses energy too, does not return.

Prints the numbers of Section 7.6.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, figure_path

MASS, KX, KY = 0.2, 2.0, 5.0  # kg, N/m, N/m
START, VELOCITY, DURATION, GAMMA = (0.3, 0.0), (0.1, 0.5), 4.0, 0.3


def run(r0, p0, gamma):
    def rates(_t, y):
        x, y_, px, py = y
        return [
            px / MASS,
            py / MASS,
            -KX * x - gamma * px,
            -KY * y_ - gamma * py,
        ]

    t = np.linspace(0.0, DURATION, 2001)
    sol = solve_ivp(
        rates,
        (0, DURATION),
        [*r0, *p0],
        t_eval=t,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    return sol.y


print(
    f"omega_x = {math.sqrt(KX / MASS):.4f}, "
    f"omega_y = {math.sqrt(KY / MASS):.4f} rad/s"
)
paths = {}
for gamma in (0.0, GAMMA):
    p0 = (MASS * VELOCITY[0], MASS * VELOCITY[1])
    forward = run(START, p0, gamma)
    back = run(forward[:2, -1], -forward[2:, -1], gamma)
    paths[gamma] = (forward, back)
    miss = math.hypot(back[0, -1] - START[0], back[1, -1] - START[1])
    print(
        f"gamma = {gamma}: B = ({forward[0, -1]:.4f}, {forward[1, -1]:.4f})"
        f" m; after the reversed run at ({back[0, -1]:.6f}, "
        f"{back[1, -1]:.6f}) m, {miss:.1e} m from A, with momentum "
        f"({back[2, -1]:.4f}, {back[3, -1]:.4f}) kg m/s"
    )
print(
    f"starting momentum ({MASS * VELOCITY[0]:.4f}, "
    f"{MASS * VELOCITY[1]:.4f}) kg m/s"
)

viz.use_style(notebook=False)
fig, axes = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.0), sharey=True, gridspec_kw=dict(wspace=0.1)
)
for ax, gamma, tag in zip(axes, (0.0, GAMMA), "ab", strict=True):
    forward, back = paths[gamma]
    ax.plot(forward[0], forward[1], color=ACCENT, lw=1.6)
    ax.plot(back[0], back[1], color=OCHRE, ls="--", lw=1.0)
    for x, y, name, dx, dy in [
        (START[0], START[1], "A", 0.02, 0.02),
        (forward[0, -1], forward[1, -1], "B", -0.05, 0.025),
    ]:
        ax.plot(x, y, "o", color="black", ms=3.5, zorder=4)
        ax.text(x + dx, y + dy, name)
    ax.plot(back[0, -1], back[1, -1], "s", color=OCHRE, ms=3.5, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-0.36, 0.36)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel(r"$x$ / m")
    viz.panel_tag(ax, tag)
axes[0].set_ylabel(r"$y$ / m")

print("wrote", viz.save(fig, figure_path("ch07_hamilton", "reversal.pdf")))
