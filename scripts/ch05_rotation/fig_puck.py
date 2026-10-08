"""Figure fig:ro-puck: a puck on a string pulled in through a hole.

A puck of 0.2 kg circles a hole at 0.5 m and 1.2 m/s on a frictionless
table; the string through the hole is then pulled down at a steady
0.05 m/s until the puck circles at 0.25 m. The string pulls only towards
the hole, so the angular momentum stays at 0.12 kg m²/s.
(a) The path seen from above, with three sectors swept in equal times of
0.25 s, at the start, in the middle and at the end of the pull; all three
have the area 0.075 m². (b) The angular momentum, the speed and the
kinetic energy against the radius, each divided by its starting value.

The motion is computed by solve_newton with the string's pull m v_θ²/r,
which keeps the radial speed steady, and the script checks that r falls
linearly and L stays constant. Prints the checks and the sector areas;
the other numbers of Section 5.1 come from numbers_chapter.py.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, viz
from mdlab.viz import ACCENT, OCHRE, figure_path

MASS, R1, V1, PULL = 0.2, 0.5, 1.2, 0.05
ELL = MASS * R1 * V1


def string(r, v):
    """The pull m v_θ²/r towards the hole, which keeps d²r/dt² zero."""
    d = np.linalg.norm(r, axis=-1, keepdims=True)
    rhat = r / d
    radial = np.sum(v * rhat, axis=-1, keepdims=True)
    tangential2 = np.sum(v**2, axis=-1, keepdims=True) - radial**2
    return -MASS * tangential2 / d * rhat


t = np.linspace(0.0, 5.0, 2001)
r, v = dynamics.solve_newton(
    string,
    MASS,
    [[R1, 0.0]],
    [[-PULL, V1]],
    t,
    force_to_accel=1.0,
    velocity_dependent=True,
)
x, y = r[:, 0, 0], r[:, 0, 1]
radius = np.hypot(x, y)
ell = MASS * (x * v[:, 0, 1] - y * v[:, 0, 0])
speed = np.linalg.norm(v[:, 0], axis=-1)
print(
    f"r falls linearly to within {np.abs(radius - (R1 - PULL * t)).max():.1e}"
    f" m; L between {ell.min():.10f} and {ell.max():.10f} kg m²/s"
)
print(f"final speed {speed[-1]:.4f} m/s (2.4 round, 0.05 inwards)")
theta = np.unwrap(np.arctan2(y, x))
print(f"turned {theta[-1]:.3f} rad")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.7), gridspec_kw=dict(wspace=0.35)
)
left.plot(x, y, color="black", lw=0.7)
for start, colour in ((0.0, ACCENT), (2.5, OCHRE), (4.75, ACCENT)):
    pick = (t >= start) & (t <= start + 0.25)
    xs = np.concatenate([[0.0], x[pick], [0.0]])
    ys = np.concatenate([[0.0], y[pick], [0.0]])
    left.fill(xs, ys, color=colour, alpha=0.35, lw=0)
    area = 0.5 * np.trapezoid(radius[pick] ** 2, theta[pick])
    print(f"sector from t = {start} s: area {area:.4f} m²")
left.plot(0, 0, "o", color="black", ms=3)
left.plot(R1, 0, "o", color=ACCENT, ms=4)
left.set_aspect("equal")
left.set_xlim(-0.55, 0.55)
left.set_ylim(-0.55, 0.55)
left.set_xlabel(r"$x$ / m")
left.set_ylabel(r"$y$ / m")
viz.panel_tag(left, "a")

right.plot(radius, ell / ELL, color=ACCENT, label=r"$L$")
right.plot(radius, speed / speed[0], color="black", label="speed")
right.plot(radius, (speed / speed[0]) ** 2, color=OCHRE, label=r"$K$")
right.set_xlim(0.5, 0.25)
right.set_ylim(0, 4.3)
right.set_xlabel(r"radius $r$ / m")
right.set_ylabel("ratio to the start")
right.legend(loc="upper left")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch05_rotation", "puck.pdf")))
