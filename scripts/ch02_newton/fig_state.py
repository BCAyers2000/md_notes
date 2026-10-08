"""Figure fig:ne-state: the state of a falling ball.

Three balls start 3 m above the ground with velocities of +2, 0 and
-2 m/s. (a) Height against time: the same starting position with
different velocities gives different motions. (b) The same motions as
paths in the (x, v) plane, the plane of states. Each arrow shows how the
state at its tail changes over 0.03 s, (v, -g) times 0.03 s.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import kinematics, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, figure_path

HEIGHT = 3.0  # m
VELOCITIES = (2.0, 0.0, -2.0)  # m/s, upwards positive
COLOURS = (ACCENT, OCHRE, OXBLOOD)
ARROW_TIME = 0.03  # s


def landing_time(v0):
    """Positive root of HEIGHT + v0 t - g t² / 2 = 0."""
    return (v0 + np.sqrt(v0**2 + 2 * G * HEIGHT)) / G


for v0 in VELOCITIES:
    t_land = landing_time(v0)
    print(
        f"v0 = {v0:+.0f} m/s: lands after {t_land:.3f} s at "
        f"{v0 - G * t_land:+.3f} m/s"
    )

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.5), gridspec_kw=dict(wspace=0.32)
)

for v0, colour in zip(VELOCITIES, COLOURS, strict=True):
    t = np.linspace(0.0, landing_time(v0), 200)
    x, v = kinematics.constant_acceleration(t, HEIGHT, v0, -G)
    sign = "+" if v0 > 0 else ""
    label = rf"$v_0 = {sign}{v0:.0f}$ m\,s$^{{-1}}$"
    left.plot(t, x, color=colour, label=label)
    right.plot(x, v, color=colour)
    right.plot(HEIGHT, v0, "o", color=colour, ms=4, zorder=4)

left.plot(0.0, HEIGHT, "o", color="black", ms=4, zorder=4)
left.set_xlim(0, 1.05)
left.set_ylim(0, 4.0)
left.set_xlabel(r"time $t$ / s")
left.set_ylabel(r"height $x$ / m")
left.legend(loc="upper right", handlelength=1.4)
viz.panel_tag(left, "a")

# the direction field: the change of state over ARROW_TIME at each point
xs, vs = np.meshgrid(np.linspace(0.3, 3.0, 7), np.linspace(-9.0, 3.0, 7))
right.quiver(
    xs,
    vs,
    vs * ARROW_TIME,
    np.full_like(vs, -G * ARROW_TIME),
    color=viz.REFERENCE,
    scale=1,
    **{**viz.ARROW, "width": 0.005, "zorder": 1},
)
right.axhline(0.0, color="black", lw=0.4)
right.set_xlim(0, 3.4)
right.set_ylim(-9.5, 3.5)
right.set_xlabel(r"height $x$ / m")
right.set_ylabel(r"velocity $v$ / m\,s$^{-1}$")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch02_newton", "state.pdf")))
