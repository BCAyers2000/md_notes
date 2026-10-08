"""Figure fig:mo-plane: velocity and acceleration in two dimensions.

(a) A projectile launched at 12 m/s, 60 degrees above the horizontal:
the velocity is tangent to the path, the acceleration always points down.
(b) Uniform motion on a circle: the velocity is tangent, the acceleration
points to the centre. Prints the numbers quoted in Sections 1.6
and 1.7.
"""

import matplotlib.pyplot as plt
import numpy as np
from ball import V0, G

from mdlab import kinematics, viz
from mdlab.viz import ACCENT, OXBLOOD, figure_path

ANGLE = np.radians(60.0)
V0_VEC = V0 * np.array([np.cos(ANGLE), np.sin(ANGLE)])
A = np.array([0.0, -G])
T_FLIGHT = 2 * V0_VEC[1] / G
RANGE = V0**2 * np.sin(2 * ANGLE) / G
HEIGHT = V0_VEC[1] ** 2 / (2 * G)
print(
    f"projectile: flight {T_FLIGHT:.4f} s, range {RANGE:.3f} m, "
    f"greatest height {HEIGHT:.3f} m; range at 45 degrees "
    f"{V0**2 / G:.3f} m"
)

# the stone of Section 1.7: a 0.5 m string, two turns a second
omega_stone = 2 * 2 * np.pi
print(
    f"stone: omega = {omega_stone:.2f} rad/s, speed = {0.5 * omega_stone:.2f} "
    f"m/s, acceleration = {omega_stone**2 * 0.5:.1f} m/s² = "
    f"{omega_stone**2 * 0.5 / G:.2f} g"
)

ARROW = dict(
    angles="xy",
    scale_units="xy",
    width=0.007,
    headwidth=4,
    headlength=4.5,
    headaxislength=4,
    zorder=3,
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1,
    2,
    figsize=(viz.FULL, 2.05),
    gridspec_kw=dict(width_ratios=(2.05, 1), wspace=0.25),
)
t = np.linspace(0.0, T_FLIGHT, 200)
r, _ = kinematics.constant_acceleration(t, np.zeros(2), V0_VEC, A)
left.plot(*r.T, color="black", lw=1.0)
ts = np.linspace(0.0, T_FLIGHT, 7)  # odd, so one arrow sits at the top
rs, vs = kinematics.constant_acceleration(ts, np.zeros(2), V0_VEC, A)
left.quiver(*rs.T, *vs.T, color=ACCENT, scale=1 / 0.13, **ARROW)
left.quiver(
    *rs.T,
    *np.broadcast_to(A, rs.shape).T,
    color=OXBLOOD,
    scale=1 / 0.12,
    **ARROW,
)
left.plot(*rs.T, "o", color="black", ms=2.5, zorder=4)
left.set_aspect("equal")
left.set_xlim(-0.6, 13.9)
left.set_ylim(-1.4, 6.6)
left.set_xlabel(r"$x$ / m")
left.set_ylabel(r"$y$ / m")
left.plot([], [], color=ACCENT, label=r"velocity $\mathbf{v}$")
left.plot([], [], color=OXBLOOD, label=r"acceleration $\mathbf{a}$")
left.legend(loc="lower center", handlelength=1.2)
viz.panel_tag(left, "a")

phase = np.linspace(0.0, 2 * np.pi, 200)
right.plot(np.cos(phase), np.sin(phase), color="black", lw=1.0)
tc = np.linspace(0.0, 2 * np.pi, 6, endpoint=False) + 0.3
rc, vc, ac = kinematics.circular_motion(tc, 1.0, 1.0)
# quiver widths are fractions of the axes width; this panel is narrower
NARROW = dict(ARROW, width=0.016)
right.quiver(*rc.T, *vc.T, color=ACCENT, scale=1 / 0.45, **NARROW)
right.quiver(*rc.T, *ac.T, color=OXBLOOD, scale=1 / 0.45, **NARROW)
right.plot(*rc.T, "o", color="black", ms=2.5, zorder=4)
right.plot(0, 0, "+", color="black", ms=5)
right.set_aspect("equal")
right.set_xlim(-1.55, 1.55)
right.set_ylim(-1.55, 1.55)
right.set_xlabel(r"$x / R$")
right.set_ylabel(r"$y / R$")
right.set_xticks([-1, 0, 1])
right.set_yticks([-1, 0, 1])
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch01_motion", "plane.pdf")))
