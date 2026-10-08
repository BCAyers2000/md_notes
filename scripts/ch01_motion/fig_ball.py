"""Figure fig:mo-ball: height, velocity and acceleration of the thrown ball.

Three panels on one time axis. At the top of the flight (dotted line) the
velocity is zero while the acceleration is -g, as at every other instant;
the shaded area under v(t) up to the top is the greatest height.
Prints the numbers quoted in Sections 1.3 and 1.4.
"""

import matplotlib.pyplot as plt
import numpy as np
from ball import H_MAX, T_LAND, T_TOP, V0, G

from mdlab import kinematics, viz
from mdlab.viz import ACCENT, THRESHOLD_STYLE, figure_path

t = np.linspace(0.0, T_LAND, 400)
x, v = kinematics.constant_acceleration(t, 0.0, V0, -G)
area = np.trapezoid(np.where(t <= T_TOP, v, 0.0), t)
print(
    f"top at t = {T_TOP:.4f} s, height {H_MAX:.4f} m; "
    f"back at the hand at t = {T_LAND:.4f} s "
    f"with v = {V0 - G * T_LAND:+.3f} m/s"
)
print(
    f"area under v(t) from 0 to the top = {area:.3f} m (trapezoid, 400 points)"
)

viz.use_style(notebook=False)
fig, axes = plt.subplots(
    3, 1, figsize=(viz.FULL, 4.3), sharex=True, gridspec_kw=dict(hspace=0.38)
)
xa, va, aa = axes
xa.plot(t, x, color=ACCENT)
xa.set_ylabel(r"$x$ / m")
xa.set_ylim(0, 8.5)
va.plot(t, v, color=ACCENT)
va.fill_between(t, v, 0.0, where=t <= T_TOP, color=ACCENT, alpha=0.15, lw=0)
va.axhline(0.0, color="black", lw=0.5)
va.set_ylabel(r"$v$ / m\,s$^{-1}$")
va.set_ylim(-14, 14)
aa.plot(t, np.full_like(t, -G), color=ACCENT)
aa.axhline(0.0, color="black", lw=0.5)
aa.set_ylabel(r"$a$ / m\,s$^{-2}$")
aa.set_ylim(-12, 4)
aa.set_xlabel(r"time $t$ / s")
aa.set_xlim(0, T_LAND)
for ax in axes:
    ax.axvline(T_TOP, **THRESHOLD_STYLE)
xa.annotate("top", (T_TOP, H_MAX), xytext=(6, -12), textcoords="offset points")
va.annotate(
    r"$v = 0$", (T_TOP, 0.0), xytext=(6, 5), textcoords="offset points"
)
va.text(0.08, 2.2, "area = greatest height", fontsize=8, color=ACCENT)
aa.annotate(
    r"$a = -g$", (T_TOP, -G), xytext=(6, 6), textcoords="offset points"
)
for ax, tag in zip(axes, "abc", strict=True):
    viz.panel_tag(ax, tag)

print("wrote", viz.save(fig, figure_path("ch01_motion", "ball.pdf")))
