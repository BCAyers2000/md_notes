"""Figure fig:mo-secant: velocity as the limit of secant slopes.

(a) Height of the ball against time, with secants from t1 over shrinking
intervals and the tangent at t1. (b) The secant slope against the
interval: a straight line that reaches v(t1) as the interval vanishes.
Prints the numbers that Section 1.2 (sec:mo-velocity) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np
from ball import T_LAND, V0, G

from mdlab import kinematics, viz
from mdlab.viz import (
    ACCENT,
    GREEN,
    OCHRE,
    OXBLOOD,
    THRESHOLD_STYLE,
    figure_path,
)

T0 = 0.4
TAUS = (1.2, 0.6, 0.2)


def height(t):
    return kinematics.constant_acceleration(t, 0.0, V0, -G)[0]


v0 = V0 - G * T0
print(f"g = {G} m/s², V0 = {V0} m/s, t1 = {T0} s, v(t1) = {v0:.3f} m/s")
print(f"height at t1: x(t1) = {height(T0):.3f} m")
print(f"top of the flight at t = V0/g = {V0 / G:.2f} s")
for tau in TAUS:
    slope = (height(T0 + tau) - height(T0)) / tau
    print(
        f"tau = {tau:.1f} s: secant slope = {slope:.3f} m/s, "
        f"minus v(t1) = {slope - v0:+.3f} = -g tau/2 = {-G * tau / 2:+.3f}"
    )

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.35)
)
t = np.linspace(0.0, T_LAND, 200)
left.plot(t, height(t), color="black", lw=1.2)
for tau, colour in zip(TAUS, (GREEN, OXBLOOD, OCHRE), strict=True):
    left.plot(
        [T0, T0 + tau],
        [height(T0), height(T0 + tau)],
        color=colour,
        lw=1.0,
        marker="o",
        ms=3,
        label=rf"$\tau = {tau}$ s",
    )
span = np.array([-0.3, 0.45])
left.plot(
    T0 + span,
    height(T0) + v0 * span,
    color=ACCENT,
    lw=1.0,
    zorder=1,
    label="tangent",
)
left.plot(T0, height(T0), "o", color="black", ms=3.5, zorder=4)
left.set_xlabel(r"time $t$ / s")
left.set_ylabel(r"height $x$ / m")
left.set_xlim(0, T_LAND)
left.set_ylim(0, 8.5)
viz.panel_tag(left, "a")

tau = np.linspace(0.0, 1.4, 50)
right.plot(tau, v0 - G * tau / 2, color="black", lw=1.2)
right.axhline(v0, **THRESHOLD_STYLE)
for tt, colour in zip(TAUS, (GREEN, OXBLOOD, OCHRE), strict=True):
    right.plot(tt, v0 - G * tt / 2, "o", color=colour, ms=4)
right.plot(0.0, v0, "o", color=ACCENT, ms=4, clip_on=False, zorder=4)
right.text(1.38, v0 + 0.12, r"$v(t_1)$", ha="right", va="bottom", color=ACCENT)
right.set_xlabel(r"interval $\tau$ / s")
right.set_ylabel(r"secant slope / m\,s$^{-1}$")
right.set_xlim(0, 1.4)
viz.panel_tag(right, "b")
fig.legend(
    *left.get_legend_handles_labels(),
    loc="upper center",
    ncols=4,
    bbox_to_anchor=(0.5, 1.06),
)

print("wrote", viz.save(fig, figure_path("ch01_motion", "secant.pdf")))
