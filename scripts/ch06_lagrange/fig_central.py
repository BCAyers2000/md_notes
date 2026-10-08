"""Figure fig:la-central: a puck held to the hole by a spring.

A puck of 0.2 kg on a frictionless table, joined to the hole by a spring
of 2 N/m and natural length zero, so U = ½kr². Started 0.5 m out, moving
across the line to the hole at 0.5 m/s: L = 0.05 kg m²/s and E = 0.275 J.
(a) The effective potential U + L²/(2mr²) of the radial motion, with the
energy line; the turning points are where they meet, at 0.1581 and
0.5 m. (b) The path seen from above, computed by solve_newton: an ellipse
between the same two distances.

Prints the numbers of Section 6.5.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, viz
from mdlab.viz import (
    ACCENT,
    OXBLOOD,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

MASS, K, R0, V0 = 0.2, 2.0, 0.5, 0.5
ELL = MASS * R0 * V0
E = 0.5 * MASS * V0**2 + 0.5 * K * R0**2
print(f"L = {ELL} kg m²/s, E = {E} J, omega = {math.sqrt(K / MASS):.4f} rad/s")
# turning points: (k/2) r^4 - E r^2 + L²/(2m) = 0, a quadratic in r²
a, b, c = 0.5 * K, -E, ELL**2 / (2 * MASS)
roots = sorted(
    math.sqrt((-b + s * math.sqrt(b * b - 4 * a * c)) / (2 * a))
    for s in (-1, 1)
)
print(f"turning points {roots[0]:.4f} and {roots[1]:.4f} m")
r_circle = (ELL**2 / (MASS * K)) ** 0.25
print(f"circular orbit for this L at r = {r_circle:.4f} m")

t = np.linspace(0.0, 2 * math.pi / math.sqrt(K / MASS), 2001)
path, _ = dynamics.solve_newton(
    lambda r: -K * r, MASS, [[R0, 0.0]], [[0.0, V0]], t, force_to_accel=1.0
)
radius = np.linalg.norm(path[:, 0], axis=1)
print(f"solve_newton: r between {radius.min():.4f} and {radius.max():.4f} m")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.35)
)
r = np.linspace(0.06, 0.65, 400)
left.plot(r, 0.5 * K * r**2, label=r"$U$", **REFERENCE_STYLE)
left.plot(
    r, ELL**2 / (2 * MASS * r**2), color="0.6", lw=0.8, label=r"$L^2/(2mr^2)$"
)
left.plot(
    r,
    0.5 * K * r**2 + ELL**2 / (2 * MASS * r**2),
    color=OXBLOOD,
    label=r"$U_{\mathrm{eff}}$",
)
left.axhline(E, color=ACCENT)
left.plot(roots, [E, E], "o", color=ACCENT, ms=3.5)
left.set_xlim(0, 0.65)
left.set_ylim(0, 0.5)
left.set_xlabel(r"distance $r$ / m")
left.set_ylabel(r"energy / J")
left.text(0.585, 0.24, r"$U$", color="0.4", ha="left")
left.text(0.62, 0.035, r"$L^2/(2mr^2)$", color="0.5", ha="right")
left.text(0.6, 0.47, r"$U_{\mathrm{eff}}$", color=OXBLOOD, ha="right")
left.text(0.03, E + 0.012, r"$E$", color=ACCENT)
viz.panel_tag(left, "a")

s = np.linspace(0, 2 * np.pi, 200)
for rr in roots:
    right.plot(rr * np.cos(s), rr * np.sin(s), **THRESHOLD_STYLE)
right.plot(path[:, 0, 0], path[:, 0, 1], color=ACCENT)
right.plot(0, 0, "o", color="black", ms=3)
right.set_aspect("equal")
right.set_xlim(-0.6, 0.6)
right.set_ylim(-0.6, 0.6)
right.set_xlabel(r"$x$ / m")
right.set_ylabel(r"$y$ / m")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch06_lagrange", "central.pdf")))
