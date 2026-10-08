"""Figure fig:la-tension: the multiplier of a pendulum is the rod's tension.

A bob of 0.5 kg on a light rod of 1.2 m, released from rest at 120°
from the hanging position. Its motion is computed in x and y with the
constraint |r| = L held by the Lagrange multiplier of Section 6.6
(lagrangian.rod_multiplier). The tension −λL, divided by the weight mg,
is drawn against the angle over the first swing, on top of
3 cos θ − 2 cos θ₀ from energy. Where it is negative the rod pushes, and
a string would go slack.

Prints the numbers of Section 6.6.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G
from scipy.integrate import solve_ivp

from mdlab import lagrangian, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, THRESHOLD_STYLE, figure_path

MASS, LENGTH, THETA0 = 0.5, 1.2, math.radians(120.0)
GRAVITY = np.array([0.0, -G])


def rates(_t, y):
    r, v = y[:2], y[2:]
    lam = lagrangian.rod_multiplier(r, v, MASS, GRAVITY)
    return [*v, *(GRAVITY + lam * r / MASS)]


start = [LENGTH * math.sin(THETA0), -LENGTH * math.cos(THETA0), 0.0, 0.0]
t = np.linspace(0.0, 3.0, 6001)
sol = solve_ivp(rates, (0, 3.0), start, t_eval=t, rtol=1e-11, atol=1e-13)
r, v = sol.y[:2].T, sol.y[2:].T
theta = np.arctan2(r[:, 0], -r[:, 1])
first = np.argmax(theta < -THETA0 + 1e-3) or len(t)
tension = np.array(
    [
        -lagrangian.rod_multiplier(ri, vi, MASS, GRAVITY) * LENGTH
        for ri, vi in zip(r, v, strict=True)
    ]
)
print(f"length kept to within {np.ptp(np.linalg.norm(r, axis=1)):.1e} m")
print(
    f"tension at release {tension[0]:.4f} N = mg cos(theta0) "
    f"{MASS * G * math.cos(THETA0):.4f}; at the bottom "
    f"{tension[np.argmin(np.abs(theta))]:.3f} N = mg (3 - 2 cos theta0) = "
    f"{MASS * G * (3 - 2 * math.cos(THETA0)):.3f}"
)
slack = math.degrees(math.acos(2 * math.cos(THETA0) / 3))
print(f"tension negative beyond {slack:.2f} degrees")
print(f"first swing ends after {t[first - 1]:.3f} s")

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF * 1.3, 2.4))
grid = np.linspace(-THETA0, THETA0, 400)
ax.plot(
    np.degrees(grid),
    3 * np.cos(grid) - 2 * math.cos(THETA0),
    label="from energy",
    **REFERENCE_STYLE,
)
ax.plot(
    np.degrees(theta[:first]),
    tension[:first] / (MASS * G),
    color=ACCENT,
    lw=0.9,
    label="multiplier",
)
ax.axhline(0.0, **THRESHOLD_STYLE)
ax.set_xlim(-125, 125)
ax.set_xticks([-120, -60, 0, 60, 120])
ax.set_xlabel(r"angle $\theta$ / degrees")
ax.set_ylabel(r"force of the rod / $mg$")
ax.legend(loc="lower center", fontsize="small")

print("wrote", viz.save(fig, figure_path("ch06_lagrange", "tension.pdf")))
