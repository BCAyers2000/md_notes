"""Numbers quoted in Chapter 6 that no figure script prints.

Run from theory/: python scripts/ch06_lagrange/numbers_chapter.py
Each block names the section that quotes it.
"""

import math

import numpy as np
from scipy.constants import g as G
from scipy.optimize import brentq

from mdlab import energy, lagrangian


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


# 6.2 The action --------------------------------------------------------
heading("6.2 action")
m_ball, v0, T = 0.145, 12.0, 2.0
print(
    f"ball: excess per m² = m pi²/(4T) = {m_ball * math.pi**2 / (4 * T):.5f}"
)
m, k, T_s = 0.5, 50.0, 0.5
omega = math.sqrt(k / m)
print(f"spring: half period pi/omega = {math.pi / omega:.4f} s; T = {T_s} s")
for n in (1, 2):
    coeff = (T_s / 4) * (m * (n * math.pi / T_s) ** 2 - k)
    t = np.linspace(0.0, T_s, 20001)
    true = 0.04 * np.cos(omega * t)
    bump = np.sin(n * math.pi * t / T_s)

    def spring_u(x):
        return 0.5 * k * x**2

    s0 = lagrangian.action(true, t, m, spring_u)
    s1 = lagrangian.action(true + 0.01 * bump, t, m, spring_u)
    print(
        f"  eta = sin({n} pi t/T): excess {coeff:+.4f} eps², numerically "
        f"{(s1 - s0) / 0.01**2:+.4f}"
    )

# 6.4 Constrained motion ------------------------------------------------
heading("6.4 constrained")
alpha = math.radians(30)
a = -G * math.tan(alpha) / (1 + math.tan(alpha) ** 2)
print(
    f"straight slope 30°: x'' = {a:.4f}, along the slope "
    f"{a / math.cos(alpha):.4f} = -g sin 30° = {-G * 0.5:.4f} m/s²"
)
print(f"pendulum L = 1.2 m: omega0 = {math.sqrt(G / 1.2):.3f} rad/s")

# 6.6 Multipliers -------------------------------------------------------
heading("6.6 multipliers")
print("rectangle of perimeter P: a square of side P/4")
print(f"pendulum from 90°: tension at the bottom 3mg = {3 * 0.5 * G:.3f} N")

# Answers to the checks and the exercises -------------------------------
heading("checks and exercises")
print(f"ex: bead from deep to shallow valley {1.5330 - 0.5542:.4f} s")
for v_start in (0.3,):
    ell = 0.2 * 0.4 * v_start
    e = 0.5 * 0.2 * v_start**2 + 0.5 * 2.0 * 0.4**2
    a4, b4, c4 = 0.5 * 2.0, -e, ell**2 / (2 * 0.2)
    roots = sorted(
        math.sqrt((-b4 + s * math.sqrt(b4 * b4 - 4 * a4 * c4)) / (2 * a4))
        for s in (-1, 1)
    )
    print(
        f"ex: puck from 0.4 m at {v_start} m/s: L = {ell:.3f}, E = {e:.3f}, "
        f"turning points {roots[0]:.4f}, {roots[1]:.4f} m"
    )
ell = 0.05
r_c = (ell**2 / (0.2 * 2.0)) ** 0.25
print(
    f"ex: circular orbit r = {r_c:.4f} m, angular speed "
    f"{ell / (0.2 * r_c**2):.4f} = sqrt(k/m) = {math.sqrt(2.0 / 0.2):.4f}"
)
print(
    f"ex: pendulum from 60°, 0.5 kg: force at bottom {2 * 0.5 * G:.3f} N, "
    f"at release {0.5 * 0.5 * G:.3f} N"
)
print(f"ex: largest xy on the unit circle: {0.5:.3f} at x = y = {2**-0.5:.4f}")
for n in (1, 2):
    print(
        f"ex: spring action coefficient for n = {n}: "
        f"{(T_s / 4) * (m * (n * math.pi / T_s) ** 2 - k):+.3f} J s per m²"
    )


def shape(x):
    h, slope = energy.hill_track(x)
    return float(h), float(slope), 1.8 * x**2 - 1.2


start = brentq(lambda x: float(energy.hill_track(x)[0]) - 0.8, -2.5, -1.5)
t = np.linspace(0.0, 4.0, 40001)
x, v = lagrangian.solve_wire(shape, start, 0.0, t, g=G)
returns = np.nonzero((v[1:] >= 0) & (v[:-1] < 0))[0]
print(
    f"ex: bead from rest at h = 0.8 m (x = {start:.4f} m): turns at "
    f"x = {x[np.argmax(x)]:.4f} m; back at the start after "
    f"{t[returns[0] + 1]:.4f} s"
)
print(f"ex: x e^x from 0 to 1 by parts: {math.e - (math.e - 1):.4f}")

# 6.4: the bead's times also follow from energy, length over speed
heading("6.4 times by quadrature")
from scipy.integrate import quad  # noqa: E402

h_e = 1.3
start13 = brentq(lambda x: float(energy.hill_track(x)[0]) - h_e, -2.5, -2.0)


def time_per_x(x):
    h, slope = energy.hill_track(x)
    return math.sqrt(1 + float(slope) ** 2) / math.sqrt(
        2 * G * (h_e - float(h))
    )


roots_h = np.sort(np.roots([0.6, 0.0, -1.2, 0.15]).real)
for name, target in zip(
    ("deep valley", "hill top", "shallow valley"), roots_h, strict=True
):
    t_q, _ = quad(time_per_x, start13, target, limit=200)
    print(f"{name}: {t_q:.4f} s by quadrature")
t_end, _ = quad(time_per_x, start13, 2.0, limit=200)
print(f"to the turning point x = 2 m: {t_end:.4f} s")
