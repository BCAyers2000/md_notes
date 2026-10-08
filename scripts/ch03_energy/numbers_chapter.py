"""Numbers quoted in Chapter 3 that no figure script prints.

Run from theory/: python scripts/ch03_energy/numbers_chapter.py
Each block names the section that quotes it.
"""

import math

import numpy as np
from scipy import constants
from scipy.constants import g as G

from mdlab import dynamics, energy, units


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


# 3.1 Work done by a constant force ---------------------------------------
heading("3.1 constant force")
box, lift = 2.0, 1.5  # kg, m
print(f"box weight {box * G:.4f} N; lifted {lift} m: {box * G * lift:.2f} J")
pull, angle, run = 40.0, 30.0, 10.0  # N, degrees, m
along = pull * math.cos(math.radians(angle))
print(
    f"sledge: cos 30° = {math.cos(math.radians(angle)):.4f}, "
    f"component along the snow {along:.2f} N, "
    f"across {pull * math.sin(math.radians(angle)):.1f} N, "
    f"work {along * run:.1f} J"
)

# 3.2 Kinetic energy ------------------------------------------------------
heading("3.2 kinetic energy")
trolley, push, distance = 20.0, 10.0, 1.0
k_trolley = push * distance
print(
    f"trolley: K = {k_trolley} J, v = {math.sqrt(2 * k_trolley / trolley):.3f}"
    f" m/s; Ch. 2: a = {push / trolley} m/s², after 2 s x = "
    f"{0.5 * push / trolley * 2**2} m, v = {push / trolley * 2} m/s"
)
v0, alpha = 12.0, math.radians(60.0)
top = (v0**2 - (v0 * math.cos(alpha)) ** 2) / (2 * G)
print(
    f"projectile 12 m/s at 60°: speed at top {v0 * math.cos(alpha):.3f} m/s,"
    f" v0² - v_top² = {v0**2 - (v0 * math.cos(alpha)) ** 2:.1f}, "
    f"height of top {top:.4f} m; check (v0 sin a)²/2g = "
    f"{(v0 * math.sin(alpha)) ** 2 / (2 * G):.4f} m"
)

# 3.3 Work done by a varying force ----------------------------------------
heading("3.3 varying force")
k_spring, block, stretch = 50.0, 0.5, 0.04
w_spring = 0.5 * k_spring * stretch**2
print(
    f"spring 50 N/m stretched 4 cm: work {w_spring:.4f} J; block 0.5 kg "
    f"reaches v = {math.sqrt(2 * w_spring / block):.4f} m/s at x = 0 "
    f"(Ch. 2: A omega = {stretch * math.sqrt(k_spring / block):.4f} m/s)"
)
print(f"trolley pushed with 10 N at 1 m/s: power {push * 1.0} W")

# 3.4 Potential energy ----------------------------------------------------
heading("3.4 potential energy")
ball, v_throw = 0.5, 12.0
e_ball = 0.5 * ball * v_throw**2
print(
    f"ball 0.5 kg at 12 m/s: E = {e_ball} J, top {e_ball / (ball * G):.4f} m"
    f" (Ch. 1: v0²/2g = {v_throw**2 / (2 * G):.4f} m), "
    f"weight {ball * G:.4f} N"
)
half = 0.5 * e_ball / (ball * G)
print(
    f"at half height {half:.4f} m: K = {e_ball / 2} J, "
    f"v = {math.sqrt(e_ball / ball):.4f} m/s = 12/sqrt 2 = "
    f"{12 / math.sqrt(2):.4f}"
)

# 3.5-3.6 The track -------------------------------------------------------
heading("3.5-3.6 track")
roots = np.sort(np.roots([0.6, 0.0, -1.2, 0.15]).real)
heights, _ = energy.hill_track(roots)
curvature = 1.8 * roots**2 - 1.2
for name, x0, h0, c0 in zip(
    ("valley", "hill", "valley"), roots, heights, curvature, strict=True
):
    print(f"{name}: x = {x0:+.4f} m, h = {h0:.4f} m, h'' = {c0:+.4f} per m")
release = 1.3  # height released from rest, m
for name, h0 in zip(
    ("left valley", "hill", "right valley"), heights, strict=True
):
    print(
        f"released at {release} m: speed at the {name} "
        f"{math.sqrt(2 * G * (release - h0)):.3f} m/s"
    )
x = np.linspace(-2.4, 2.4, 48001)
h, _ = energy.hill_track(x)
for level in (0.4, 0.8, 1.3):
    print(
        f"E/mg = {level} m: turning points",
        energy.turning_points(x, h, level).round(3),
    )
print(f"ends of the track: h(-2.4) = {h[0]:.3f} m, h(2.4) = {h[-1]:.3f} m")
print(
    f"to cross the hill from the deep valley at rest: "
    f"{math.sqrt(2 * G * (heights[1] - heights[0])):.3f} m/s"
)

# 3.7 Friction ------------------------------------------------------------
heading("3.7 friction")
box, v_slide, mu = 5.0, 2.0, 0.3
print(
    f"box 5 kg at 2 m/s, mu 0.3: friction {mu * box * G:.2f} N, stops after "
    f"{v_slide**2 / (2 * mu * G):.3f} m, heat {0.5 * box * v_slide**2} J"
)
block, ramp, slope, mu_ramp = 2.0, 3.0, math.radians(30.0), 0.2
drop = ramp * math.sin(slope)
w_weight = block * G * drop
w_friction = -mu_ramp * block * G * math.cos(slope) * ramp
k_bottom = w_weight + w_friction
print(
    f"ramp 3 m at 30°: drop {drop} m, weight's work {w_weight:.2f} J, "
    f"normal force {block * G * math.cos(slope):.3f} N, friction "
    f"{mu_ramp * block * G * math.cos(slope):.3f} N, its work "
    f"{w_friction:.2f} J, K at bottom {k_bottom:.2f} J, v = "
    f"{math.sqrt(2 * k_bottom / block):.3f} m/s (smooth: "
    f"{math.sqrt(2 * G * drop):.3f} m/s)"
)
bead, b_drag = 0.01, 0.05
v_inf = bead * G / b_drag
print(
    f"Ch. 2 bead at v_inf = {v_inf:.4f} m/s: m g v_inf = "
    f"{bead * G * v_inf:.4f} W, b v_inf² = {b_drag * v_inf**2:.4f} W"
)

# 3.8 Several bodies ------------------------------------------------------
heading("3.8 carts")
m1, m2, k_carts, squeeze = 1.0, 3.0, 200.0, 0.1
stored = 0.5 * k_carts * squeeze**2
v2 = math.sqrt(2 * stored * m1 / (m2 * (m1 + m2)))
v1 = -m2 / m1 * v2
print(
    f"spring stores {stored} J; v2² = {v2**2:.4f} = 1/6, v2 = {v2:+.4f}, "
    f"v1 = {v1:+.4f} m/s; K1 = {0.5 * m1 * v1**2:.4f} J, "
    f"K2 = {0.5 * m2 * v2**2:.4f} J"
)

# 3.9 Paths in space ------------------------------------------------------
heading("3.9 friction on two paths")
print(
    f"box 5 kg, mu 0.3: straight 5 m {-mu * 5.0 * G * 5:.2f} J, "
    f"round two sides 3 + 4 m {-mu * 5.0 * G * 7:.2f} J"
)

# 3.10 The hill on a map --------------------------------------------------
heading("3.10 hill")
px, py = 0.6, 0.5  # km
height = 300.0 * math.exp(-(px**2 + 2 * py**2) / 2)
dhdx, dhdy = -px * height, -2 * py * height
steep = math.hypot(dhdx, dhdy)
print(
    f"h(P) = {height:.1f} m; dh/dx = {dhdx:.1f} m/km, dh/dy = {dhdy:.1f} "
    f"m/km; |grad h| = {steep:.1f} m/km = slope {steep / 1000:.4f}, "
    f"angle {math.degrees(math.atan(steep / 1000)):.1f}°"
)
toward_summit = np.array([-px, -py]) / math.hypot(px, py)
along_summit = np.dot([dhdx, dhdy], toward_summit)
print(
    f"walking straight towards the summit: rate {along_summit:.1f} m/km; "
    f"gradient direction "
    f"{math.degrees(math.atan2(dhdy, dhdx)):.1f}°, summit direction "
    f"{math.degrees(math.atan2(-py, -px)):.1f}°"
)

# 3.12 Atoms --------------------------------------------------------------
heading("3.12 atoms")
amu = constants.physical_constants["atomic mass constant"][0]
print(f"1 eV = {constants.e} J; 1 amu = {amu} kg")
k_text = 0.5 * 6.94 * 1.66054e-27 * 1390**2  # as on the page
print(f"Li mass {6.94 * 1.66054e-27:.4e} kg; K at 1390 m/s {k_text:.4e} J")
print(
    f"1 amu (Å/fs)² = {amu} kg × 1e10 m²/s² = {amu * 1e10:.11e} J "
    f"= {amu * 1e10 / constants.e:.7f} eV; "
    f"ratio 1.66053906892/1.602176634 = {1.66053906892 / 1.602176634:.7f}"
)
print(
    f"MV2_TO_EV = {units.MV2_TO_EV:.7f} eV; 1/MV2_TO_EV = "
    f"{1 / units.MV2_TO_EV:.9f} = FORCE_TO_ACCEL {units.FORCE_TO_ACCEL:.9f}"
)
m_li = 6.94
a_li = dynamics.acceleration([[1.0, 0.0, 0.0]], [m_li])[0, 0]
v_li = a_li * 10.0
k_si = 0.5 * m_li * amu * (v_li * 1e5) ** 2
print(
    f"Li after 10 fs under 1 eV/Å: v = {v_li:.5f} Å/fs = {v_li * 1e5:.1f} m/s;"
    f" K = {k_si:.4e} J = {k_si / constants.e:.4f} eV; "
    f"K from mdlab {energy.kinetic_energy(m_li, [v_li]):.4f} eV; "
    f"distance {0.5 * a_li * 100:.4f} Å"
)
barrier = energy.SURFACE_BARRIER
for k0 in (barrier, 0.32):
    v = math.sqrt(2 * k0 / (m_li * units.MV2_TO_EV))
    print(f"Li with K = {k0} eV: v = {v:.5f} Å/fs = {v * 1e5:.0f} m/s")
g_len = 4 * math.pi / (math.sqrt(3) * energy.SURFACE_SPACING)
print(
    f"surface: |g| = {g_len:.4f} per Å, ripple spacing 2 pi/|g| = "
    f"{2 * math.pi / g_len:.4f} Å = sqrt(3) a / 2; top at a/sqrt 3 = "
    f"{energy.SURFACE_SPACING / math.sqrt(3):.3f} Å from a hollow; "
    f"U(top) = 9/8 barrier = {9 * barrier / 8:.4f} eV"
)

# Answers to the checks --------------------------------------------------
heading("answers to the checks")
print(f"3.2: K at half speed {50 / 4} J; 2 kg at 3 m/s {0.5 * 2 * 9} J")
print(f"3.3: 4 to 8 cm: {0.5 * 50 * (0.08**2 - 0.04**2):.2f} J")
print(f"3.4: balcony 5 m, 12 m/s: {math.sqrt(144 + 2 * G * 5):.2f} m/s")
print(
    f"3.5: least speed at the deep valley to cross the hill "
    f"{math.sqrt(2 * G * (heights[1] - heights[0])):.3f} m/s"
)
print(
    f"3.6: E/mg = 0.8 from x = 0.795: fastest at {roots[2]:.3f} m, "
    f"{math.sqrt(2 * G * (0.8 - heights[2])):.3f} m/s"
)
print(f"3.8: equal 2 kg carts: {math.sqrt(1.0 / 2.0):.4f} m/s each")
print(f"3.12: O at 0.01 Å/fs: {energy.kinetic_energy(15.999, [0.01]):.4f} eV")
print(
    f"3.13: Li with 0.25 eV where U = 0.1 eV: "
    f"{math.sqrt(2 * 0.15 / (6.94 * units.MV2_TO_EV)):.5f} Å/fs"
)

# Exercises ---------------------------------------------------------------
heading("exercises")
print(f"suitcase: {60 * 25 * math.cos(math.radians(50)):.1f} J")
climb = 70 * G * 3
print(f"stairs: weight {70 * G:.2f} N, {climb:.1f} J, {climb / 5:.1f} W")
v50 = 50 / 3.6
k50 = 0.5 * 1200 * v50**2
print(
    f"braking: 50 km/h = {v50:.3f} m/s, K = {k50:.0f} J, "
    f"d = {k50 / 6000:.2f} m; at 100 km/h {4 * k50 / 6000:.2f} m"
)
print(
    f"spring 200 N/m: 0-10 cm {0.5 * 200 * 0.01:.2f} J, 10-20 cm "
    f"{0.5 * 200 * (0.04 - 0.01):.2f} J"
)
stored = 0.5 * 500 * 0.08**2
print(
    f"launcher: {stored:.2f} J, v = {math.sqrt(2 * stored / 0.1):.3f} m/s, "
    f"h = {stored / (0.1 * G):.3f} m"
)
drop = 1.2 * (1 - math.cos(math.radians(40)))
cos40 = math.cos(math.radians(40))
print(
    f"pendulum: cos 40° = {cos40:.6f}, drop {drop:.5f} m, "
    f"v = {math.sqrt(2 * G * drop):.3f} m/s"
)
h_reach = heights[0] + 3.0**2 / (2 * G)
print(
    f"bead 3 m/s at the deep valley: H = {h_reach:.3f} m; turning points",
    energy.turning_points(x, h, h_reach).round(3),
)
print(
    "double well E = U0/2: x/a = ±",
    np.round(np.sqrt([1 - 1 / math.sqrt(2), 1 + 1 / math.sqrt(2)]), 4),
)
print(f"ice: {9 / (2 * 0.05 * G):.3f} m")
height, slope, mu = 2.0, math.radians(25.0), 0.15
length = height / math.sin(slope)
v2 = 2 * (G * height - mu * G * math.cos(slope) * length)
print(
    f"ramp: length {length:.3f} m, friction work per kg "
    f"{mu * G * math.cos(slope) * length:.3f} J, v = {math.sqrt(v2):.3f} m/s"
)
e_carts = 0.5 * 200 * 0.05**2
v2c = math.sqrt(e_carts / 3.0)
print(
    f"carts 1 and 2 kg: E = {e_carts} J, v2 = {v2c:.4f}, v1 = {-2 * v2c:.4f}"
)
print(
    f"collision: before {0.5 * 2 * 9} J, after {0.5 * 3 * 2**2} J, lost "
    f"{0.5 * 2 * 9 - 0.5 * 3 * 4} J"
)
print(f"line integral a = 0.5: a x y at (2, 1) = {0.5 * 2 * 1} eV")
print(f"curl exercise: U(1,2,3) = {-(1 * 2 + 2 * 9)} eV")
k0 = 0.5 * 15.999 * 0.01**2 * units.MV2_TO_EV
print(f"oxygen: K = {k0:.5f} eV, stops after {k0 / 0.5:.4f} Å")
k_li = 0.5 * 6.94 * 0.02**2 * units.MV2_TO_EV
print(
    f"Li drag: K(0) = {k_li:.5f} eV, after 100 fs {k_li * math.exp(-1):.5f} eV"
)
