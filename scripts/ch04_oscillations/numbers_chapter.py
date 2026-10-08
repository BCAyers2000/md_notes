"""Numbers quoted in Chapter 4 that no figure script prints.

Run from theory/: python scripts/ch04_oscillations/numbers_chapter.py
Each block names the section that quotes it.
"""

import math

import numpy as np
from scipy.constants import c as LIGHT
from scipy.constants import g as G

from mdlab import energy, oscillators, units


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


MASS, K = 0.5, 50.0  # the block of Section 2.8
OMEGA = math.sqrt(K / MASS)

# 4.1 The shadow of a circle ---------------------------------------------
heading("4.1 circle")
print(
    f"omega = {OMEGA} rad/s, period {2 * math.pi / OMEGA:.4f} s, "
    f"frequency {OMEGA / (2 * math.pi):.4f} Hz"
)
x0, v0 = 0.03, 0.4
amp = math.hypot(x0, v0 / OMEGA)
phi = math.atan2(-v0 / OMEGA, x0)
print(f"kicked start: A = {amp:.3f} m, phi = {phi:.4f} rad")
print(f"speed of the point on the circle A omega = {amp * OMEGA:.2f} m/s")
print(
    f"P starts {-phi:.3f} rad = {math.degrees(-phi):.1f}° below the axis, "
    f"reaches it at t = {-phi / OMEGA:.4f} s"
)

# 4.2 Energy --------------------------------------------------------------
heading("4.2 energy")
e = 0.5 * K * 0.04**2
print(f"E = {e:.4f} J; mean K = mean U = {e / 2:.4f} J")
x = 0.02
print(
    f"at x = 2 cm: U = {0.5 * K * x**2:.4f} J, K = {e - 0.5 * K * x**2:.4f} J"
)
t = np.linspace(0.0, 2 * math.pi / OMEGA, 200001)[:-1]
xs, vs = oscillators.harmonic(t, 0.04, 0.0, OMEGA)
print(
    f"numerical averages over one period: K {np.mean(0.5 * MASS * vs**2):.6f},"
    f" U {np.mean(0.5 * K * xs**2):.6f} J"
)

# 4.3 Phase portrait ------------------------------------------------------
heading("4.3 phase portrait")
print(
    f"semi-axes for E = 0.04 J: {math.sqrt(2 * e / K):.3f} m and "
    f"{math.sqrt(2 * e / MASS):.3f} m/s"
)

# 4.4 Small oscillations --------------------------------------------------
heading("4.4 small oscillations")
length = 1.2
w_pend = math.sqrt(G / length)
print(
    f"pendulum L = 1.2 m: omega {w_pend:.4f} rad/s, period "
    f"{2 * math.pi / w_pend:.4f} s"
)
roots = np.sort(np.roots([0.6, 0.0, -1.2, 0.15]).real)
curv = 1.8 * roots[0] ** 2 - 1.2
w_bead = math.sqrt(G * curv)
print(
    f"bead in the deep valley: h'' = {curv:.4f} /m, omega {w_bead:.4f} "
    f"rad/s, period {2 * math.pi / w_bead:.4f} s"
)
ub, a = energy.SURFACE_BARRIER, energy.SURFACE_SPACING
k_li = 2 * math.pi**2 * ub / a**2
g_len = 4 * math.pi / (math.sqrt(3) * a)
print(
    f"Li hollow: k = 2 pi² U_b / a² = {k_li:.4f} eV/Å² "
    f"(3 U_b |g|²/8 = {3 * ub * g_len**2 / 8:.4f})"
)
print(f"  a² = {a**2:.4f} Å², 2 pi² = {2 * math.pi**2:.4f}")
w_li = math.sqrt(k_li * units.FORCE_TO_ACCEL / 6.94)
print(
    f"  k/m × FORCE_TO_ACCEL = {k_li * units.FORCE_TO_ACCEL / 6.94:.4e} "
    f"fs⁻²; omega = {w_li:.6f} rad/fs; period {2 * math.pi / w_li:.1f} fs"
)
print(f"  k to five figures {k_li:.5f} eV/Å²")

# 4.5 Anharmonic wells ----------------------------------------------------
heading("4.5 anharmonic wells")
lj_k = 72 / 2 ** (1 / 3)
print(
    f"LJ: r_min = 2^(1/6) sigma = {2 ** (1 / 6):.4f} sigma; "
    f"k = 72 eps / (2^(1/3) sigma²) = {lj_k:.3f} eps/sigma²; harmonic "
    f"period {2 * math.pi / math.sqrt(lj_k):.4f} sigma sqrt(m/eps)"
)


def pendulum_u(theta):
    return 1.0 - math.cos(theta)  # U / (m g L)


for degrees in (10, 45, 90, 150, 179):
    amplitude = math.radians(degrees)
    t_ratio = oscillators.period(
        pendulum_u,
        pendulum_u(amplitude),
        1.0,
        0.0,
        0.01,
        force_to_accel=1.0,
        nodes=400,
    ) / (2 * math.pi)
    print(
        f"pendulum amplitude {degrees:3d}°: T/T0 = {t_ratio:.4f}, "
        f"period at L = 1.2 m {t_ratio * 2 * math.pi / w_pend:.3f} s"
    )


def hop_u(x):
    return float(oscillators.cosine_well(x, ub, a)[0])


t0_li = 2 * math.pi / w_li
for frac in (0.1, 0.5, 0.8333, 0.9, 0.99):
    t_li = oscillators.period(
        hop_u,
        frac * ub,
        6.94,
        0.0,
        0.01,
        force_to_accel=units.FORCE_TO_ACCEL,
        nodes=400,
    )
    print(
        f"Li hop E = {frac:.4f} U_b ({frac * ub:.3f} eV): period "
        f"{t_li:.1f} fs, {t_li / t0_li:.3f} T0"
    )


def morse_u(y):
    return float(oscillators.morse(y, 1.0, 1.0, 0.0)[0])


def lj_u(r):
    return float(oscillators.lennard_jones(r, 1.0, 1.0)[0]) + 1.0


for frac in (0.1, 0.5, 0.9):
    tm = oscillators.period(
        morse_u, frac, 1.0, 0.0, 0.01, force_to_accel=1.0, nodes=400
    )
    tl = oscillators.period(
        lj_u, frac, 1.0, 2 ** (1 / 6), 0.005, force_to_accel=1.0, nodes=400
    )
    morse_t0, lj_t0 = 2 * math.pi / math.sqrt(2), 2 * math.pi / math.sqrt(lj_k)
    print(
        f"E = {frac} of depth: Morse T/T0 = {tm / morse_t0:.3f}, "
        f"LJ T/T0 = {tl / lj_t0:.3f}"
    )
half_parabola = math.sqrt(2 * 0.5 / lj_k)
print(
    f"LJ parabola at E = 0.5 eps: ±{half_parabola:.4f} sigma, turning "
    f"points {2 ** (1 / 6) - half_parabola:.4f}, "
    f"{2 ** (1 / 6) + half_parabola:.4f} sigma"
)
clock = [
    oscillators.period(
        pendulum_u,
        pendulum_u(math.radians(d)),
        1.0,
        0.0,
        0.01,
        force_to_accel=1.0,
        nodes=400,
    )
    for d in (10, 11)
]
print(
    f"clock: 10° to 11° lengthens the period by "
    f"{100 * (clock[1] / clock[0] - 1):.4f} %, "
    f"{86400 * (clock[1] / clock[0] - 1):.0f} s a day"
)
for frac in (0.1, 0.5, 0.9):
    left = oscillators.turning_point(lj_u, frac, 2 ** (1 / 6), -0.001)
    right = oscillators.turning_point(lj_u, frac, 2 ** (1 / 6), 0.001)
    print(f"LJ E = {frac} eps: turning points {left:.4f}, {right:.4f} sigma")

# 4.6 Damping -------------------------------------------------------------
heading("4.6 damping")
for gamma in (4.0, 20.0, 40.0):
    rate = OMEGA**2 - gamma**2 / 4
    print(f"gamma = {gamma}: omega0² − gamma²/4 = {rate}", end="")
    if rate > 0:
        print(f", omega_d = {math.sqrt(rate):.4f} rad/s")
    else:
        print(f", kappa = {math.sqrt(-rate):.4f} /s" if rate < 0 else "")
kappa = math.sqrt(40.0**2 / 4 - OMEGA**2)
c_slow = 0.5 * (0.04 + 0.5 * 40.0 * 0.04 / kappa)  # y(0) = 0.04, y'(0) = 0.8
c_fast = 0.04 - c_slow
x_over = c_slow * math.exp(-(20.0 - kappa) * 0.5) + c_fast * math.exp(
    -(20.0 + kappa) * 0.5
)
x_crit = (0.04 + 10.0 * 0.04 * 0.5) * math.exp(-5.0)
print(
    f"released from 4 cm: critical x(0.5 s) = {100 * x_crit:.4f} cm; "
    f"over-damped c1 = {c_slow:.4f}, c2 = {c_fast:.4f} m, x(0.5 s) = "
    f"{100 * x_over:.4f} cm"
)
print(
    f"gamma = 4: amplitude e^(-gamma t/2) halves every "
    f"{2 * math.log(2) / 4:.4f} s; energy decays as e^(-4t)"
)

# 4.7 Driving -------------------------------------------------------------
heading("4.7 driving")
f = 0.5 / MASS  # 0.5 N on 0.5 kg
for gamma in (1.0, 2.0, 4.0):
    a_res, _ = oscillators.driven_steady_state(OMEGA, OMEGA, gamma, f)
    peak = np.linspace(5, 15, 200001)
    amps, _ = oscillators.driven_steady_state(peak, OMEGA, gamma, f)
    print(
        f"gamma {gamma}: A at omega0 {float(a_res):.4f} m = f/(gamma "
        f"omega0) = {f / (gamma * OMEGA):.4f}; peak {amps.max():.4f} m at "
        f"{peak[np.argmax(amps)]:.4f} rad/s; Q = {OMEGA / gamma:.1f}"
    )
print(f"static stretch f/omega0² = {f / OMEGA**2:.4f} m (= F/k {0.5 / K:.4f})")

# 4.8 Two bodies ----------------------------------------------------------
heading("4.8 two bodies")
mu = oscillators.reduced_mass(1.0, 3.0)
print(
    f"carts: mu = {mu}, omega = {math.sqrt(200 / mu):.4f} rad/s, quarter "
    f"period {0.5 * math.pi * math.sqrt(mu / 200):.4f} s"
)
mu_oh = oscillators.reduced_mass(1.008, 15.999)
print(f"O-H: mu = {mu_oh:.4f} amu, {mu_oh / 1.008:.4f} of the H mass")
print(f"H moves {15.999 / 1.008:.2f} times as far as O")

# 4.9 Normal modes --------------------------------------------------------
heading("4.9 normal modes")
w2, modes = oscillators.normal_modes(
    [[2, -1], [-1, 2]], [1, 1], force_to_accel=1.0
)
print("two carts, k = m = 1: omega² =", w2.round(6), "modes", modes.round(4))
mo, mc = 15.999, 12.011
ratio = math.sqrt(1 + 2 * mo / mc)
print(f"triatomic O-C-O: omega3/omega1 = sqrt(1 + 2 mO/mC) = {ratio:.4f}")
print(f"  carbon moves 2 mO/mC = {2 * mo / mc:.3f} times as far as an O")

# 4.10 Molecular vibrations -----------------------------------------------
heading("4.10 vibrations")
print(f"c = {LIGHT} m/s = {units.C_CM_PER_FS:.6e} cm/fs")
for name, wavenumber in (
    ("H2O antisymmetric stretch", 3756),
    ("H2O symmetric stretch", 3657),
    ("CH4 degenerate stretch", 3019),
    ("CH4 symmetric stretch", 2917),
    ("H2O bend", 1595),
    ("C2H6 CC stretch", 995),
):
    nu = wavenumber * units.C_CM_PER_FS  # cycles per fs
    print(
        f"{name:26s} {wavenumber} cm-1: nu = {nu:.5f} /fs, "
        f"period {1 / nu:.2f} fs, omega {2 * math.pi * nu:.4f} rad/fs"
    )
w_fast = 2 * math.pi * 3756 * units.C_CM_PER_FS
print(
    f"fastest water mode: omega {w_fast:.4f} rad/fs, "
    f"1/omega {1 / w_fast:.3f} fs"
)
print(
    f"Li model hollow: {2 * math.pi / w_li:.1f} fs, wavenumber "
    f"{w_li / (2 * math.pi * units.C_CM_PER_FS):.0f} cm-1"
)

# Answers to the checks and the exercises ---------------------------------
heading("checks and exercises")
print(f"4.1 check: v = {-0.05 * 10} m/s")
print(f"4.2 check: x = {4 / math.sqrt(2):.2f} cm")
print(
    f"4.6 check: gamma 8: omega_d {math.sqrt(84):.3f}, period "
    f"{2 * math.pi / math.sqrt(84):.3f} s"
)
print(f"4.10 check: 1000 cm-1 period {1 / (1000 * units.C_CM_PER_FS):.2f} fs")
e2 = 0.5 * 80 * 0.03**2
print(
    f"ex 4.2: E {e2} J, vmax {0.03 * math.sqrt(400)} m/s, v(1.5 cm) "
    f"{math.sqrt(2 * (e2 - 0.5 * 80 * 0.015**2) / 0.2):.3f} m/s"
)
e4 = 0.5 * 0.5 * 0.16 + 0.5 * 50 * 0.03**2
print(
    f"ex 4.4: E {e4} J, axes {math.sqrt(2 * e4 / 50):.3f} m, "
    f"{math.sqrt(2 * e4 / 0.5):.3f} m/s"
)
print(f"ex 4.5: L = {G / math.pi**2:.4f} m; ex 4.6: {math.pi / 60:.4f} s")
print(f"ex 4.9: (26/7)^(1/6) = {(26 / 7) ** (1 / 6):.4f} sigma")
print(
    f"ex 4.11: period {2 * math.pi / math.sqrt(64):.3f} s, tenth "
    f"{math.log(10) / 6:.3f} s"
)
for w in (5.0, 15.0):
    amp_w, lag_w = oscillators.driven_steady_state(w, 10.0, 2.0, 1.0)
    print(
        f"ex 4.13: omega {w}: A = {100 * float(amp_w):.3f} cm, "
        f"lag {float(lag_w):.4f} rad"
    )
mu_od = oscillators.reduced_mass(2.014, 15.999)
print(
    f"ex 4.15: m_r(OD) = {mu_od:.4f} amu, ratio "
    f"{math.sqrt(mu_oh / mu_od):.4f}, period ratio "
    f"{math.sqrt(mu_od / mu_oh):.3f}"
)
w_oh = 2 * math.pi * 3657 * units.C_CM_PER_FS
print(
    f"ex 4.19: omega {w_oh:.4f} rad/fs, k "
    f"{mu_oh * w_oh**2 / units.FORCE_TO_ACCEL:.2f} eV/Å², "
    f"{mu_oh * w_oh**2 / units.FORCE_TO_ACCEL / k_li:.1f} times the Li hollow"
)
step = 0.1 / (3756 * units.C_CM_PER_FS)
print(
    f"ex 4.20: step {step:.3f} fs, {1000 / step:.2f} -> "
    f"{math.ceil(1000 / step)} steps per ps"
)
