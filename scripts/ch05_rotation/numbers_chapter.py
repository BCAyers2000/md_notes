"""Numbers quoted in Chapter 5 that no figure script prints.

Run from theory/: python scripts/ch05_rotation/numbers_chapter.py
Each block names the section that quotes it.
"""

import math

import numpy as np
from scipy.constants import g as G

from mdlab import dynamics, oscillators, rotation, units


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


# water at its measured equilibrium geometry (CCCBDB), O first, Å and amu
R_OH, ANGLE = 0.958, math.radians(104.4776)
WATER = np.array(
    [
        [0.0, 0.0, 0.0],
        [R_OH * math.sin(ANGLE / 2), R_OH * math.cos(ANGLE / 2), 0.0],
        [-R_OH * math.sin(ANGLE / 2), R_OH * math.cos(ANGLE / 2), 0.0],
    ]
)
WATER_MASSES = np.array([15.999, 1.008, 1.008])

# 5.1 Turning in a plane ---------------------------------------------------
heading("5.1 plane")
m, r1, v1, r2 = 0.2, 0.5, 1.2, 0.25
ell = m * r1 * v1
v2 = ell / (m * r2)
k1, k2 = 0.5 * m * v1**2, 0.5 * m * v2**2
print(f"puck: ell = {ell:.3f} kg m²/s; at {r2} m speed {v2:.2f} m/s")
print(f"K {k1:.3f} J -> {k2:.3f} J; the pull does {k2 - k1:.3f} J")
print(f"area swept per second ell/(2m) = {ell / (2 * m):.3f} m²/s")
print(f"radial K at 0.05 m/s: {0.5 * m * 0.05**2 * 1e3:.2f} mJ")
u = 0.05  # steady pull, m/s
turned = ell / (m * u) * (1 / r2 - 1 / r1)
print(
    f"pulled at {u} m/s: {turned:.2f} rad = {turned / (2 * math.pi):.2f} turns"
)
print(
    f"door: 10 N at 0.8 m gives {10 * 0.8:.1f} N m; "
    f"at 0.2 m needs {8 / 0.2:.0f} N"
)

# 5.2 Cross product --------------------------------------------------------
heading("5.2 cross product")
a, b = np.array([1.0, 2.0, 0.0]), np.array([3.0, -1.0, 2.0])
c = np.cross(a, b)
print(f"(1, 2, 0) x (3, -1, 2) = {c}; dots {a @ c}, {b @ c}")
area = np.linalg.norm(c)
lagrange = math.sqrt((a @ a) * (b @ b) - (a @ b) ** 2)
print(f"|a x b| = {area:.4f} = sqrt(|a|²|b|² - (a.b)²) = {lagrange:.4f}")

# 5.4 Rigid rotation -------------------------------------------------------
heading("5.4 rigid rotation")
i0, md, out, inn, w1 = 1.5, 2.0, 0.8, 0.2, 1.0
i1 = i0 + 2 * md * out**2
i2 = i0 + 2 * md * inn**2
w2 = i1 * w1 / i2
print(f"stool: I {i1:.2f} -> {i2:.2f} kg m², omega {w1} -> {w2:.3f} rad/s")
e1, e2 = 0.5 * i1 * w1**2, 0.5 * i2 * w2**2
print(f"  K {e1:.3f} -> {e2:.3f} J, the arms do {e2 - e1:.3f} J")
h, slope, length = 1.0, math.radians(30.0), 2.0
for name, kappa in (("slide", 0.0), ("disc", 0.5), ("hoop", 1.0)):
    speed = math.sqrt(2 * G * h / (1 + kappa))
    accel = G * math.sin(slope) / (1 + kappa)
    t_down = math.sqrt(2 * length / accel)
    print(
        f"{name:5s} kappa {kappa}: v after 1 m drop {speed:.3f} m/s; on 30°, "
        f"a = {accel:.3f} m/s², 2 m in {t_down:.3f} s; "
        f"share of K in turning {kappa / (1 + kappa):.3f}"
    )

# 5.5 Inertia tensor -------------------------------------------------------
heading("5.5 inertia tensor")
centre = dynamics.centre_of_mass(WATER_MASSES, WATER)
tensor = rotation.inertia_tensor(WATER_MASSES, WATER, centre)
moments, axes = np.linalg.eigh(tensor)
print(f"water centre of mass {centre.round(4)} Å (O at origin)")
print("tensor (amu Å²):")
print(tensor.round(4))
print(f"principal moments {moments.round(4)} amu Å²; axes (columns)")
print(axes.round(3))
print(f"I_a + I_b = {moments[0] + moments[1]:.4f} = I_c {moments[2]:.4f}")
half_hh = R_OH * math.sin(ANGLE / 2)
print(
    f"H-H half-distance {half_hh:.4f} Å, "
    f"H height {R_OH * math.cos(ANGLE / 2):.4f}"
)

# 5.6 Removing rigid motion ------------------------------------------------
heading("5.6 removal")
print("see fig_remove.py for the ring of six atoms")

# 5.7 Degrees of freedom ---------------------------------------------------
heading("5.7 degrees of freedom")
for name, n, linear in (
    ("water", 3, False),
    ("carbon dioxide", 3, True),
    ("methane", 5, False),
    ("benzene", 12, False),
):
    print(f"{name}: 3N = {3 * n}, internal {3 * n - (5 if linear else 6)}")
w_bend = 2 * math.pi * 1595 * units.C_CM_PER_FS
w_anti = 2 * math.pi * 3756 * units.C_CM_PER_FS
pairs = [[0, 1], [0, 2], [1, 2]]
masses9 = np.repeat(WATER_MASSES, 3)
MIRROR = np.array([-1.0, 1.0, 1.0])  # x -> -x swaps the two H atoms


def model(k_oh, k_hh):
    hessian = rotation.spring_hessian(WATER, pairs, [k_oh, k_oh, k_hh])
    w2, modes = oscillators.normal_modes(
        hessian, masses9, force_to_accel=units.FORCE_TO_ACCEL
    )
    return w2, modes


def mismatch(p):
    w2, _ = model(*p)
    w = np.sqrt(np.clip(w2[6:], 0.0, None))
    return [w[0] - w_bend, w[1] - w_anti]


from scipy.optimize import fsolve  # noqa: E402

k_oh, k_hh = fsolve(mismatch, [48.0, 17.0])
print(
    f"spring model fitted to bend and antisymmetric stretch: k_OH = "
    f"{k_oh:.3f}, k_HH = {k_hh:.3f} eV/Å²"
)
w2r, modes_r = model(round(k_oh, 2), round(k_hh, 2))
wavenumbers = np.sqrt(np.clip(w2r[6:], 0, None)) / (
    2 * math.pi * units.C_CM_PER_FS
)
for k, nu in zip(range(6, 9), wavenumbers, strict=True):
    u = modes_r[:, k].reshape(3, 3)
    kind = "symmetric" if np.allclose(u[1] * MIRROR, u[2]) else "antisymmetric"
    print(
        f"  at the rounded k: omega² {w2r[k]:.5f} fs⁻², {nu:.1f} cm⁻¹, "
        f"{kind} under the mirror"
    )
print(f"  symmetric stretch off by {100 * (wavenumbers[2] / 3657 - 1):.1f} %")
print(f"  largest |omega²| among the six rigid: {np.abs(w2r[:6]).max():.1e}")
w_anti_only = [
    np.sqrt(model(round(k_oh, 2), kh)[0][7])
    / (2 * math.pi * units.C_CM_PER_FS)
    for kh in (5.0, 10.0, round(k_hh, 2), 25.0)
]
print(
    f"  antisymmetric stretch for k_HH = 5, 10, fitted, 25: "
    f"{np.round(w_anti_only, 1)} cm⁻¹"
)
w2_no_hh, _ = model(round(k_oh, 2), 0.0)
print(f"  without the H-H spring: {np.sum(np.abs(w2_no_hh) < 1e-9)} zeros")
diatomic = rotation.spring_hessian(
    [[0, 0, 0], [0, 0, 1.5639]], [[0, 1]], [1.0]
)
w2d, _ = oscillators.normal_modes(
    diatomic, np.repeat([6.94, 18.998], 3), force_to_accel=1.0
)
print(f"diatomic spring: omega² {w2d.round(6)}")

# 5.8 Spinning molecules ---------------------------------------------------
heading("5.8 LiF")
d_lif = 1.5639  # Å, r_e of LiF (1.563864, NIST WebBook) as quoted
m_r = oscillators.reduced_mass(6.94, 18.998)
inertia = m_r * d_lif**2
d_li = 18.998 / (6.94 + 18.998) * d_lif
print(
    f"LiF: m_r = {m_r:.4f} amu, I = m_r d² = {inertia:.3f} amu Å²; "
    f"Li {d_li:.4f} Å and F {d_lif - d_li:.4f} Å from the centre"
)
k_t = units.KB * 300.0
print(f"k_B T at 300 K = {k_t:.6f} eV")
k_amu = k_t * units.FORCE_TO_ACCEL  # amu Å² fs⁻²
w_rot = math.sqrt(2 * k_amu / inertia)
print(
    f"rotation with K = k_B T: omega {w_rot:.6f} rad/fs, period "
    f"{2 * math.pi / w_rot:.0f} fs, L = {inertia * w_rot:.4f} amu Å²/fs"
)
v_li = w_rot * d_li
print(f"  Li moves at {v_li:.5f} Å/fs = {v_li * 1e5:.0f} m/s")
t_vib = 1 / (910.34 * units.C_CM_PER_FS)
print(f"vibration 910.34 cm⁻¹: period {t_vib:.2f} fs")
print(
    f"rotation period / vibration period = {2 * math.pi / w_rot / t_vib:.1f}"
)
b_e = 1.3452576  # cm⁻¹, rotational constant of 7LiF, NIST WebBook
hbar_amu = units.HBAR * units.FORCE_TO_ACCEL  # amu Å² fs⁻¹
i_7lif = hbar_amu / (4 * math.pi * units.C_CM_PER_FS * b_e)
print(
    f"7LiF: I from B_e = {i_7lif:.3f} amu Å², "
    f"{100 * (i_7lif / inertia - 1):.1f} % above the average-mass value"
)

# Answers to the checks and the exercises ----------------------------------
heading("checks and exercises")
print(
    f"5.1 check: let out to 1 m: {ell / (m * 1.0):.2f} m/s, work "
    f"{0.5 * m * (ell / m) ** 2 - k1:.3f} J"
)
masses2 = [1.0, 1.0]
r2b = [[0.5, 0, 0], [-0.5, 0, 0]]
v2b = [[0, -1, 0], [0, 1, 0]]
print(
    f"5.3 check: L about origin {rotation.angular_momentum(masses2, r2b, v2b)}"
    f", about (0, 1, 0) "
    f"{rotation.angular_momentum(masses2, r2b, v2b, [0, 1, 0])}"
)
r6 = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
v6 = np.array([[0, 0.01, 0], [0, 0.03, 0]])
print(
    f"5.6 check: omega {rotation.angular_velocity(masses2, r6, v6)}, left "
    f"{rotation.remove_rigid_motion(masses2, r6, v6).round(12).tolist()}"
)
print(f"5.8 check: period ratio sqrt 2 = {math.sqrt(2):.3f}")
print(f"ex 5.1: door at 30°: {0.8 * 10 * math.sin(math.radians(30)):.1f} N m")
v_in = ell / (m * 0.1)
print(
    f"ex 5.2: to 0.1 m: {v_in:.1f} m/s, K {0.5 * m * v_in**2:.3f} J, "
    f"work {0.5 * m * v_in**2 - k1:.3f} J"
)
a4, b4 = np.array([2.0, 0, 1]), np.array([1.0, 3, -1])
c4 = np.cross(a4, b4)
print(
    f"ex 5.4: {c4}, dots {a4 @ c4}, {b4 @ c4}, area "
    f"{np.linalg.norm(c4):.4f} = "
    f"sqrt({(a4 @ a4) * (b4 @ b4) - (a4 @ b4) ** 2})"
)
i1b = 2.0 + 2 * 3.0 * 0.7**2
i2b = 2.0 + 2 * 3.0 * 0.3**2
w2b = 1.5 * i1b / i2b
lz = 1.5 * i1b
print(
    f"ex 5.9: I {i1b:.2f} -> {i2b:.2f}, L {lz:.2f}, omega {w2b:.3f} rad/s, "
    f"K {0.5 * i1b * 1.5**2:.4f} -> {lz**2 / (2 * i2b):.4f} J, work "
    f"{lz**2 / (2 * i2b) - 0.5 * i1b * 1.5**2:.3f} J"
)
a_sph = G * math.sin(slope) / 1.4
print(
    f"ex 5.10: sphere v {math.sqrt(2 * G / 1.4):.3f} m/s, a {a_sph:.3f}, "
    f"2 m in {math.sqrt(4 / a_sph):.3f} s"
)
print(f"ex 5.11: friction on 1 kg disc {G * 0.5 / 3:.3f} N")
square = np.array([[1.0, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
print(f"ex 5.12: tensor diag {rotation.inertia_tensor([1] * 4, square)}")
A15 = np.array([[4.0, 1, 0], [1, 4, 1], [0, 1, 4]])
print(f"ex 5.15: solution {np.linalg.solve(A15, [6.0, 12, 14])}")
r16 = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
v16 = np.array([[0.01, 0.01, 0], [-0.01, 0.03, 0]])
u16 = rotation.remove_rigid_motion(masses2, r16, v16)
w16 = rotation.angular_velocity(masses2, r16, v16)
kin = 0.5 * np.sum(np.array(masses2)[:, None] * v16**2)
drift16 = np.array(masses2) @ v16 / 2
parts = (
    0.5 * 2 * drift16 @ drift16,
    0.5 * w16 @ rotation.inertia_tensor(masses2, r16) @ w16,
    0.5 * np.sum(u16**2),
)
print(f"ex 5.16: V {drift16}, omega {w16.round(6)}, u {u16.round(6).tolist()}")
print(
    f"  K {kin:.4f} = {parts[0]:.4f} + {parts[1]:.4f} + {parts[2]:.4f} "
    f"amu Å²/fs² = {kin * units.MV2_TO_EV:.4f} eV "
    f"({' + '.join(f'{p * units.MV2_TO_EV:.4f}' for p in parts)})"
)
for name, n, linear in (
    ("NH3", 4, False),
    ("C2H2", 4, True),
    ("H2O2", 4, False),
    ("C2H6", 8, False),
):
    print(f"ex 5.17 / check: {name} internal {3 * n - (5 if linear else 6)}")
w_v = 2 * math.pi / t_vib
print(
    f"ex 5.19: omega_vib {w_v:.5f} rad/fs; stretch "
    f"{d_lif * (w_rot / w_v) ** 2:.5f} Å"
)
w_hot = math.sqrt(2 * units.KB * 1000 * units.FORCE_TO_ACCEL / inertia)
print(f"  (omega/omega_v)² = {(w_rot / w_v) ** 2:.4e}")
k_hot = units.KB * 1000
print(
    f"ex 5.20: k_B T at 1000 K = {k_hot:.5f} eV = "
    f"{k_hot * units.FORCE_TO_ACCEL:.4e} amu Å²/fs², omega {w_hot:.5f} "
    f"rad/fs, period {2 * math.pi / w_hot:.0f} fs = 992 x "
    f"{math.sqrt(k_t / k_hot):.5f}"
)
