"""Numbers quoted in Chapter 7 that no figure script prints.

Run from theory/: python scripts/ch07_hamilton/numbers_chapter.py
"""

import math

import numpy as np
from scipy.constants import g as G

from mdlab import hamiltonian


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


heading("7.1 check: block of 2 kg on 8 N/m with E = 1 J")
print(f"H = p²/{2 * 2} + {0.5 * 8} x²; at x = 0, p = {math.sqrt(2 * 2 * 1.0)}")

heading("7.2 and exercises: the pendulum, 0.5 kg on 1.2 m")
MASS, LENGTH = 0.5, 1.2
INERTIA, MGL = MASS * LENGTH**2, MASS * G * LENGTH
OMEGA0 = math.sqrt(G / LENGTH)
p_sep = 2 * INERTIA * OMEGA0
print(
    f"separatrix momentum {p_sep:.4f} kg m²/s, bob speed p/(mL) = "
    f"{p_sep / (MASS * LENGTH):.4f} m/s, 2 sqrt(gL) = "
    f"{2 * math.sqrt(G * LENGTH):.4f}"
)
print(
    f"10 deg to 1 deg by the approximation: ln10/omega0 = "
    f"{math.log(10) / OMEGA0:.4f} s"
)
exact = (
    math.log(math.tan(math.radians(10) / 4) / math.tan(math.radians(1) / 4))
    / OMEGA0
)
print(f"  exactly, from pi − theta = 4 arctan(e^(−omega0 t)): {exact:.4f} s")
ball_top = 12.0 / G
print(
    f"ball thrown at 12 m/s: top after {ball_top:.4f} s, back at the hand "
    f"after {2 * ball_top:.4f} s"
)

heading("7.4: the Jacobian of the pendulum's flow")


def dh_dq(q, p):
    return MGL * np.sin(q)


def dh_dp(q, p):
    return p / INERTIA


jac = hamiltonian.flow_jacobian(dh_dq, dh_dp, 1.0, 0.5, 2.5)
print(f"J from (1 rad, 0.5) over 2.5 s:\n{jac.round(4)}")
print(f"det J − 1 = {np.linalg.det(jac) - 1:.1e}")
jac_d = hamiltonian.flow_jacobian(dh_dq, dh_dp, 1.0, 0.5, 2.5, gamma=0.5)
print(
    f"with gamma = 0.5: det J = {np.linalg.det(jac_d):.6f}, "
    f"exp(−1.25) = {math.exp(-1.25):.6f}"
)
print(f"check: area 1 J s after 4 s at gamma 0.5: {math.exp(-2.0):.4f} J s")
print(f"ex: halving time ln2/gamma = {math.log(2) / 0.5:.4f} s")

heading("exercise: the shear of free flight, 1 kg over 2 s")
corners = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
moved = np.column_stack(
    [corners[:, 0] + corners[:, 1] * 2.0 / 1.0, corners[:, 1]]
)
print(
    f"corners go to {moved.tolist()}, area "
    f"{hamiltonian.polygon_area(moved[:, 0], moved[:, 1]):.4f} J s"
)

heading("exercise: torque of the box pair about (5, 5) Å")
F1 = 0.2360680 * np.array([-2.0, 1.0]) / math.sqrt(5)
pair = np.array([[1.0, 1.0], [9.0, 2.0]]) - [5.0, 5.0]
forces = np.array([F1, -F1])
print(
    f"F1 = {F1.round(6)}, torque about the centre "
    f"{np.sum(pair[:, 0] * forces[:, 1] - pair[:, 1] * forces[:, 0]):.4f} eV"
)

heading("7.7: step maps of the spring")
SPRING_M, SPRING_K = 0.5, 50.0


def spring_force(q, p):
    return SPRING_K * q


def spring_rate(q, p):
    return p / SPRING_M


for wdt in (0.05,):
    print(f"check: (1 + {wdt}²)^100 = {(1 + wdt**2) ** 100:.4f}")
print(
    "text: doubling at omega dt = 0.1: "
    f"{math.log(2) / math.log(1.01):.2f} steps"
)
n_ex = math.log(2) / math.log(1.0025)
print(
    f"ex: doubling at dt = 0.005 s: {n_ex:.2f} steps, "
    f"{math.ceil(n_ex) * 0.005:.3f} s; 1.0025^278 = {1.0025**278:.4f}"
)
for dt in (0.02, 0.01):
    q, p = 0.04, 0.0
    e0 = 0.5 * SPRING_K * q**2
    lo = hi = 1.0
    for _ in range(int(round(2000 / dt))):
        q, p = hamiltonian.symplectic_euler_step(
            spring_force, spring_rate, q, p, dt
        )
        e = (p * p / (2 * SPRING_M) + 0.5 * SPRING_K * q * q) / e0
        lo, hi = min(lo, e), max(hi, e)
    print(
        f"symplectic Euler, dt = {dt} s, 2000 s: E/E0 from {lo:.4f} "
        f"to {hi:.4f}"
    )

heading("7.7: symplectic Euler keeps area for the pendulum (dt = 0.05 s)")
H_STEP = 1e-6
for q0, p0 in [(1.0, 0.5), (2.5, -3.0), (-0.3, 4.5)]:
    cols = []
    for dq, dp in [(H_STEP, 0.0), (0.0, H_STEP)]:
        up = hamiltonian.symplectic_euler_step(
            dh_dq, dh_dp, q0 + dq, p0 + dp, 0.05
        )
        down = hamiltonian.symplectic_euler_step(
            dh_dq, dh_dp, q0 - dq, p0 - dp, 0.05
        )
        cols.append(
            [(u - d) / (2 * H_STEP) for u, d in zip(up, down, strict=True)]
        )
    det = cols[0][0] * cols[1][1] - cols[1][0] * cols[0][1]
    print(f"state ({q0}, {p0}): det J − 1 = {float(det) - 1:.1e}")
