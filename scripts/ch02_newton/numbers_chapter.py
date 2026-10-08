"""Numbers quoted in Chapter 2 that no figure script prints.

Each block names the section that quotes it. Physical data (the Earth's
radius, rotation and mass, the speed of sound) are standard values,
stated where they are used; the constants of Section 2.10 come from the
CODATA values that ship with SciPy.
"""

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from scipy import constants

from mdlab import dynamics, units

G = constants.g
AMU = constants.physical_constants["atomic mass constant"][0]
EV_PER_ANGSTROM = constants.e / 1e-10  # one eV/Å in newtons
LITHIUM = atomic_masses[atomic_numbers["Li"]]
HYDROGEN = atomic_masses[atomic_numbers["H"]]

# Section 2.1: adding two forces --------------------------------------------
print(
    f"2.1  4 east + 3 north = {np.hypot(4, 3):.0f} units at "
    f"{np.degrees(np.arctan2(3, 4)):.1f} degrees north of east"
)

# Section 2.2: is a frame of reference fixed to the ground inertial? -------
EARTH_RADIUS = 6.378e6  # m, at the equator
SIDEREAL_DAY = 86164.0  # s, one turn relative to the stars
print(
    f"2.2  sidereal day from the solar day: 86400 x 365.24 / 366.24 = "
    f"{86400 * 365.24 / 366.24:.0f} s"
)
omega_earth = 2 * np.pi / SIDEREAL_DAY
a_equator = omega_earth**2 * EARTH_RADIUS
print(
    f"2.2  Earth: omega = {omega_earth:.4e} rad/s, acceleration at the "
    f"equator {a_equator:.4f} m/s² = {100 * a_equator / G:.2f}% of g"
)

# Section 2.3: worked examples of the second law ---------------------------
print("2.3  same pull: 3 m/s² against 1 m/s², so the mass ratio is 3")
print(f"2.3  20 kg pushed by 10 N: a = {10 / 20} m/s²")

# Section 2.4: the Earth's recoil from a falling ball ----------------------
EARTH_MASS = 5.972e24  # kg
BALL = 0.5  # kg
print(
    f"2.4  ball of {BALL} kg: the Earth accelerates at "
    f"{BALL * G / EARTH_MASS:.1e} m/s²"
)

# Section 2.5: weights and the hanging lamp --------------------------------
print(f"2.5  a 100 g apple weighs {0.1 * G:.3f} N")
weight_li = LITHIUM * AMU * G
print(
    f"2.5  a lithium atom ({LITHIUM} amu) weighs {weight_li:.2e} N = "
    f"{weight_li / EV_PER_ANGSTROM:.1e} eV/Å"
)
print(f"2.5  tension holding a 2 kg lamp: {2 * G:.1f} N")

# Section 2.6: one step by hand for the thrown ball ------------------------
v0, tau = 12.0, 0.1
x_step, v_step = 0.0 + v0 * tau, v0 - G * tau
x_exact = v0 * tau - 0.5 * G * tau**2
print(
    f"2.6  one step of {tau} s: x = {x_step:.3f} m, v = {v_step:.3f} m/s; "
    f"exact x = {x_exact:.4f} m; overshoot g tau²/2 = "
    f"{0.5 * G * tau**2:.4f} m"
)

# Section 2.8: amplitude and phase from a starting state -------------------
omega, v0 = 10.0, 0.4
for x0 in (0.03, -0.03):
    amplitude = np.hypot(x0, v0 / omega)
    cos_phi, sin_phi = x0 / amplitude, -v0 / (omega * amplitude)
    # the arcsine when the cosine is not negative, its partner otherwise
    phase = np.arcsin(sin_phi)
    if cos_phi < 0:
        phase = np.pi - phase
    turn = 2 * np.pi
    assert np.isclose(phase % turn, np.arctan2(sin_phi, cos_phi) % turn)
    print(
        f"2.8  x0 = {x0} m, v0 = {v0} m/s, omega = {omega} rad/s: "
        f"A = {amplitude:.4f} m, cos phi = {cos_phi:.3f}, "
        f"sin phi = {sin_phi:.3f}, phi = {phase:.3f} rad"
    )
print(
    f"2.8  a 50 N/m spring under 1 N stretches {100 * 1 / 50:.0f} cm; "
    f"a 0.5 kg block on it has omega = {np.sqrt(50 / 0.5):.0f} rad/s"
)

# Section 2.9: momentum ----------------------------------------------------
print(f"2.9  20 kg pushed by 10 N for 2 s: p = {10 * 2} kg m/s, v = 1 m/s")
print(f"2.9  carts' centre of mass: {(1 * 0 + 3 * 0.2) / 4:.2f} m")

# Section 2.10: Newton's law in eV, Å, fs and amu --------------------------
print(
    f"2.10 1 eV/Å = {EV_PER_ANGSTROM:.9e} N; 1 eV/Å on 1 amu gives "
    f"{EV_PER_ANGSTROM / AMU:.6e} m/s² = "
    f"{units.FORCE_TO_ACCEL:.6e} Å/fs²"
)
a = dynamics.acceleration([[1.0, 0.0, 0.0]], [LITHIUM])[0, 0]
DURATION = 10.0  # fs
SPEED_OF_SOUND = 343.0  # m/s, in air at 20 °C
speed = a * DURATION * 1e5  # Å/fs to m/s
print(
    f"2.10 Li under 1 eV/Å: a = {a:.4e} Å/fs²; after {DURATION:.0f} fs, "
    f"v = {a * DURATION:.4f} Å/fs = {speed:.0f} m/s = "
    f"{speed / SPEED_OF_SOUND:.1f} x the speed of sound; moved "
    f"{0.5 * a * DURATION**2:.4f} Å"
)
print(f"2.10 lithium is {LITHIUM / HYDROGEN:.2f} times heavier than hydrogen")
