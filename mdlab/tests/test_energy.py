"""Chapter 3: work and energy, against ASE, finite differences and
exact results."""

import math

import numpy as np
import pytest
from scipy import constants
from scipy.integrate import cumulative_trapezoid

from mdlab import dynamics, energy, units

rng = np.random.default_rng(2026)
SITES = energy.surface_sites()


def surface_force(r):
    return energy.hexagonal_surface(r)[1]


def central_gradient(potential, r, h=1e-5):
    """−∇U by central differences, one coordinate at a time."""
    r = np.asarray(r, dtype=float)
    gradient = np.zeros_like(r)
    for k in range(r.shape[-1]):
        step = np.zeros(r.shape[-1])
        step[k] = h
        gradient[..., k] = (potential(r + step) - potential(r - step)) / (
            2 * h
        )
    return gradient


# -----------------------------------------------------------------------------
# kinetic energy and its unit
# -----------------------------------------------------------------------------
def test_kinetic_energy_matches_ase():
    ase = pytest.importorskip("ase")
    ase_units = pytest.importorskip("ase.units")
    masses = np.array([6.94, 12.011, 15.999])
    v = rng.normal(scale=0.01, size=(3, 3))
    atoms = ase.Atoms("LiCO", positions=np.zeros((3, 3)), masses=masses)
    atoms.set_velocities(v / ase_units.fs)  # ASE's own time unit
    # ASE's default constants are CODATA 2014; mdlab follows SciPy's.
    assert math.isclose(
        energy.kinetic_energy(masses, v),
        atoms.get_kinetic_energy(),
        rel_tol=1e-6,
    )


def test_kinetic_energy_unit_follows_the_derivation_of_section_3_12():
    # 1 amu (1 Å/fs)² = 1 amu × (1e5 m/s)², in J, then divided by 1 eV in J
    amu = constants.physical_constants["atomic mass constant"][0]
    in_joules = amu * (1e5) ** 2
    assert math.isclose(
        units.MV2_TO_EV, in_joules / constants.e, rel_tol=1e-15
    )
    assert f"{units.MV2_TO_EV:.4f}" == "103.6427"


def test_lithium_kinetic_energy_equals_the_work_done_on_it():
    # Section 2.10: 1 eV/Å on Li from rest for 10 fs. K = F d exactly.
    a = dynamics.acceleration([[1.0, 0.0, 0.0]], [6.94])[0, 0]
    t = 10.0
    kinetic = energy.kinetic_energy([6.94], [[a * t, 0.0, 0.0]])
    assert math.isclose(kinetic, 1.0 * 0.5 * a * t**2, rel_tol=1e-12)


def test_kinetic_energy_needs_one_mass_per_atom():
    with pytest.raises(ValueError):
        energy.kinetic_energy([1.0, 2.0], np.zeros((3, 3)))


# -----------------------------------------------------------------------------
# turning points and the track
# -----------------------------------------------------------------------------
def test_turning_points_of_a_spring():
    k, e = 50.0, 0.04  # N/m, J: amplitude sqrt(2E/k) = 0.04 m
    x = np.linspace(-0.1, 0.1, 2001)
    found = energy.turning_points(x, 0.5 * k * x**2, e)
    assert np.allclose(found, [-0.04, 0.04], atol=1e-6)


def test_track_slope_is_the_derivative_of_its_height():
    x = rng.uniform(-2.4, 2.4, size=50)
    _, slope = energy.hill_track(x)
    h = 1e-6
    up, _ = energy.hill_track(x + h)
    down, _ = energy.hill_track(x - h)
    assert np.allclose(slope, (up - down) / (2 * h), atol=1e-8)


def test_track_has_two_valleys_and_a_hill():
    # the equilibria are the roots of the cubic dh/dx = 0.6x³ − 1.2x + 0.15
    roots = np.sort(np.roots([0.6, 0.0, -1.2, 0.15]).real)
    height, slope = energy.hill_track(roots)
    assert np.allclose(slope, 0.0, atol=1e-12)
    curvature = 1.8 * roots**2 - 1.2  # d²h/dx²
    assert curvature[0] > 0 and curvature[1] < 0 and curvature[2] > 0
    assert height[0] < height[2] < height[1]


# -----------------------------------------------------------------------------
# the model surface
# -----------------------------------------------------------------------------
def test_surface_force_is_minus_the_gradient():
    r = rng.uniform(-4.0, 4.0, size=(30, 2))
    numerical = -central_gradient(lambda p: energy.hexagonal_surface(p)[0], r)
    assert np.allclose(surface_force(r), numerical, atol=1e-8)


def test_surface_site_energies():
    u, f = energy.hexagonal_surface(np.array(list(SITES.values())))
    barrier = energy.SURFACE_BARRIER
    assert np.allclose(u, [0.0, barrier, 9.0 * barrier / 8.0])
    assert np.allclose(f, 0.0, atol=1e-12)  # every site is an equilibrium


def test_surface_is_periodic_on_its_triangular_lattice():
    a = energy.SURFACE_SPACING
    lattice = np.array([[a, 0.0], [a / 2, a * np.sqrt(3) / 2]])
    r = rng.uniform(-3.0, 3.0, size=(20, 2))
    u0, f0 = energy.hexagonal_surface(r)
    for shift in lattice:
        u1, f1 = energy.hexagonal_surface(r + shift)
        assert np.allclose(u0, u1, atol=1e-12)
        assert np.allclose(f0, f1, atol=1e-12)


def test_hollow_is_a_minimum_top_a_maximum_and_bridge_a_saddle():
    def u(p):
        return energy.hexagonal_surface(p)[0]

    h = 0.05
    along = np.array([h, 0.0])  # from hollow to bridge
    across = np.array([0.0, h])  # along the bond, towards a top
    hollow, bridge, top = SITES["hollow"], SITES["bridge"], SITES["top"]
    for step in (along, across):
        assert u(hollow + step) > u(hollow)
        assert u(top + step) < u(top)
    assert u(bridge + along) < u(bridge)  # down into the next hollow
    assert u(bridge + across) > u(bridge)  # up towards a top


# -----------------------------------------------------------------------------
# work along paths
# -----------------------------------------------------------------------------
def test_work_of_a_constant_force_is_force_dot_displacement():
    def push(r):
        return np.broadcast_to([0.5, -0.2], r.shape)

    path = np.linspace([0.0, 0.0], [2.0, 1.0], 7)
    assert math.isclose(energy.work(push, path), 0.5 * 2 - 0.2 * 1)


def wiggly_path(a, b, n_points, bulge=1.0):
    s = np.linspace(0.0, 1.0, n_points)[:, np.newaxis]
    normal = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
    wiggle = bulge * np.sin(np.pi * s) + 0.3 * np.sin(5 * np.pi * s)
    return a + s * (b - a) + wiggle * normal


def test_work_of_a_conservative_force_is_the_drop_in_potential():
    start, end = SITES["hollow"], SITES["top"] + [2.46, 0.0]
    u = energy.hexagonal_surface(np.array([start, end]))[0]
    exact = u[0] - u[1]
    errors = [
        abs(energy.work(surface_force, wiggly_path(start, end, n)) - exact)
        for n in (401, 801, 1601)
    ]
    assert errors[-1] < 1e-5
    # the midpoint rule: halving the chords quarters the error
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


def test_work_reverses_sign_with_the_path():
    path = wiggly_path(SITES["hollow"], SITES["top"], 501)
    forwards = energy.work(surface_force, path)
    backwards = energy.work(surface_force, path[::-1])
    assert math.isclose(forwards, -backwards, rel_tol=1e-12)


def test_swirl_circulation_is_twice_its_strength_times_the_area():
    c, radius = 0.02, 1.3
    angle = np.linspace(0.0, 2 * np.pi, 4001)[:, np.newaxis]
    loop = radius * np.hstack([np.cos(angle), np.sin(angle)]) + [0.4, -0.2]
    circulation = energy.work(lambda r: energy.swirl(r, c), loop)
    # The swirl is linear, so the midpoint rule is exact on each chord
    # and integrates the inscribed polygon rather than the circle.
    n = len(loop) - 1
    polygon_area = 0.5 * n * radius**2 * np.sin(2 * np.pi / n)
    assert math.isclose(circulation, 2 * c * polygon_area, rel_tol=1e-12)
    assert math.isclose(circulation, 2 * c * np.pi * radius**2, rel_tol=1e-6)


def test_swirl_work_on_two_paths_differs_by_twice_strength_times_area():
    c, bulge = 0.02, 1.2
    start, end = SITES["hollow"], SITES["top"] + [2.46, 0.0]
    s = np.linspace(0.0, 1.0, 4001)[:, np.newaxis]
    normal = np.array([-(end - start)[1], (end - start)[0]])
    normal /= np.linalg.norm(end - start)
    arc = start + s * (end - start) + bulge * np.sin(np.pi * s) * normal
    straight = np.linspace(start, end, 2)
    area = 2 * bulge * np.linalg.norm(end - start) / np.pi

    def force(r):
        return energy.swirl(r, c)

    # The arc bulges to the left of start -> end, so the arc followed by
    # the reversed line is clockwise: W_arc − W_line = −2 c area.
    difference = energy.work(force, arc) - energy.work(force, straight)
    assert math.isclose(difference, -2 * c * area, rel_tol=1e-6)


# -----------------------------------------------------------------------------
# conservation of energy along motions
# -----------------------------------------------------------------------------
MASS_LI = 6.94


def launch(force, kinetic, direction, t):
    """A lithium atom leaving a hollow with the given kinetic energy."""
    direction = np.asarray(direction, dtype=float)
    speed = math.sqrt(2 * kinetic / (MASS_LI * units.MV2_TO_EV))
    v0 = speed * direction / np.linalg.norm(direction)
    return dynamics.solve_newton(
        force,
        MASS_LI,
        SITES["hollow"],
        v0,
        t,
        force_to_accel=units.FORCE_TO_ACCEL,
    )


def total_energy(r, v):
    kinetic = 0.5 * MASS_LI * np.sum(v**2, axis=-1) * units.MV2_TO_EV
    return kinetic + energy.hexagonal_surface(r)[0], kinetic


def test_energy_is_conserved_on_the_surface():
    t = np.linspace(0.0, 1000.0, 2001)
    r, v = launch(surface_force, 0.32, (1.0, 0.35), t)
    total, kinetic = total_energy(r, v)
    assert np.ptp(kinetic) > 0.2  # energy really is exchanged
    assert np.max(np.abs(total - total[0])) < 1e-8
    # never in a forbidden region: U ≤ E everywhere along the motion
    assert np.all(energy.hexagonal_surface(r)[0] <= total[0] + 1e-8)


def test_atom_below_the_barrier_stays_in_its_hollow():
    t = np.linspace(0.0, 1000.0, 2001)
    r, _ = launch(surface_force, 0.25, (1.0, 0.35), t)
    # the hollow's region lies within a/√3 of its centre
    reach = energy.SURFACE_SPACING / np.sqrt(3.0)
    assert np.max(np.linalg.norm(r, axis=-1)) < reach


def test_energy_changes_by_the_work_of_a_nonconservative_force():
    c = 0.005

    def force(r):
        return surface_force(r) + energy.swirl(r, c)

    t = np.linspace(0.0, 1000.0, 10001)
    r, v = launch(force, 0.32, (1.0, 0.35), t)
    total, _ = total_energy(r, v)
    power = np.sum(energy.swirl(r, c) * v, axis=-1)
    supplied = cumulative_trapezoid(power, t, initial=0.0)
    assert np.max(np.abs(total - total[0])) > 1e-2  # the swirl changes E
    assert np.max(np.abs(total - total[0] - supplied)) < 1e-6


def test_carts_speeds_follow_from_energy_and_momentum():
    # Section 3.8: 1 kg and 3 kg pushed apart by a 200 N/m spring
    # compressed by 0.1 m; the spring pushes but cannot pull.
    masses, k, natural, gap0 = np.array([1.0, 3.0]), 200.0, 0.30, 0.20

    def push(r):
        compression = max(natural - (r[1, 0] - r[0, 0]), 0.0)
        return np.array([[-k * compression], [k * compression]])

    t = np.linspace(0.0, 0.6, 1201)
    r, v = dynamics.solve_newton(
        push,
        masses,
        [[0.0], [gap0]],
        np.zeros((2, 1)),
        t,
        force_to_accel=1.0,
    )
    compression = np.maximum(natural - (r[:, 1, 0] - r[:, 0, 0]), 0.0)
    total = 0.5 * np.sum(masses * v[:, :, 0] ** 2, axis=-1)
    total += 0.5 * k * compression**2
    assert np.max(np.abs(total - 1.0)) < 1e-8  # ½ k (0.1 m)² = 1 J
    # ½ k c² = ½ m2 v2² (m2/m1 + 1), with m1 v1 = −m2 v2
    v2 = math.sqrt(
        k * (natural - gap0) ** 2 * masses[0] / (masses[1] * masses.sum())
    )
    assert math.isclose(v[-1, 1, 0], v2, rel_tol=1e-8)
    assert math.isclose(v[-1, 0, 0], -3.0 * v2, rel_tol=1e-8)
