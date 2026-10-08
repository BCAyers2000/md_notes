"""Chapter 7: Hamilton's equations keep energy and phase-space area, the
drag shrinks area at its rate, reversing the momenta retraces the path,
symmetric potentials give forces with no total and no torque, and of two
step maps only the symplectic one keeps area."""

import math

import numpy as np
import pytest

from mdlab import hamiltonian

# the pendulum in reduced units: H = p²/2 + 1 − cos q


def dh_dq(q, p):
    return np.sin(q)


def dh_dp(q, p):
    return p


def energy(q, p):
    return 0.5 * p**2 + 1 - np.cos(q)


def blob(centre=(1.0, 0.5), radius=0.2, n=2000):
    s = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return centre[0] + radius * np.cos(s), centre[1] + radius * np.sin(s)


def test_energy_is_conserved_along_the_flow():
    q0, p0 = np.array([0.5, 2.0, 3.0]), np.array([0.0, 0.5, 0.1])
    q, p = hamiltonian.flow(dh_dq, dh_dp, q0, p0, np.linspace(0, 10, 101))
    assert np.allclose(energy(q, p), energy(q0, p0), atol=1e-10)


def test_area_of_a_blob_is_kept():
    q0, p0 = blob()
    q, p = hamiltonian.flow(dh_dq, dh_dp, q0, p0, [0.0, 3.0])
    before = hamiltonian.polygon_area(q[0], p[0])
    after = hamiltonian.polygon_area(q[-1], p[-1])
    assert after == pytest.approx(before, rel=1e-4)
    assert before == pytest.approx(math.pi * 0.2**2, rel=1e-5)


@pytest.mark.parametrize("gamma", [0.3, 1.0])
def test_drag_shrinks_area_at_its_rate(gamma):
    q0, p0 = blob()
    q, p = hamiltonian.flow(dh_dq, dh_dp, q0, p0, [0.0, 2.0], gamma=gamma)
    ratio = hamiltonian.polygon_area(q[-1], p[-1]) / hamiltonian.polygon_area(
        q[0], p[0]
    )
    assert ratio == pytest.approx(math.exp(-gamma * 2.0), rel=1e-4)


@pytest.mark.parametrize(("gamma", "expected"), [(0.0, 1.0), (0.5, None)])
def test_jacobian_determinant(gamma, expected):
    jac = hamiltonian.flow_jacobian(dh_dq, dh_dp, 1.0, 0.5, 2.5, gamma=gamma)
    det = np.linalg.det(jac)
    target = math.exp(-gamma * 2.5) if expected is None else expected
    assert det == pytest.approx(target, rel=1e-6)


def test_reversing_the_momenta_retraces_the_path():
    q, p = hamiltonian.flow(dh_dq, dh_dp, [1.0], [0.7], [0.0, 4.0])
    qb, pb = hamiltonian.flow(dh_dq, dh_dp, q[-1], -p[-1], [0.0, 4.0])
    assert qb[-1, 0] == pytest.approx(1.0, abs=1e-8)
    assert pb[-1, 0] == pytest.approx(-0.7, abs=1e-8)


def test_poisson_brackets():
    sp = pytest.importorskip("sympy")
    q, p = sp.symbols("q p")

    def bracket(a, b):
        return sp.diff(a, q) * sp.diff(b, p) - sp.diff(a, p) * sp.diff(b, q)

    h = p**2 / 2 + 1 - sp.cos(q)
    assert bracket(q, p) == 1
    assert sp.simplify(bracket(h, h)) == 0
    assert sp.simplify(bracket(q, h) - sp.diff(h, p)) == 0
    assert sp.simplify(bracket(p, h) + sp.diff(h, q)) == 0


def test_separatrix_creeps_to_the_top():
    # on the separatrix θ = 4 arctan(e^t) − π, so π − θ ≈ 4 e^{−t}
    times = np.linspace(0.0, 4.0, 9)
    q, _ = hamiltonian.flow(dh_dq, dh_dp, [0.0], [2.0], times)
    exact = 4 * np.arctan(np.exp(times)) - np.pi
    assert np.allclose(q[:, 0], exact, atol=1e-8)
    assert np.pi - q[-1, 0] == pytest.approx(4 * math.exp(-4.0), rel=1e-3)


def test_liouville_series_of_the_spring():
    sp = pytest.importorskip("sympy")
    q, p, t, m, w = sp.symbols("q p t m omega", positive=True)
    h = p**2 / (2 * m) + m * w**2 * q**2 / 2

    def liouville(a):
        return sp.diff(a, q) * sp.diff(h, p) - sp.diff(a, p) * sp.diff(h, q)

    term, series = q, 0
    for n in range(10):
        series += t**n / sp.factorial(n) * term
        term = sp.expand(liouville(term))
    exact = q * sp.cos(w * t) + p / (m * w) * sp.sin(w * t)
    assert sp.simplify(sp.series(exact, t, 0, 10).removeO() - series) == 0


def angle_energy(r, k=2.0, theta0=1.9):
    """½k(θ − θ0)² for the angle at atom 1 between atoms 0 and 2."""
    a, b = r[0] - r[1], r[2] - r[1]
    cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    return 0.5 * k * (np.arccos(cos) - theta0) ** 2


def test_three_body_forces_have_no_total_and_no_torque():
    rng = np.random.default_rng(7)
    r = rng.normal(size=(3, 3))
    h = 1e-6
    forces = np.zeros_like(r)
    for i in range(3):
        for a in range(3):
            up, down = r.copy(), r.copy()
            up[i, a] += h
            down[i, a] -= h
            forces[i, a] = -(angle_energy(up) - angle_energy(down)) / (2 * h)
    assert np.abs(forces).max() > 0.1
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-8)
    assert np.allclose(np.cross(r, forces).sum(axis=0), 0.0, atol=1e-8)


SPRING_DT, SPRING_W = 0.01, 10.0  # s, rad/s


def spring_dh_dq(q, p):
    return 0.5 * SPRING_W**2 * q  # m = 0.5 kg, k = 50 N/m


def spring_dh_dp(q, p):
    return p / 0.5


def spring_energy(q, p):
    return p**2 / (2 * 0.5) + 0.5 * 50.0 * q**2


def test_euler_multiplies_the_spring_energy():
    q, p = 0.04, 0.0
    factor = 1 + (SPRING_W * SPRING_DT) ** 2
    for _ in range(5):
        e0 = spring_energy(q, p)
        q, p = hamiltonian.euler_step(
            spring_dh_dq, spring_dh_dp, q, p, SPRING_DT
        )
        assert spring_energy(q, p) == pytest.approx(factor * e0, rel=1e-12)


def step_determinant(step, dh_q, dh_p, q0, p0, dt, h=1e-6):
    qa, pa = step(dh_q, dh_p, q0 + h, p0, dt)
    qb, pb = step(dh_q, dh_p, q0 - h, p0, dt)
    qc, pc = step(dh_q, dh_p, q0, p0 + h, dt)
    qd, pd = step(dh_q, dh_p, q0, p0 - h, dt)
    jac = np.array([[qa - qb, qc - qd], [pa - pb, pc - pd]]) / (2 * h)
    return np.linalg.det(jac)


def test_step_maps_and_area():
    euler = step_determinant(
        hamiltonian.euler_step,
        spring_dh_dq,
        spring_dh_dp,
        0.03,
        0.2,
        SPRING_DT,
    )
    assert euler == pytest.approx(1 + (SPRING_W * SPRING_DT) ** 2, rel=1e-8)
    for q0, p0 in [(0.3, 0.1), (2.0, -1.5)]:  # the pendulum, nonlinear
        det = step_determinant(
            hamiltonian.symplectic_euler_step, dh_dq, dh_dp, q0, p0, 0.3
        )
        assert det == pytest.approx(1.0, rel=1e-8)


def test_symplectic_euler_keeps_the_spring_energy_in_a_band():
    q, p = 0.04, 0.0
    e0 = spring_energy(q, p)
    worst = 0.0
    for _ in range(10000):
        q, p = hamiltonian.symplectic_euler_step(
            spring_dh_dq, spring_dh_dp, q, p, SPRING_DT
        )
        worst = max(worst, abs(spring_energy(q, p) / e0 - 1))
    assert worst == pytest.approx(
        1 / (1 - SPRING_W * SPRING_DT / 2) - 1, rel=1e-2
    )
