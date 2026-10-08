"""Hamiltonian mechanics and phase space: Chapter 7.

Motion as a flow of states (q, p) in phase space:

- ``flow``: Hamilton's equations, dq/dt = ∂H/∂p and dp/dt = −∂H/∂q, for
  one degree of freedom, solved accurately for many starting states at
  once (Section 7.2);
- ``polygon_area``: the area inside a closed chain of points, for following
  a blob of states as it flows (Section 7.4);
- ``flow_jacobian``: the Jacobian matrix of the map that carries a state
  forward by a time t, by central differences (Section 7.4);
- ``euler_step`` and ``symplectic_euler_step``: one step of two simple
  methods for a separable H = K(p) + U(q), the first of which does not
  keep area and the second of which does (Section 7.7).

A damping rate ``gamma`` adds the drag −γp to dp/dt, so that the same
functions show a flow that does not keep area.

Units
-----
Everything works in any consistent units: q in its own unit, p in energy ×
time per unit of q, t in the unit of time, γ per unit of time. The
examples use reduced units.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


def flow(
    dh_dq: Callable[[NDArray, NDArray], NDArray],
    dh_dp: Callable[[NDArray, NDArray], NDArray],
    q0: ArrayLike,
    p0: ArrayLike,
    times: ArrayLike,
    gamma: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Carry states (q, p) along Hamilton's equations.

    Parameters
    ----------
    dh_dq, dh_dp : callable
        ∂H/∂q and ∂H/∂p, each taking arrays q and p of the same shape.
    q0, p0 : array_like, shape (n,) or scalar
        Starting states, one per point.
    times : array_like, shape (n_times,)
        Increasing times at which to report the states.
    gamma : float
        Drag rate; 0 for Hamilton's equations as they stand.

    Returns
    -------
    q, p : ndarray, shape (n_times, n)

    Examples
    --------
    A spring, H = (q² + p²)/2, a quarter turn on: (1, 0) goes to (0, −1).

    >>> q, p = flow(lambda q, p: q, lambda q, p: p, [1.0], [0.0],
    ...             [0.0, np.pi / 2])
    >>> print(q[-1].round(8), p[-1].round(8))
    [0.] [-1.]
    """
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    p0 = np.atleast_1d(np.asarray(p0, dtype=float))
    n = q0.size
    times = np.asarray(times, dtype=float)

    def rates(_t, y):
        q, p = y[:n], y[n:]
        return np.concatenate([dh_dp(q, p), -dh_dq(q, p) - gamma * p])

    solution = solve_ivp(
        rates,
        (times[0], times[-1]),
        np.concatenate([q0, p0]),
        t_eval=times,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    return solution.y[:n].T, solution.y[n:].T


def polygon_area(q: ArrayLike, p: ArrayLike) -> float:
    """Area inside the closed chain of points (q_k, p_k), by the shoelace rule.

    The area is ½|Σ_k (q_k p_{k+1} − q_{k+1} p_k)|, the chain closing from
    the last point back to the first (Section 7.4).

    Examples
    --------
    >>> polygon_area([0, 1, 1, 0], [0, 0, 1, 1])
    1.0
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(0.5 * abs(np.sum(q * np.roll(p, -1) - np.roll(q, -1) * p)))


def flow_jacobian(
    dh_dq: Callable[[NDArray, NDArray], NDArray],
    dh_dp: Callable[[NDArray, NDArray], NDArray],
    q0: float,
    p0: float,
    duration: float,
    gamma: float = 0.0,
    step: float = 1e-5,
) -> NDArray:
    """The 2 × 2 Jacobian of the map (q0, p0) → (q(t), p(t)).

    Column 1 holds the derivatives with respect to q0, column 2 those with
    respect to p0, each by a central difference of size ``step``. Its
    determinant is the factor by which the map stretches small areas.
    """
    starts_q = np.array([q0 + step, q0 - step, q0, q0])
    starts_p = np.array([p0, p0, p0 + step, p0 - step])
    q, p = flow(dh_dq, dh_dp, starts_q, starts_p, [0.0, duration], gamma)
    q, p = q[-1], p[-1]
    return np.array(
        [
            [(q[0] - q[1]) / (2 * step), (q[2] - q[3]) / (2 * step)],
            [(p[0] - p[1]) / (2 * step), (p[2] - p[3]) / (2 * step)],
        ]
    )


def euler_step(
    dh_dq: Callable[[NDArray, NDArray], NDArray],
    dh_dp: Callable[[NDArray, NDArray], NDArray],
    q: ArrayLike,
    p: ArrayLike,
    dt: float,
) -> tuple[NDArray, NDArray]:
    """One step of the forward Euler method for Hamilton's equations.

    Both rates are taken at the start of the step:

        q' = q + δt ∂H/∂p(q, p),    p' = p − δt ∂H/∂q(q, p).

    For the spring this multiplies H, and every area, by 1 + ω²δt² at
    each step (Section 7.7).

    Parameters
    ----------
    dh_dq, dh_dp : callable
        ∂H/∂q and ∂H/∂p, as for ``flow``.
    q, p : array_like
        The state at the start of the step.
    dt : float
        The time step δt, in the unit of time.

    Examples
    --------
    The spring H = (q² + p²)/2 from (1, 0), a step of 0.1:

    >>> q, p = euler_step(lambda q, p: q, lambda q, p: p, 1.0, 0.0, 0.1)
    >>> print(float(q), float(p), float(q**2 + p**2))
    1.0 -0.1 1.01
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    return q + dt * dh_dp(q, p), p - dt * dh_dq(q, p)


def symplectic_euler_step(
    dh_dq: Callable[[NDArray, NDArray], NDArray],
    dh_dp: Callable[[NDArray, NDArray], NDArray],
    q: ArrayLike,
    p: ArrayLike,
    dt: float,
) -> tuple[NDArray, NDArray]:
    """One step of the symplectic Euler method, momentum first.

    The momentum is moved with the force at the old position, and the
    position with the new momentum:

        p' = p − δt ∂H/∂q(q),    q' = q + δt ∂H/∂p(p').

    For a separable H = K(p) + U(q), ∂H/∂q depends on q alone and ∂H/∂p
    on p alone, and the map keeps area exactly (Section 7.7).

    Parameters
    ----------
    dh_dq, dh_dp : callable
        ∂H/∂q and ∂H/∂p of a separable H, called as for ``flow``.
    q, p : array_like
        The state at the start of the step.
    dt : float
        The time step δt, in the unit of time.

    Examples
    --------
    >>> q, p = symplectic_euler_step(lambda q, p: q, lambda q, p: p,
    ...                              1.0, 0.0, 0.1)
    >>> print(float(q), float(p))
    0.99 -0.1
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    p_new = p - dt * dh_dq(q, p)
    return q + dt * dh_dp(q, p_new), p_new
