"""Forces and Newton's laws: Chapter 2.

Newton's second law turns forces into accelerations. This module holds
the pieces of Chapter 2 that the book computes:

- ``change_frame``: the same motion seen from a frame of reference
  moving at constant velocity (Section 2.2);
- ``drag_motion``: the exact fall through a liquid with a drag
  proportional to the velocity (Section 2.7);
- ``momentum``, ``total_momentum`` and ``centre_of_mass`` for a group of
  bodies (Section 2.9);
- ``acceleration``: the second law for atoms, in eV, Å, fs and amu
  (Section 2.10);
- ``solve_newton``: an accurate numerical solution of the equations of
  motion (Section 2.6), for checking and for figures.

Units
-----
Only ``acceleration`` assumes the units of atoms. Everything else works
in any consistent units, and ``solve_newton`` asks to be told, through
its ``force_to_accel`` argument, how a force divided by a mass becomes
an acceleration: 1.0 in SI units, ``units.FORCE_TO_ACCEL`` for atoms.

Array convention
----------------
A group of N bodies has positions and velocities of shape ``(N, d)``,
one row per body, and masses of shape ``(N,)``. A path through time adds
a first axis, ``(n_times, N, d)``.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from . import units


def acceleration(forces: ArrayLike, masses: ArrayLike) -> NDArray:
    """Newton's second law for atoms, a = F / m, in Å/fs² (Section 2.10).

    Parameters
    ----------
    forces : array_like, shape (N, 3)
        Force on each atom, in eV/Å.
    masses : array_like, shape (N,)
        Mass of each atom, in amu.

    Returns
    -------
    ndarray, shape (N, 3)
        Acceleration of each atom, in Å/fs².

    Examples
    --------
    A lithium atom (6.94 amu) under a force of 1 eV/Å along x:

    >>> a = acceleration([[1.0, 0.0, 0.0]], [6.94])
    >>> print(f"{a[0, 0]:.4e} Å/fs²")
    1.3903e-03 Å/fs²
    """
    forces = np.asarray(forces, dtype=float)
    masses = np.asarray(masses, dtype=float)
    # Each atom's single mass divides all three components of its force.
    return forces / masses[..., np.newaxis] * units.FORCE_TO_ACCEL


def momentum(masses: ArrayLike, velocities: ArrayLike) -> NDArray:
    """Momentum of each body, p = m v (Section 2.9).

    Parameters
    ----------
    masses : array_like, shape (N,)
        Masses.
    velocities : array_like, shape (N, d)
        Velocities.

    Returns
    -------
    ndarray, shape (N, d)
        Momentum of each body, in mass times velocity units.

    Examples
    --------
    >>> momentum([1.0, 3.0], [[-1.5], [0.5]])
    array([[-1.5],
           [ 1.5]])
    """
    masses = np.asarray(masses, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    return masses[..., np.newaxis] * velocities


def total_momentum(masses: ArrayLike, velocities: ArrayLike) -> NDArray:
    """Total momentum of a group, P = Σ m_i v_i (Section 2.9).

    Parameters
    ----------
    masses : array_like, shape (N,)
        Masses.
    velocities : array_like, shape (N, d) or (n_times, N, d)
        Velocities, optionally at a series of times.

    Returns
    -------
    ndarray, shape (d,) or (n_times, d)
        Total momentum, at each time if times were given.

    Examples
    --------
    Two carts pushed apart from rest have equal and opposite momenta:

    >>> total_momentum([1.0, 3.0], [[-1.5], [0.5]])
    array([0.])
    """
    # Add the momenta of the bodies, which sit on the second-to-last axis.
    return np.sum(momentum(masses, velocities), axis=-2)


def centre_of_mass(masses: ArrayLike, positions: ArrayLike) -> NDArray:
    """Centre of mass, R = Σ m_i r_i / M with M = Σ m_i (Section 2.9).

    Parameters
    ----------
    masses : array_like, shape (N,)
        Masses.
    positions : array_like, shape (N, d) or (n_times, N, d)
        Positions, optionally at a series of times.

    Returns
    -------
    ndarray, shape (d,) or (n_times, d)
        The centre of mass, at each time if times were given.

    Examples
    --------
    Carts of 1 kg at x = 0 and 3 kg at x = 0.2 m:

    >>> centre_of_mass([1.0, 3.0], [[0.0], [0.2]]).round(6)
    array([0.15])
    """
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    # Weight each position by its mass, as momentum weights a velocity,
    # then add over the bodies on the second-to-last axis.
    weighted = np.sum(masses[..., np.newaxis] * positions, axis=-2)
    return weighted / masses.sum()


def change_frame(
    times: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    frame_velocity: ArrayLike,
    frame_origin: ArrayLike = 0.0,
) -> tuple[NDArray, NDArray]:
    """The same motion seen from a uniformly moving frame (Section 2.2).

    If the origin of the new frame of reference starts at r_0 and moves
    with velocity u,

        r' = r − r_0 − u t,
        v' = v − u,

    and the acceleration is unchanged.

    Parameters
    ----------
    times : array_like, shape (n_times,)
        Times.
    positions, velocities : array_like, shape (n_times, ..., d)
        The motion seen from the original frame.
    frame_velocity : array_like, shape (d,)
        Velocity u of the new frame, seen from the original one.
    frame_origin : array_like, shape (d,) or scalar
        Position r_0 of the new frame's origin at t = 0.

    Returns
    -------
    positions, velocities : ndarray, shape (n_times, ..., d)
        The same motion seen from the new frame.

    Examples
    --------
    A ball leaves a passenger's hand at (5, 6) m/s as seen from the
    platform. In the train, which moves at (5, 0) m/s, it goes straight
    up at 6 m/s:

    >>> r, v = change_frame([0.0], [[0.0, 0.0]], [[5.0, 6.0]], [5.0, 0.0])
    >>> v
    array([[0., 6.]])
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    u = np.asarray(frame_velocity, dtype=float)

    # Give each time as many trailing axes as a position has, so that
    # u * t is a displacement for every body at every time.
    t = times.reshape(times.shape + (1,) * (positions.ndim - 1))
    origin_now = np.asarray(frame_origin, dtype=float) + u * t
    return positions - origin_now, velocities - u


def drag_motion(
    t: ArrayLike, x0: float, v0: float, g: float, gamma: float
) -> tuple[NDArray, NDArray]:
    """Exact fall under gravity with a linear drag (Section 2.7).

    Measuring x downwards, the equation of motion is dv/dt = g − γ v. Its
    solution approaches the terminal velocity v_∞ = g / γ:

        v(t) = v_∞ + (v0 − v_∞) e^(−γt),
        x(t) = x0 + v_∞ t + (v0 − v_∞)(1 − e^(−γt)) / γ.

    Parameters
    ----------
    t : array_like
        Times, in s.
    x0, v0 : float
        Depth (m) and downward velocity (m/s) at t = 0.
    g : float
        Acceleration due to gravity, in m/s².
    gamma : float
        Drag rate γ = b / m, in 1/s.

    Returns
    -------
    x, v : ndarray
        Depth and downward velocity at each time.

    Examples
    --------
    After 1/γ a bead released from rest has 63% of its terminal velocity:

    >>> x, v = drag_motion(0.2, 0.0, 0.0, g=9.80665, gamma=5.0)
    >>> print(f"{v / (9.80665 / 5.0):.4f}")
    0.6321
    """
    t = np.asarray(t, dtype=float)
    v_inf = g / gamma
    decay = np.exp(-gamma * t)  # the gap v − v_∞ shrinks by this factor

    v = v_inf + (v0 - v_inf) * decay
    x = x0 + v_inf * t + (v0 - v_inf) * (1.0 - decay) / gamma
    return x, v


def solve_newton(
    force: Callable[..., NDArray],
    masses: ArrayLike,
    r0: ArrayLike,
    v0: ArrayLike,
    times: ArrayLike,
    *,
    force_to_accel: float,
    velocity_dependent: bool = False,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> tuple[NDArray, NDArray]:
    """Solve Newton's equations accurately, as a reference (Section 2.6).

    The equations of motion m d²r/dt² = F are rewritten as the
    first-order system of Section 2.6,

        dr/dt = v,    dv/dt = F / m,

    and handed to SciPy's adaptive eighth-order Runge-Kutta solver with
    tight tolerances. Chapter 9 builds the integrators that molecular
    dynamics actually uses; until then this solver supplies motions to
    compare against.

    Parameters
    ----------
    force : callable
        ``force(r)`` returns the forces at positions ``r``, with the shape
        of ``r``. With ``velocity_dependent=True`` it is called as
        ``force(r, v)``, as needed for drag.
    masses : array_like, shape (N,) or scalar
        Masses.
    r0, v0 : array_like, shape (N, d) or (d,)
        Positions and velocities at the first time.
    times : array_like, shape (n_times,)
        Increasing times at which to report the motion.
    force_to_accel : float
        The factor that turns force / mass into an acceleration: 1.0 for
        SI units, ``units.FORCE_TO_ACCEL`` for eV, Å, fs and amu. It has
        no default, so that the units are always stated.
    velocity_dependent : bool
        Whether ``force`` also takes the velocities.
    rtol, atol : float
        Relative and absolute tolerances of the solver.

    Returns
    -------
    positions, velocities : ndarray, shape (n_times, ...)
        The motion at the requested times.

    Examples
    --------
    A ball of 0.145 kg thrown up at 12 m/s, in SI units, 1 s later; the
    only force is its weight, m g downwards:

    >>> def weight(r):
    ...     return np.array([-0.145 * 9.80665])
    >>> r, v = solve_newton(weight, 0.145, [0.0], [12.0], [0.0, 1.0],
    ...                     force_to_accel=1.0)
    >>> print(f"{r[-1, 0]:.3f} m, {v[-1, 0]:.3f} m/s")
    7.097 m, 2.193 m/s
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if masses.ndim == 1:
        masses = masses[:, np.newaxis]  # one mass per row of r
    shape = r0.shape
    size = r0.size

    # The solver works with one flat vector y holding every position
    # followed by every velocity; ``unpack`` restores their shapes.
    def unpack(y):
        return y[:size].reshape(shape), y[size:].reshape(shape)

    def rates(_t, y):
        r, v = unpack(y)
        f = force(r, v) if velocity_dependent else force(r)
        a = f / masses * force_to_accel
        return np.concatenate([v.ravel(), a.ravel()])

    y0 = np.concatenate([r0.ravel(), v0.ravel()])
    times = np.asarray(times, dtype=float)
    solution = solve_ivp(
        rates,
        (times[0], times[-1]),
        y0,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    y = solution.y.T  # one row per requested time
    positions = y[:, :size].reshape(-1, *shape)
    velocities = y[:, size:].reshape(-1, *shape)
    return positions, velocities
