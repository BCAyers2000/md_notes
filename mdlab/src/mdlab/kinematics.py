"""Describing motion: Chapter 1.

Kinematics describes how things move without asking why. This module
holds the two motions that Chapter 1 solves exactly, constant
acceleration (Section 1.4) and uniform circular motion (Section 1.7),
and the two ways of recovering a velocity from positions saved at equal
intervals of time (Section 1.9).

Nothing here involves force or mass, so any consistent units work: the
book uses metres and seconds for balls and stones, and ångströms and
femtoseconds for atoms.

Array convention
----------------
Time runs along the first axis. For motion in space the last axis holds
the Cartesian components, so a path of ``n`` positions in the plane has
shape ``(n, 2)``.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def constant_acceleration(
    t: ArrayLike, x0: ArrayLike, v0: ArrayLike, a: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Position and velocity under a constant acceleration (Section 1.4).

    The motion is

        x(t) = x0 + v0 t + a t² / 2,
        v(t) = v0 + a t,

    where x0 and v0 are the position and velocity at t = 0.

    Parameters
    ----------
    t : array_like, shape (n,) or scalar
        Times, measured from the instant at which x = x0 and v = v0.
    x0, v0, a : array_like, shape (d,) or scalar
        Starting position, starting velocity and the acceleration. Give
        vectors for motion in two or three dimensions.

    Returns
    -------
    x, v : ndarray, shape (n, d), (n,) or scalar
        Position and velocity at each time.

    Raises
    ------
    ValueError
        If x0, v0 or a has more than one axis; one body at a time.

    Examples
    --------
    A ball thrown up at 12 m/s, after 1 s:

    >>> x, v = constant_acceleration(1.0, 0.0, 12.0, -9.80665)
    >>> print(f"{x:.3f} m, {v:.3f} m/s")
    7.097 m, 2.193 m/s
    """
    t = np.asarray(t, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    a = np.asarray(a, dtype=float)
    vector_axes = max(x0.ndim, v0.ndim, a.ndim)
    if vector_axes > 1:
        raise ValueError("give one body: x0, v0 and a of shape (d,)")

    # For vector motion, give time a final axis of length one so that each
    # time multiplies a whole vector: t has shape (n, 1), a has shape (d,).
    if vector_axes == 1:
        t = t[..., np.newaxis]

    x = x0 + v0 * t + 0.5 * a * t**2
    v = v0 + a * t
    return x, v


def circular_motion(
    t: ArrayLike,
    radius: float,
    omega: float,
    centre: ArrayLike = (0.0, 0.0),
) -> tuple[NDArray, NDArray, NDArray]:
    """Uniform anticlockwise motion on a circle (Section 1.7).

    The angle grows as ω t, so

        r(t) = c + R (cos ωt, sin ωt),
        v(t) = R ω (−sin ωt, cos ωt),
        a(t) = −ω² (r − c),

    and the acceleration always points at the centre c.

    Parameters
    ----------
    t : array_like, shape (n,)
        Times.
    radius : float
        Radius R of the circle, in length units.
    omega : float
        Angular speed ω, in radians per unit time.
    centre : array_like, shape (2,)
        Centre c of the circle.

    Returns
    -------
    r, v, a : ndarray, shape (n, 2)
        Position, velocity and acceleration at each time.

    Examples
    --------
    The speed is R ω and the acceleration has magnitude ω² R:

    >>> r, v, a = circular_motion([0.0], radius=0.5, omega=4.0)
    >>> print(np.linalg.norm(v), np.linalg.norm(a))
    2.0 8.0
    """
    angle = omega * np.asarray(t, dtype=float)
    outward = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    along = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)

    position = np.asarray(centre, dtype=float) + radius * outward
    velocity = radius * omega * along
    acceleration = -(omega**2) * radius * outward
    return position, velocity, acceleration


def forward_difference(positions: ArrayLike, tau: float) -> NDArray:
    """Velocities from saved positions, (x(t + τ) − x(t)) / τ (Section 1.9).

    This is the slope of the secant over one interval. Its error is
    x'' τ / 2 to leading order, proportional to τ.

    Parameters
    ----------
    positions : array_like, shape (n, ...)
        Positions at the times t_0, t_0 + τ, t_0 + 2τ, ...
    tau : float
        Interval τ between saved positions.

    Returns
    -------
    ndarray, shape (n − 1, ...)
        Velocities at the first n − 1 times.

    Examples
    --------
    >>> forward_difference([0.0, 1.0, 4.0, 9.0], tau=1.0)
    array([1., 3., 5.])
    """
    x = np.asarray(positions, dtype=float)
    later, earlier = x[1:], x[:-1]
    return (later - earlier) / tau


def central_difference(positions: ArrayLike, tau: float) -> NDArray:
    """Velocities from saved positions, (x(t+τ) − x(t−τ)) / 2τ (Section 1.9).

    Using the positions on both sides cancels the even terms of the
    Taylor series, leaving an error of x''' τ² / 6 to leading order,
    proportional to τ².

    Parameters
    ----------
    positions : array_like, shape (n, ...)
        Positions at the times t_0, t_0 + τ, t_0 + 2τ, ...
    tau : float
        Interval τ between saved positions.

    Returns
    -------
    ndarray, shape (n − 2, ...)
        Velocities at the interior times t_1, ..., t_(n−2).

    Examples
    --------
    For x = t² the central difference is exact, v = 2t:

    >>> central_difference([0.0, 1.0, 4.0, 9.0], tau=1.0)
    array([2., 4.])
    """
    x = np.asarray(positions, dtype=float)
    return (x[2:] - x[:-2]) / (2.0 * tau)
