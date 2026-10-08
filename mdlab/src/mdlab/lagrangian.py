"""Lagrangian mechanics: Chapter 6.

Motion found from the kinetic and potential energies alone:

- ``action``: the action of a path sampled at a series of times, the
  quantity that the true motion makes stationary (Section 6.2);
- ``wire_acceleration`` and ``solve_wire``: a bead sliding without
  friction on a wire of any shape y = h(x), the constrained motion that
  Chapter 3 could only describe through its energy (Section 6.4);
- ``rod_multiplier``: the Lagrange multiplier that holds a pendulum's bob
  at a fixed distance from its pivot, which gives the rod's force
  (Section 6.6).

Units
-----
Everything works in any consistent units; the examples use SI.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import g as GRAVITY
from scipy.integrate import solve_ivp


def action(
    path: ArrayLike,
    times: ArrayLike,
    mass: float,
    potential: Callable[[NDArray], NDArray],
) -> float:
    """The action ∫ (K − U) dt of a path through the sampled points.

    Between samples the path is taken as a straight line at constant
    velocity, so each step contributes ½ m (Δx/Δt)² Δt of kinetic energy,
    and the potential energy is averaged over its two ends (the trapezium
    rule).

    Parameters
    ----------
    path : array_like, shape (n_times,) or (n_times, d)
        Positions at the sample times, in m (or any consistent unit).
    times : array_like, shape (n_times,)
        Increasing times, in s.
    mass : float
        The mass, in the unit that makes ½ m v² an energy in the unit of U.
    potential : callable
        ``potential(x)`` returns U at each of an array of positions, in J.

    Returns
    -------
    float
        The action, in J s.

    Examples
    --------
    A ball of 1 kg at rest for 1 s, with U = 0, has zero action:

    >>> action([0.0, 0.0, 0.0], [0.0, 0.5, 1.0], 1.0, lambda x: 0 * x)
    0.0
    """
    path = np.asarray(path, dtype=float)
    times = np.asarray(times, dtype=float)
    if path.ndim == 1:
        path = path[:, np.newaxis]
    steps = np.diff(times)
    velocity = np.diff(path, axis=0) / steps[:, np.newaxis]
    kinetic = 0.5 * mass * np.sum(velocity**2, axis=1)  # on each step
    energy = np.asarray(potential(path.squeeze()), dtype=float)
    mean_u = 0.5 * (energy[1:] + energy[:-1])  # trapezium rule
    return float(np.sum((kinetic - mean_u) * steps))


def wire_acceleration(
    v: ArrayLike,
    slope: ArrayLike,
    curvature: ArrayLike,
    g: float = GRAVITY,
) -> NDArray:
    """d²x/dt² for a bead sliding without friction on a wire y = h(x).

    From the Lagrangian ½ m (1 + h'²) ẋ² − m g h(x) (Section 6.4),

        ẍ = −h' (g + h'' ẋ²) / (1 + h'²),

    with h' the slope and h'' the curvature of the wire at the bead. The
    mass cancels.

    Parameters
    ----------
    v : array_like
        Horizontal velocity dx/dt, in m/s (or any consistent unit).
    slope, curvature : array_like
        h' (dimensionless) and h'' (per unit length) at the bead.
    g : float
        Acceleration of free fall, in the units of v per unit time.

    Examples
    --------
    On a straight slope h = x, released from rest, ẍ = −g/2:

    >>> float(wire_acceleration(0.0, 1.0, 0.0, g=9.8))
    -4.9
    """
    v = np.asarray(v, dtype=float)
    slope = np.asarray(slope, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    return -slope * (g + curvature * v**2) / (1 + slope**2)


def solve_wire(
    shape: Callable[[float], tuple[float, float, float]],
    x0: float,
    v0: float,
    times: ArrayLike,
    g: float = GRAVITY,
) -> tuple[NDArray, NDArray]:
    """The motion of a bead on the wire y = h(x), accurately.

    Parameters
    ----------
    shape : callable
        ``shape(x)`` returns (h, h', h'') at a single x, h in m and h''
        in 1/m.
    x0, v0 : float
        Starting position (m) and horizontal velocity dx/dt (m/s).
    times : array_like
        Increasing times at which to report the motion, in s.
    g : float
        Acceleration of free fall, in m/s².

    Returns
    -------
    x, v : ndarray
        Horizontal position and velocity dx/dt at the requested times.
    """
    times = np.asarray(times, dtype=float)

    def rates(_t, y):
        _, slope, curvature = shape(y[0])
        return [
            y[1],
            float(wire_acceleration(y[1], slope, curvature, g)),
        ]

    solution = solve_ivp(
        rates,
        (times[0], times[-1]),
        [x0, v0],
        t_eval=times,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0], solution.y[1]


def rod_multiplier(
    position: ArrayLike,
    velocity: ArrayLike,
    mass: float,
    gravity: ArrayLike,
) -> float:
    """Lagrange multiplier λ of a bob held at a fixed distance from a pivot.

    The constraint is ½(|r|² − L²) = 0 with the pivot at the origin, so
    the rod pushes with λ r and m r̈ = m g + λ r (Section 6.6).
    Differentiating the constraint twice, r·r̈ + |v|² = 0, which fixes

        λ = −m (|v|² + r·g) / |r|².

    The force of the rod, its pull towards the pivot (negative when it
    pushes), is −λ|r|.

    Parameters
    ----------
    position, velocity : array_like, shape (d,)
        r from the pivot (m) and v (m/s).
    mass : float
        The mass of the bob, in kg.
    gravity : array_like, shape (d,)
        The acceleration of free fall as a vector, in m/s².

    Returns
    -------
    float
        λ, in N/m (force per unit of the gradient r of the constraint).

    Examples
    --------
    A bob of 1 kg hanging at rest 1 m below the pivot, g = 9.8 m/s²
    downwards: the rod pulls up with 9.8 N.

    >>> lam = rod_multiplier([0.0, -1.0], [0.0, 0.0], 1.0, [0.0, -9.8])
    >>> round(-lam * 1.0, 6)
    9.8
    """
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    g = np.asarray(gravity, dtype=float)
    return float(-mass * (v @ v + r @ g) / (r @ r))
