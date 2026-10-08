"""Rotation and angular momentum: Chapter 5.

The turning motion of a group of bodies, and how to take it away:

- ``angular_momentum`` and ``torque`` about a chosen point (Sections
  5.1-5.3);
- ``inertia_tensor``, the matrix that turns an angular velocity into an
  angular momentum (Section 5.5);
- ``angular_velocity``, the rigid rotation that carries a group's
  angular momentum, and ``remove_rigid_motion``, which subtracts it and
  the drift of the centre of mass (Section 5.6);
- ``spring_hessian``: the Hessian of a network of springs at rest, whose
  zero eigenvalues count the rigid motions (Section 5.7).

Units
-----
Everything works in any consistent units: kg, m and s, or amu, Å and fs,
in which an angular momentum is in amu Å² fs⁻¹ and a moment of inertia
in amu Å².

Array convention
----------------
As in ``dynamics``: N bodies have positions and velocities of shape
``(N, 3)`` and masses of shape ``(N,)``; a path through time adds a first
axis.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def angular_momentum(
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    origin: ArrayLike = (0.0, 0.0, 0.0),
) -> NDArray:
    """L = Σ m_i (r_i − O) × v_i, about the point O (Sections 5.2-5.3).

    Parameters
    ----------
    masses : array_like, shape (N,)
        Masses.
    positions, velocities : array_like, shape (N, 3) or (n_times, N, 3)
        Positions and velocities, optionally at a series of times.
    origin : array_like, shape (3,)
        The point O about which the angular momentum is taken.

    Returns
    -------
    ndarray, shape (3,) or (n_times, 3)

    Examples
    --------
    A 1 kg puck 0.5 m from the origin moving at 2 m/s round it:

    >>> angular_momentum([1.0], [[0.5, 0.0, 0.0]], [[0.0, 2.0, 0.0]])
    array([0., 0., 1.])
    """
    masses = np.asarray(masses, dtype=float)
    arms = np.asarray(positions, dtype=float) - np.asarray(origin, dtype=float)
    moments = np.cross(arms, velocities)  # (r_i − O) × v_i, one row per body
    return np.sum(masses[:, np.newaxis] * moments, axis=-2)


def torque(
    positions: ArrayLike,
    forces: ArrayLike,
    origin: ArrayLike = (0.0, 0.0, 0.0),
) -> NDArray:
    """G = Σ (r_i − O) × F_i, the total torque about O (Section 5.2).

    Examples
    --------
    A push of 10 N at right angles to a door, 0.8 m from its hinge:

    >>> torque([[0.8, 0.0, 0.0]], [[0.0, 10.0, 0.0]])
    array([0., 0., 8.])
    """
    arms = np.asarray(positions, dtype=float) - np.asarray(origin, dtype=float)
    return np.sum(np.cross(arms, forces), axis=-2)


def inertia_tensor(
    masses: ArrayLike,
    positions: ArrayLike,
    origin: ArrayLike = (0.0, 0.0, 0.0),
) -> NDArray:
    """I_ab = Σ m_i (|r_i|² δ_ab − r_ia r_ib), r_i measured from O.

    The matrix that gives the angular momentum of a rigid rotation,
    L = I ω (Section 5.5). Pass the centre of mass as ``origin`` for the
    tensor of a molecule.

    Parameters
    ----------
    masses : array_like, shape (N,)
    positions : array_like, shape (N, 3)
    origin : array_like, shape (3,)

    Returns
    -------
    ndarray, shape (3, 3)
        Symmetric.

    Examples
    --------
    Two 1 kg masses 1 m apart on the x axis, about their midpoint:

    >>> inertia_tensor([1.0, 1.0], [[-0.5, 0, 0], [0.5, 0, 0]]).diagonal()
    array([0. , 0.5, 0.5])
    """
    masses = np.asarray(masses, dtype=float)
    arms = np.asarray(positions, dtype=float) - np.asarray(origin, dtype=float)
    squared = np.sum(masses * np.sum(arms**2, axis=1))  # Σ m_i |r_i|²
    products = (masses[:, np.newaxis] * arms).T @ arms  # Σ m_i r_ia r_ib
    return squared * np.eye(3) - products


def angular_velocity(
    masses: ArrayLike, positions: ArrayLike, velocities: ArrayLike
) -> NDArray:
    """The ω of the rigid rotation about the centre of mass carrying L.

    Solves I ω = L, both about the centre of mass (Section 5.6). For a
    linear molecule I has no inverse, since a turn about the molecular
    axis moves no atom; least squares then returns the ω with no part
    along that axis.

    Examples
    --------
    >>> r = [[-0.5, 0, 0], [0.5, 0, 0]]
    >>> angular_velocity([1.0, 1.0], r, [[0, -1, 0], [0, 1, 0]]).round(6)
    array([0., 0., 2.])
    """
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    centre = masses @ positions / masses.sum()
    inertia = inertia_tensor(masses, positions, centre)
    ang_mom = angular_momentum(masses, positions, velocities, centre)
    return np.linalg.lstsq(inertia, ang_mom, rcond=None)[0]


def remove_rigid_motion(
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    rotation: bool = True,
) -> NDArray:
    """Velocities with the drift and, optionally, the rigid rotation removed.

    Subtracts V = P/M from every velocity, then ω × (r_i − R) with ω from
    ``angular_velocity``, so that the total momentum and the angular
    momentum about the centre of mass R are both zero (Section 5.6).

    Parameters
    ----------
    masses : array_like, shape (N,)
    positions, velocities : array_like, shape (N, 3)
    rotation : bool
        False removes the drift only, as for a periodic cell.

    Returns
    -------
    ndarray, shape (N, 3)
    """
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    total = masses.sum()
    drift = masses @ velocities / total  # V = P/M
    velocities = velocities - drift
    if rotation:
        centre = masses @ positions / total  # R
        omega = angular_velocity(masses, positions, velocities)
        velocities = velocities - np.cross(omega, positions - centre)
    return velocities


def spring_hessian(
    positions: ArrayLike, pairs: ArrayLike, stiffnesses: ArrayLike
) -> NDArray:
    """Hessian of springs between pairs of bodies, all at their rest length.

    A spring of stiffness k from body i to body j stores ½k(r_ij − d)²,
    with d its rest length. At r_ij = d its second derivatives fill the
    3 × 3 blocks k u uᵀ at (i, i) and (j, j) and −k u uᵀ at (i, j) and
    (j, i), with u the unit vector along the spring (Section 5.7).

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        The resting positions.
    pairs : array_like of int, shape (n_springs, 2)
        The bodies joined by each spring.
    stiffnesses : array_like, shape (n_springs,)

    Returns
    -------
    ndarray, shape (3N, 3N)
        Rows and columns ordered x₁, y₁, z₁, x₂, ...

    Examples
    --------
    >>> spring_hessian([[0, 0, 0], [1, 0, 0]], [[0, 1]], [2.0])[0]
    array([ 2.,  0.,  0., -2.,  0.,  0.])
    """
    positions = np.asarray(positions, dtype=float)
    hessian = np.zeros((positions.size, positions.size))
    for (i, j), k in zip(
        np.asarray(pairs), np.asarray(stiffnesses), strict=True
    ):
        u = positions[j] - positions[i]
        u = u / np.linalg.norm(u)
        block = k * np.outer(u, u)
        a, b = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
        hessian[a, a] += block
        hessian[b, b] += block
        hessian[a, b] -= block
        hessian[b, a] -= block
    return hessian
