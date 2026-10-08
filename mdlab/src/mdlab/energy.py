"""Work and energy: Chapter 3.

The pieces of Chapter 3 that the book computes:

- ``work``: the work done by a force along a path, a line integral
  (Section 3.9);
- ``turning_points``: where a motion of given energy turns back
  (Section 3.6);
- ``hill_track``: the hill-shaped track of Sections 3.5-3.6;
- ``swirl``: a force with no potential energy (Sections 3.9-3.11);
- ``kinetic_energy``: the kinetic energy of atoms, in eV (Section 3.12);
- ``hexagonal_surface`` and ``surface_sites``: a model of a lithium atom
  on a hexagonal layer of carbon (Section 3.13).

Units
-----
``kinetic_energy`` assumes the units of atoms: mass in amu, velocity in
Å/fs, energy in eV. ``hexagonal_surface`` and ``swirl`` take positions in
Å and return eV and eV/Å. ``hill_track`` is in metres. ``work`` and
``turning_points`` work in any consistent units.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import units

#: Coefficients of the track height h(x) = Σ c_n x^n, in metres, lowest
#: power first: h = 0.15 x⁴ − 0.6 x² + 0.15 x + 1.0.
TRACK = (1.0, 0.15, -0.6, 0.0, 0.15)

#: Barrier between neighbouring hollows of the model surface, in eV.
#: Illustrative, not fitted to any material.
SURFACE_BARRIER = 0.3
#: Distance between neighbouring hollows of the model surface, in Å.
SURFACE_SPACING = 2.46


def work(force: Callable[[NDArray], NDArray], path: ArrayLike) -> float:
    """Work done by a force along a path (Section 3.9).

    The path is a chain of short straight chords. On each chord the force
    is taken at the midpoint, so the error falls as the square of the
    chord length.

    Parameters
    ----------
    force : callable
        ``force(r)`` maps positions of shape (n, d) to forces of the same
        shape.
    path : array_like, shape (n_points, d)
        Successive points of the path.

    Returns
    -------
    float
        The work, in force times length units: eV for eV/Å and Å.

    Examples
    --------
    A constant force of 0.5 eV/Å along x, over 2 Å along x:

    >>> def push(r):
    ...     return np.broadcast_to([0.5, 0.0], r.shape)
    >>> work(push, [[0.0, 0.0], [2.0, 0.0]])
    1.0
    """
    path = np.asarray(path, dtype=float)
    chords = np.diff(path, axis=0)  # the step from each point to the next
    midpoints = 0.5 * (path[1:] + path[:-1])
    # F · Δr on every chord, added over the components and the chords
    return float(np.sum(force(midpoints) * chords))


def turning_points(
    x: ArrayLike, potential: ArrayLike, energy: float
) -> NDArray:
    """Positions where the potential energy equals the total energy.

    A motion of total energy E is allowed only where U(x) ≤ E, and it
    turns back where U(x) = E (Section 3.6). The crossings are found
    between successive grid points and placed by straight-line
    interpolation.

    Parameters
    ----------
    x : array_like, shape (n,)
        Increasing positions.
    potential : array_like, shape (n,)
        U at those positions.
    energy : float
        The total energy E, in the units of ``potential``.

    Returns
    -------
    ndarray
        The crossings, in increasing order.

    Examples
    --------
    A spring of stiffness 2 with E = 1 turns at x = ±1:

    >>> x = np.linspace(-2.0, 2.0, 4001)
    >>> turning_points(x, x**2, 1.0).round(6)
    array([-1.,  1.])
    """
    x = np.asarray(x, dtype=float)
    above = np.asarray(potential, dtype=float) - energy
    k = np.flatnonzero(np.sign(above[:-1]) * np.sign(above[1:]) < 0)
    # the straight line through the two neighbouring points crosses zero
    # a fraction above[k] / (above[k] - above[k + 1]) of the way along
    fraction = above[k] / (above[k] - above[k + 1])
    exact = x[np.flatnonzero(above == 0.0)]
    return np.sort(
        np.concatenate([x[k] + fraction * (x[k + 1] - x[k]), exact])
    )


def hill_track(x: ArrayLike) -> tuple[NDArray, NDArray]:
    """Height and slope of the hill-shaped track (Sections 3.5-3.6).

    h(x) = 0.15 x⁴ − 0.6 x² + 0.15 x + 1.0, with x and h in metres: a
    deep valley near x = −1.47 m, a hill near 0.13 m and a shallower
    valley near 1.35 m.

    Parameters
    ----------
    x : array_like
        Horizontal positions, in m.

    Returns
    -------
    height : ndarray
        h(x), in m.
    slope : ndarray
        dh/dx, dimensionless.
    """
    x = np.asarray(x, dtype=float)
    height = np.polynomial.polynomial.polyval(x, TRACK)
    slope = np.polynomial.polynomial.polyval(
        x, np.polynomial.polynomial.polyder(TRACK)
    )
    return height, slope


def swirl(
    r: ArrayLike, strength: float = 0.02, centre: ArrayLike = (0.0, 0.0)
) -> NDArray:
    """A force in the plane that circles a centre and has no potential.

    F(r) = c (−(y − y0), x − x0). Its curl is 2c everywhere, so the work
    once anticlockwise around any closed loop is 2c times the area the
    loop encloses (Section 3.11).

    Parameters
    ----------
    r : array_like, shape (..., 2)
        Positions, in Å.
    strength : float
        c, in eV/Å².
    centre : array_like, shape (2,)
        (x0, y0), in Å.

    Returns
    -------
    ndarray, shape (..., 2)
        Forces, in eV/Å.
    """
    offset = np.asarray(r, dtype=float) - np.asarray(centre)
    x, y = offset[..., 0], offset[..., 1]
    return strength * np.stack([-y, x], axis=-1)


def kinetic_energy(masses: ArrayLike, velocities: ArrayLike) -> float:
    """Total kinetic energy of atoms, K = Σ m v² / 2, in eV (Section 3.12).

    Parameters
    ----------
    masses : array_like, shape (N,) or scalar
        Masses, in amu.
    velocities : array_like, shape (N, d) or (d,)
        Velocities, in Å/fs.

    Returns
    -------
    float
        The kinetic energy, in eV.

    Raises
    ------
    ValueError
        If there is not one mass for each atom.

    Examples
    --------
    A lithium atom (6.94 amu) moving at 0.0139 Å/fs:

    >>> print(f"{kinetic_energy(6.94, [0.0139]):.4f} eV")
    0.0695 eV
    """
    masses = np.asarray(masses, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    speed_squared = np.sum(velocities**2, axis=-1)  # v · v, one per atom
    if masses.ndim > 0 and masses.shape != speed_squared.shape:
        raise ValueError("give one mass for each row of velocities")
    in_amu_a2_per_fs2 = 0.5 * np.sum(masses * speed_squared)
    return float(in_amu_a2_per_fs2) * units.MV2_TO_EV


def hexagonal_surface(
    r: ArrayLike,
    barrier: float = SURFACE_BARRIER,
    spacing: float = SURFACE_SPACING,
) -> tuple[NDArray, NDArray]:
    """Potential energy of an atom on a model hexagonal surface.

    Three cosine ripples whose wave vectors g_k, of length
    4π / (√3 a), point at −30°, 90° and 210° add up to a pattern of
    hollows on a triangular lattice of spacing a:

        U(r) = (U_b/4) (3 − Σ_k cos(g_k · r)),
        F(r) = −∇U = −(U_b/4) Σ_k sin(g_k · r) g_k.

    U is 0 in the hollows, U_b at the bridges midway between neighbouring
    hollows and 9 U_b/8 at the tops (Section 3.13). Illustrative, not
    fitted to any material.

    Parameters
    ----------
    r : array_like, shape (..., 2)
        Positions in the plane of the surface, in Å.
    barrier : float
        U_b, the barrier: the rise from a hollow to a bridge, in eV.
    spacing : float
        a, the distance between neighbouring hollows, in Å.

    Returns
    -------
    energy : ndarray, shape (...)
        U, in eV.
    forces : ndarray, shape (..., 2)
        F = −∇U, in eV/Å.

    Examples
    --------
    >>> sites = surface_sites()
    >>> energy, _ = hexagonal_surface([sites["bridge"], sites["top"]])
    >>> energy.round(4)
    array([0.3   , 0.3375])
    """
    g = 4.0 * np.pi / (np.sqrt(3.0) * spacing)  # length of each g_k
    angles = np.radians([-30.0, 90.0, 210.0])
    ripples = g * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    phase = np.asarray(r, dtype=float) @ ripples.T  # g_k · r for each k
    energy = 0.25 * barrier * (3.0 - np.cos(phase).sum(axis=-1))
    forces = -0.25 * barrier * np.sin(phase) @ ripples
    return energy, forces


def surface_sites(spacing: float = SURFACE_SPACING) -> dict[str, NDArray]:
    """One hollow, bridge and top of ``hexagonal_surface``, in Å.

    Parameters
    ----------
    spacing : float
        a, the distance between neighbouring hollows, in Å.

    Returns
    -------
    dict
        ``"hollow"`` at the origin, ``"bridge"`` midway to the next
        hollow along x, and ``"top"`` a distance a/√3 from the hollow,
        30° above the x axis.
    """
    a = spacing
    return {
        "hollow": np.array([0.0, 0.0]),
        "bridge": np.array([a / 2.0, 0.0]),
        "top": np.array([a / 2.0, a / (2.0 * np.sqrt(3.0))]),
    }
