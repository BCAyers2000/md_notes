"""Interatomic potentials: Chapter 8.

Pair potentials and what is built from them:

- ``lennard_jones``, ``morse``, ``buckingham``: the energy φ(r) of a pair
  at the distance r and its slope dφ/dr (Section 8.3);
- ``with_cutoff``: the same pair potential truncated, truncated and
  shifted, or switched smoothly to zero at a cutoff (Section 8.5);
- ``pair_energy_forces``: the energy, the forces and the virial of N atoms
  that interact in pairs (Section 8.4);
- ``finite_difference_forces``: forces from central differences of any
  energy, the check every force routine must pass (Section 8.4).

Many-body and bonded terms (Sections 8.6 and 8.7):

- ``second_moment``: a simple embedded-atom energy, the square root of a
  sum over neighbours, with its forces;
- ``bond_angle`` and ``dihedral_angle``: the angles of three and four
  atoms; ``opls_torsion``: the cosine series of a dihedral term;
- ``coulomb_energy`` and ``madelung_partial_sum``: the energy of point
  charges, and the slowly converging sum of an ionic crystal (Section 8.8).

Units
-----
Energies in eV, lengths in Å, forces in eV/Å, charges in units of the
elementary charge; or any consistent set, such as the reduced units of
Section 8.3 in which σ = ε = 1.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.units import COULOMB

PairFunction = Callable[[NDArray], tuple[NDArray, NDArray]]


def lennard_jones(
    r: ArrayLike, epsilon: float = 1.0, sigma: float = 1.0
) -> tuple[NDArray, NDArray]:
    """The Lennard-Jones energy 4ε[(σ/r)¹² − (σ/r)⁶] and its slope.

    Parameters
    ----------
    r : array_like
        Distances between pairs, in Å (or σ).
    epsilon : float
        Depth of the well, in eV.
    sigma : float
        Distance at which the energy is zero, in Å.

    Returns
    -------
    phi, dphi_dr : ndarray
        The energy in eV and its slope in eV/Å.

    Examples
    --------
    The minimum is at 2^(1/6) σ, with depth ε:

    >>> phi, slope = lennard_jones(2 ** (1 / 6))
    >>> print(round(float(phi), 12), round(float(slope), 12))
    -1.0 0.0
    """
    r = np.asarray(r, dtype=float)
    s6 = (sigma / r) ** 6
    phi = 4 * epsilon * (s6 * s6 - s6)
    dphi = 4 * epsilon * (-12 * s6 * s6 + 6 * s6) / r
    return phi, dphi


def morse(
    r: ArrayLike, depth: float, width: float, r0: float
) -> tuple[NDArray, NDArray]:
    """The Morse energy D[e^{−2a(r−r0)} − 2e^{−a(r−r0)}] and its slope.

    This is the well of Section 4.5 lowered by D, so that it is zero for
    atoms far apart and −D at its minimum r0.

    Parameters
    ----------
    r : array_like
        Distances, in Å.
    depth : float
        D, the energy needed to pull the pair apart, in eV.
    width : float
        a, an inverse length, in 1/Å.
    r0 : float
        The distance of the minimum, in Å.

    Returns
    -------
    phi, dphi_dr : ndarray
        In eV and eV/Å.

    Examples
    --------
    >>> phi, slope = morse(1.5, depth=2.0, width=1.8, r0=1.5)
    >>> print(float(phi), float(slope))
    -2.0 0.0
    """
    r = np.asarray(r, dtype=float)
    e = np.exp(-width * (r - r0))
    phi = depth * (e * e - 2 * e)
    dphi = depth * (-2 * width * e * e + 2 * width * e)
    return phi, dphi


def buckingham(
    r: ArrayLike, a: float, rho: float, c: float
) -> tuple[NDArray, NDArray]:
    """The Buckingham energy A e^{−r/ρ} − C/r⁶ and its slope.

    Parameters
    ----------
    r : array_like
        Distances, in Å.
    a : float
        A, the strength of the repulsion, in eV.
    rho : float
        ρ, its range, in Å.
    c : float
        C, the strength of the attraction, in eV Å⁶.

    Returns
    -------
    phi, dphi_dr : ndarray
        In eV and eV/Å.
    """
    r = np.asarray(r, dtype=float)
    rep = a * np.exp(-r / rho)
    phi = rep - c / r**6
    dphi = -rep / rho + 6 * c / r**7
    return phi, dphi


def _smoothstep(x: NDArray) -> tuple[NDArray, NDArray]:
    """S(x) = 1 − 10x³ + 15x⁴ − 6x⁵ on 0 ≤ x ≤ 1, and its slope."""
    x = np.clip(x, 0.0, 1.0)
    s = 1 - x**3 * (10 - 15 * x + 6 * x * x)
    ds = -30 * x * x * (1 - x) ** 2
    return s, ds


def with_cutoff(
    pair: PairFunction,
    r_cut: float,
    scheme: str = "shift",
    r_switch: float | None = None,
) -> PairFunction:
    """A pair potential that is zero beyond the distance ``r_cut``.

    Parameters
    ----------
    pair : callable
        ``pair(r)`` returns (φ, dφ/dr).
    r_cut : float
        The cutoff distance r_c, in the length unit of ``pair``.
    scheme : {"truncate", "shift", "switch"}
        "truncate" sets φ to zero beyond r_c, so φ jumps by φ(r_c) there;
        "shift" subtracts φ(r_c) inside, so φ is continuous and its slope
        jumps; "switch" multiplies φ by S(x), x = (r − r_s)/(r_c − r_s),
        which falls from 1 at r_s to 0 at r_c with its first two
        derivatives zero at both ends, so φ and its slope are continuous.
    r_switch : float, optional
        r_s for "switch"; by default 0.9 r_c.

    Returns
    -------
    callable
        The new pair function.

    Examples
    --------
    >>> lj = with_cutoff(lennard_jones, 2.5, "shift")
    >>> print(round(float(lj(2.5)[0]), 12), float(lj(3.0)[0]))
    0.0 0.0
    """
    phi_c = float(pair(np.array(r_cut))[0])
    r_s = 0.9 * r_cut if r_switch is None else r_switch

    def cut(r: NDArray) -> tuple[NDArray, NDArray]:
        r = np.asarray(r, dtype=float)
        phi, dphi = pair(r)
        inside = r < r_cut
        if scheme == "truncate":
            pass
        elif scheme == "shift":
            phi = phi - phi_c
        elif scheme == "switch":
            s, ds = _smoothstep((r - r_s) / (r_cut - r_s))
            phi, dphi = phi * s, dphi * s + phi * ds / (r_cut - r_s)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        return np.where(inside, phi, 0.0), np.where(inside, dphi, 0.0)

    return cut


def pair_energy_forces(
    positions: ArrayLike, pair: PairFunction
) -> tuple[float, NDArray, float]:
    """Energy, forces and virial of atoms that interact in pairs.

    U = Σ_{i<j} φ(r_ij), and F_i = −Σ_j φ'(r_ij) (r_i − r_j)/r_ij, so that
    each pair pushes its two atoms equally and oppositely (Section 8.4).
    The virial is W = Σ_i r_i·F_i = −Σ_{i<j} r_ij φ'(r_ij).

    Parameters
    ----------
    positions : array_like, shape (N, d)
        Positions, in Å.
    pair : callable
        ``pair(r)`` returns (φ, dφ/dr) for an array of distances.

    Returns
    -------
    energy : float
        U, in eV.
    forces : ndarray, shape (N, d)
        In eV/Å.
    virial : float
        W, in eV.

    Examples
    --------
    Two atoms at the Lennard-Jones minimum feel no force:

    >>> u, f, w = pair_energy_forces([[0, 0, 0], [2 ** (1 / 6), 0, 0]],
    ...                              lennard_jones)
    >>> print(round(u, 12), np.abs(f).max().round(12), abs(round(w, 12)))
    -1.0 0.0 0.0
    """
    r = np.asarray(positions, dtype=float)
    n = len(r)
    i, j = np.triu_indices(n, k=1)  # every pair once
    rij = r[i] - r[j]
    dist = np.linalg.norm(rij, axis=1)
    phi, dphi = pair(dist)
    f_on_i = -(dphi / dist)[:, None] * rij  # force on i from j
    forces = np.zeros_like(r)
    np.add.at(forces, i, f_on_i)
    np.add.at(forces, j, -f_on_i)
    return float(phi.sum()), forces, float(-(dist * dphi).sum())


def finite_difference_forces(
    energy: Callable[[NDArray], float], positions: ArrayLike, step: float
) -> NDArray:
    """Forces −∂U/∂x by central differences, one coordinate at a time.

    Each component is −[U(x + h) − U(x − h)]/(2h), with an error of
    order h² from the Taylor series and of order (round-off)/h from
    subtracting two nearly equal energies (Section 8.4).

    Parameters
    ----------
    energy : callable
        ``energy(positions)`` returns U in eV.
    positions : array_like, shape (N, d)
        In Å.
    step : float
        h, in Å.

    Returns
    -------
    ndarray, shape (N, d)
        In eV/Å.
    """
    r = np.asarray(positions, dtype=float)
    forces = np.zeros_like(r)
    for index in np.ndindex(r.shape):
        up, down = r.copy(), r.copy()
        up[index] += step
        down[index] -= step
        forces[index] = -(energy(up) - energy(down)) / (2 * step)
    return forces


def second_moment(
    positions: ArrayLike,
    repulsion: float,
    hopping: float,
    p: float,
    q: float,
    r0: float,
) -> tuple[float, NDArray]:
    """A simple embedded-atom energy and its forces.

    U = Σ_i [Σ_{j≠i} A e^{−p(r_ij/r0 − 1)} − √ρ_i], with the density
    ρ_i = Σ_{j≠i} ξ² e^{−2q(r_ij/r0 − 1)}: pair repulsion plus a bond
    energy that grows only as the square root of the number of neighbours
    (Section 8.6).

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        In Å.
    repulsion : float
        A, in eV.
    hopping : float
        ξ, in eV.
    p, q : float
        Dimensionless decay rates of the repulsion and of the density.
    r0 : float
        Nearest-neighbour distance of the crystal, in Å.

    Returns
    -------
    energy : float
        In eV.
    forces : ndarray, shape (N, 3)
        In eV/Å.
    """
    r = np.asarray(positions, dtype=float)
    n = len(r)
    d = r[:, None, :] - r[None, :, :]  # r_i − r_j
    dist = np.linalg.norm(d, axis=2)
    off = ~np.eye(n, dtype=bool)
    x = np.where(off, dist / r0 - 1, 0.0)
    rep = np.where(off, repulsion * np.exp(-p * x), 0.0)
    dens = np.where(off, hopping**2 * np.exp(-2 * q * x), 0.0)
    rho = dens.sum(axis=1)
    energy = rep.sum() - np.sqrt(rho).sum()
    # dU/dr_ij for the pair (i, j), counting both orders of the pair
    drep = -p / r0 * rep
    ddens = -2 * q / r0 * dens
    half = 0.5 / np.sqrt(rho)
    dU_ddist = 2 * drep - (half[:, None] + half[None, :]) * ddens
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(off[:, :, None], d / dist[:, :, None], 0.0)
    forces = -(dU_ddist[:, :, None] * unit).sum(axis=1)
    return float(energy), forces


def bond_angle(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> float:
    """The angle at b between the bonds to a and to c, in rad."""
    u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    cos = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def dihedral_angle(
    a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike
) -> float:
    """The dihedral angle of the chain a-b-c-d, in rad, between −π and π.

    It is the angle between the plane of a, b, c and the plane of b, c, d,
    seen along the bond b-c: with n1 = (b − a) × (c − b) and
    n2 = (c − b) × (d − c), cos ψ = n1·n2/(|n1||n2|), and the sign is that
    of (n1 × n2)·(c − b) (Section 8.7).

    Examples
    --------
    A planar zigzag, as in the extended chain of butane, is 180°:

    >>> psi = dihedral_angle([0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0])
    >>> print(round(np.degrees(psi), 6))
    180.0
    """
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    x = n1 @ n2
    y = np.cross(n1, n2) @ b2 / np.linalg.norm(b2)
    return float(np.arctan2(y, x))


def opls_torsion(phi: ArrayLike, v: ArrayLike) -> NDArray:
    """The OPLS torsion energy Σ_n (V_n/2)[1 + (−1)^{n+1} cos nψ].

    Parameters
    ----------
    phi : array_like
        Dihedral angles, in rad.
    v : array_like
        V_1, V_2, ... in eV (or any energy unit).

    Returns
    -------
    ndarray
        The energy, in the unit of ``v``.
    """
    phi = np.asarray(phi, dtype=float)
    total = np.zeros_like(phi)
    for n, vn in enumerate(np.asarray(v, dtype=float), start=1):
        total += 0.5 * vn * (1 + (-1) ** (n + 1) * np.cos(n * phi))
    return total


def coulomb_energy(charges: ArrayLike, positions: ArrayLike) -> float:
    """Σ_{i<j} k q_i q_j / r_ij for point charges, in eV.

    Parameters
    ----------
    charges : array_like, shape (N,)
        In units of the elementary charge.
    positions : array_like, shape (N, 3)
        In Å.

    Examples
    --------
    Two opposite unit charges 1 Å apart: −14.4 eV.

    >>> print(round(coulomb_energy([1, -1], [[0, 0, 0], [1, 0, 0]]), 4))
    -14.3996
    """
    q = np.asarray(charges, dtype=float)
    r = np.asarray(positions, dtype=float)
    i, j = np.triu_indices(len(q), k=1)
    dist = np.linalg.norm(r[i] - r[j], axis=1)
    return float(COULOMB * np.sum(q[i] * q[j] / dist))


def madelung_partial_sum(extent: float, shape: str = "cube") -> float:
    """The sum −Σ (−1)^{a+b+c}/√(a²+b²+c²) over a rock-salt lattice.

    The ion at the origin feels the others at the integer points (a, b,
    c), charged (−1)^{a+b+c} relative to it. The sum is taken over the
    points inside a cube |a|, |b|, |c| ≤ extent, with the ions on its
    faces, edges and corners counted with the fractions 1/2, 1/4 and 1/8
    that lie inside it (Evjen), or over the points inside a sphere of
    radius ``extent``. For the cube it approaches the Madelung constant of
    rock salt, 1.7476; for the sphere it does not settle.

    Parameters
    ----------
    extent : float
        Half the side of the cube, or the radius of the sphere, in units
        of the nearest-neighbour distance.
    shape : {"cube", "sphere"}

    Returns
    -------
    float
    """
    m = int(np.floor(extent))
    a = np.arange(-m, m + 1)
    x, y, z = np.meshgrid(a, a, a, indexing="ij")
    dist = np.sqrt(x * x + y * y + z * z, dtype=float)
    sign = np.where((x + y + z) % 2 == 0, 1.0, -1.0)
    with np.errstate(divide="ignore"):
        term = np.where(dist > 0, -sign / dist, 0.0)
    if shape == "cube":
        weight = np.ones_like(dist)
        for coord in (x, y, z):
            weight = weight * np.where(np.abs(coord) == m, 0.5, 1.0)
        return float(np.sum(weight * term))
    if shape == "sphere":
        return float(np.sum(np.where(dist <= extent, term, 0.0)))
    raise ValueError(f"unknown shape {shape!r}")
