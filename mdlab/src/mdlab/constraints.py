"""Bonds held at fixed lengths: Chapter 11.

A bond between atoms i and j held at the length d is the constraint
χ = ½(|rⱼ − rᵢ|² − d²) = 0 of Section 6.6. Its force acts along the bond,
equal and opposite on the two atoms, with a size (the Lagrange multiplier)
chosen at every step so that the constraint holds:

- ``bond_errors``: how far each constrained distance is from its length;
- ``shake``: new positions moved along the bonds of the old ones until
  every constrained distance has its length (SHAKE, Section 11.1);
- ``rattle_velocities``: the velocities corrected so that no constrained
  distance is changing (RATTLE, Section 11.2);
- ``rattle_step``: one step of velocity Verlet with both corrections;
- ``run``: the loop of ``md.run`` with the bonds held, recording the
  largest error of the constraints;
- ``repartition_masses``: mass moved from heavy atoms to the hydrogen
  atoms bonded to them, the total kept (Section 11.6).

Both corrections take the bonds in groups that share no atom, so that a
whole group is corrected at once. For water, whose bonds are listed
molecule by molecule, this applies the same corrections in the same order
as taking the bonds one by one; for other lists the order within a sweep
can differ (a chain A–B, B–C, C–D is corrected A–B, C–D, B–C), which
changes the path to the converged positions, not the conditions they
meet.

Units
-----
Positions in Å, velocities in Å/fs, masses in amu, energies in eV, times
in fs; or reduced units with ``force_to_accel = 1``.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.cell import minimum_image
from mdlab.units import FORCE_TO_ACCEL, MV2_TO_EV

EnergyForces = Callable[[NDArray], tuple[float, NDArray]]


def _groups(bonds: NDArray) -> list[NDArray]:
    """The bonds in groups that share no atom.

    Each bond joins the first group none of whose bonds touches its atoms.
    """
    groups: list[list[int]] = []
    atoms: list[set[int]] = []
    for k, (i, j) in enumerate(bonds.tolist()):
        for group, used in zip(groups, atoms, strict=True):
            if i not in used and j not in used:
                group.append(k)
                used.update((i, j))
                break
        else:
            groups.append([k])
            atoms.append({i, j})
    return [np.array(group) for group in groups]


def _bond_vectors(
    positions: NDArray, bonds: NDArray, cell: ArrayLike | None
) -> NDArray:
    d = positions[bonds[:, 1]] - positions[bonds[:, 0]]
    return d if cell is None else minimum_image(d, cell)


def bond_errors(
    positions: ArrayLike,
    bonds: ArrayLike,
    lengths: ArrayLike,
    cell: ArrayLike | None = None,
) -> NDArray:
    """|rⱼ − rᵢ| − d for each bond (i, j), in Å.

    >>> r = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]
    >>> bond_errors(r, [[0, 1]], [1.0]).round(12)
    array([0.1])
    """
    r = np.asarray(positions, dtype=float)
    pairs = np.asarray(bonds, dtype=int).reshape(-1, 2)
    d = _bond_vectors(r, pairs, cell)
    return np.linalg.norm(d, axis=1) - np.asarray(lengths, dtype=float)


def shake(
    positions: ArrayLike,
    reference: ArrayLike,
    masses: ArrayLike,
    bonds: ArrayLike,
    lengths: ArrayLike,
    cell: ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_sweeps: int = 500,
    strict: bool = True,
) -> tuple[NDArray, int]:
    """Correct ``positions`` along the old bonds until each has its length.

    Each bond (i, j) is to have its length d. Each correction moves atom j
    by (m_r η/mⱼ) r_old and atom i by −(m_r η/mᵢ) r_old, where r_old is
    the bond vector rⱼ − rᵢ at ``reference``, m_r = 1/(1/mᵢ + 1/mⱼ) the
    reduced mass, and η = (d² − |r_now|²)/(2 r_old·r_now) for the current
    bond vector r_now: one step of Newton's method for |r_now + η r_old|²
    = d² (Section 11.1). A sweep corrects every bond in turn; the sweeps
    stop when every |η|, about the relative error of the bond's length, is
    below ``tolerance``.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        The positions to correct, in Å (those after an unconstrained step).
    reference : array_like, shape (N, 3)
        The positions at the start of the step, where the bonds held.
    masses : array_like, shape (N,)
        In amu.
    bonds : array_like, shape (M, 2)
        The pairs (i, j) of atoms held at fixed distances.
    lengths : array_like, shape (M,)
        Their lengths d, in Å.
    cell : array_like, shape (3, 3), optional
        Lattice vectors as columns; if given, each bond is taken between
        the nearest copies of its atoms, at the start and in ``positions``
        separately, so either may have been wrapped into the cell.
    tolerance : float
        The largest |η| accepted.
    max_sweeps : int
        Sweeps allowed before giving up.
    strict : bool
        If True, fail when ``max_sweeps`` sweeps do not converge; if
        False, return the positions they reach, so that the sweeps can be
        followed one at a time.

    Returns
    -------
    positions : ndarray, shape (N, 3)
    sweeps : int
        The sweeps made, the last of which moved nothing.

    Examples
    --------
    Two atoms of masses 1 and 3 that drifted to 1.2 apart, pulled back to
    1 along the line they started on; the heavier moves a third as far:

    >>> r, sweeps = shake([[0.0, 0, 0], [1.2, 0, 0]],
    ...                   [[0.0, 0, 0], [1.0, 0, 0]], [1.0, 3.0],
    ...                   [[0, 1]], [1.0])
    >>> print(r[:, 0].round(9), sweeps)
    [0.15 1.15] 5
    """
    r = np.array(positions, dtype=float)
    start = np.asarray(reference, dtype=float)
    m = np.asarray(masses, dtype=float)
    pairs = np.asarray(bonds, dtype=int).reshape(-1, 2)
    d2 = np.asarray(lengths, dtype=float) ** 2
    old = _bond_vectors(start, pairs, cell)  # the bonds at the start
    # the lattice shift that makes each provisional bond its nearest copy
    copy = _bond_vectors(r, pairs, cell) - (r[pairs[:, 1]] - r[pairs[:, 0]])
    m_r = 1 / (1 / m[pairs[:, 0]] + 1 / m[pairs[:, 1]])
    groups = _groups(pairs)
    for sweep in range(1, max_sweeps + 1):
        moved = False
        for g in groups:
            i, j = pairs[g, 0], pairs[g, 1]
            now = r[j] - r[i] + copy[g]
            eta = 0.5 * (d2[g] - np.sum(now * now, 1)) / np.sum(
                old[g] * now, 1
            )
            eta[np.abs(eta) <= tolerance] = 0.0
            moved = moved or bool(np.any(eta))
            push = (m_r[g] * eta)[:, None] * old[g]
            r[j] += push / m[j, None]
            r[i] -= push / m[i, None]
        if not moved:
            return r, sweep
    if strict:
        raise RuntimeError(f"SHAKE did not converge in {max_sweeps} sweeps")
    return r, max_sweeps


def rattle_velocities(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: ArrayLike,
    bonds: ArrayLike,
    lengths: ArrayLike,
    cell: ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_sweeps: int = 500,
) -> tuple[NDArray, int]:
    """Correct ``velocities`` so that no bond is lengthening or shortening.

    Each correction changes vⱼ by (m_r ζ/mⱼ) r and vᵢ by −(m_r ζ/mᵢ) r,
    with r the bond vector rⱼ − rᵢ at ``positions`` and
    ζ = −(vⱼ − vᵢ)·r/d², which makes (vⱼ − vᵢ)·r zero for that bond
    (Section 11.2). The sweeps stop when every |ζ|, the rate at which a
    bond's length is changing divided by the length, in 1/fs, is below
    ``tolerance``.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Positions at which the bonds hold, in Å.
    velocities : array_like, shape (N, 3)
        In Å/fs.
    masses, bonds, lengths, cell, max_sweeps
        As for ``shake``.
    tolerance : float
        The largest |ζ| accepted, in 1/fs.

    Returns
    -------
    velocities : ndarray, shape (N, 3)
    sweeps : int

    Examples
    --------
    The part of the relative velocity along the bond is removed, and the
    total momentum is unchanged:

    >>> v, _ = rattle_velocities([[0.0, 0, 0], [1.0, 0, 0]],
    ...                          [[0.0, 0, 0], [0.3, 0.4, 0]], [1.0, 3.0],
    ...                          [[0, 1]], [1.0])
    >>> print(v.round(9).tolist())
    [[0.225, 0.0, 0.0], [0.225, 0.4, 0.0]]
    """
    r = np.asarray(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    pairs = np.asarray(bonds, dtype=int).reshape(-1, 2)
    d2 = np.asarray(lengths, dtype=float) ** 2
    bond = _bond_vectors(r, pairs, cell)
    m_r = 1 / (1 / m[pairs[:, 0]] + 1 / m[pairs[:, 1]])
    groups = _groups(pairs)
    for sweep in range(1, max_sweeps + 1):
        moved = False
        for g in groups:
            i, j = pairs[g, 0], pairs[g, 1]
            zeta = -np.sum((v[j] - v[i]) * bond[g], 1) / d2[g]
            zeta[np.abs(zeta) <= tolerance] = 0.0
            moved = moved or bool(np.any(zeta))
            push = (m_r[g] * zeta)[:, None] * bond[g]
            v[j] += push / m[j, None]
            v[i] -= push / m[i, None]
        if not moved:
            return v, sweep
    raise RuntimeError(f"RATTLE did not converge in {max_sweeps} sweeps")


def rattle_step(
    positions: ArrayLike,
    velocities: ArrayLike,
    forces: ArrayLike,
    masses: ArrayLike,
    model: EnergyForces,
    dt: float,
    bonds: ArrayLike,
    lengths: ArrayLike,
    cell: ArrayLike | None = None,
    tolerance: float = 1e-10,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> tuple[NDArray, NDArray, float, NDArray]:
    """One step of velocity Verlet with the bonds held (RATTLE).

    Half kick, drift, positions corrected by ``shake`` and the half-step
    velocities by the same displacement over δt; then the forces at the
    new positions, the second half kick, and ``rattle_velocities``.

    Parameters
    ----------
    positions, velocities : array_like, shape (N, 3)
        A state at which the bonds hold, in Å and Å/fs.
    forces : array_like, shape (N, 3)
        The forces of ``model`` at ``positions``, in eV/Å.
    masses : array_like, shape (N,)
        In amu.
    model : callable
        ``model(positions)`` returns (U in eV, forces in eV/Å).
    dt : float
        The step, in fs.
    bonds, lengths, cell, tolerance
        As for ``shake``; the same tolerance serves both corrections.
    force_to_accel : float
        Converts eV/Å per amu to Å/fs².

    Returns
    -------
    positions, velocities : ndarray
    potential : float
    forces : ndarray
    """
    m = np.asarray(masses, dtype=float)[:, None]
    r0 = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    keep = dict(cell=cell, tolerance=tolerance)
    v_half = v + 0.5 * dt * force_to_accel * np.asarray(forces) / m
    r, _ = shake(r0 + dt * v_half, r0, m[:, 0], bonds, lengths, **keep)
    v_half = (r - r0) / dt
    u, f = model(r)
    v = v_half + 0.5 * dt * force_to_accel * f / m
    v, _ = rattle_velocities(r, v, m[:, 0], bonds, lengths, **keep)
    return r, v, u, f


def run(
    model: EnergyForces,
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    cell: ArrayLike | None,
    dt: float,
    n_steps: int,
    bonds: ArrayLike,
    lengths: ArrayLike,
    every: int = 1,
    tolerance: float = 1e-10,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> dict[str, NDArray]:
    """``rattle_step`` repeated, recording every ``every`` steps.

    The starting velocities are first corrected by ``rattle_velocities``.
    Positions are integrated without wrapping, as in ``md.run``. Returns
    the recorded "times", "positions", "velocities", "potential" and
    "kinetic" energies, and "bond_error", the largest |rᵢⱼ − d| in Å.
    """
    m = np.asarray(masses, dtype=float)
    r = np.array(positions, dtype=float)
    keep = dict(cell=cell, tolerance=tolerance)
    v, _ = rattle_velocities(r, velocities, m, bonds, lengths, **keep)
    to_energy = 1.0 if force_to_accel == 1 else MV2_TO_EV
    keys = ("times", "positions", "velocities", "potential", "kinetic",
            "bond_error")
    record: dict[str, list] = {key: [] for key in keys}
    u, f = model(r)
    for step in range(n_steps + 1):
        if step > 0:
            r, v, u, f = rattle_step(
                r, v, f, m, model, dt, bonds, lengths,
                force_to_accel=force_to_accel, **keep,
            )
        if step % every == 0:
            kinetic = 0.5 * to_energy * float(m @ (v * v).sum(1))
            error = np.abs(bond_errors(r, bonds, lengths, cell)).max()
            for key, value in zip(
                keys, (step * dt, r.copy(), v.copy(), u, kinetic, error),
                strict=True,
            ):
                record[key].append(value)
    return {key: np.array(value) for key, value in record.items()}


def repartition_masses(
    masses: ArrayLike, hydrogen_bonds: ArrayLike, hydrogen_mass: float
) -> NDArray:
    """Masses with each hydrogen made heavier at its partner's expense.

    For each pair (heavy, hydrogen), the hydrogen's mass is raised to
    ``hydrogen_mass`` and the same amount taken from the heavy atom, so
    the total mass, and with it the motion of the centre of mass, is
    unchanged.

    >>> repartition_masses([15.999, 1.008, 1.008], [[0, 1], [0, 2]], 3.024)
    array([11.967,  3.024,  3.024])
    """
    new = np.array(masses, dtype=float)
    for heavy, hydrogen in np.asarray(hydrogen_bonds, dtype=int):
        extra = hydrogen_mass - new[hydrogen]
        new[hydrogen] += extra
        new[heavy] -= extra
    if np.any(new <= 0):
        raise ValueError("a heavy atom would be left with no mass")
    return new
