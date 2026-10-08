"""Checks of a trajectory's health: Chapter 11 onwards.

Each function reads what a run records and returns a number to set
beside its plots. Chapter 11 introduces them with the faults they catch
(Section 11.8); later chapters add checks of temperature, pressure and
thermostats, and Chapter 17 collects them all.

- ``energy_fluctuation``: the spread of the total energy against that of
  the kinetic energy, the yardstick of Section 9.6;
- ``energy_drift``: the change of the mean total energy from the first
  half of a run to the second, per unit time;
- ``closest_approach``: the shortest distance between two atoms, leaving
  out bonded pairs;
- ``largest_force``: the largest force on any atom;
- ``centre_of_mass_path``: how far the centre of mass has moved;
- ``largest_jump``: the largest move of an atom between two frames, as a
  fraction of the cell's smallest width;
- ``rms_displacement``: the root-mean-square distance the atoms have
  moved from their starting positions;
- ``unwrap`` and ``fractional_spread``: wrapped frames made continuous,
  and how far a frame's atoms spread across the cell;
- ``excess_kurtosis``: how far the shape of the velocities is from the
  Maxwell-Boltzmann distribution (Chapter 12);
- ``health_report``: every check that a run's record allows, each with
  its limit and whether it passed (Chapter 17).

Units
-----
As recorded: energies in eV, times in fs, lengths in Å, masses in amu.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.cell import minimum_image, perpendicular_widths, to_fractional
from mdlab.neighbours import all_pairs


def energy_fluctuation(potential: ArrayLike, kinetic: ArrayLike) -> float:
    """The standard deviation of K + U divided by that of K.

    Small in a healthy run without a thermostat; it should fall as δt²
    when the step is shortened (Section 9.6). A fluctuation that does not
    fall with the step points to a fault in the energies themselves.

    >>> k = np.array([1.0, 2.0, 1.0, 2.0])
    >>> print(energy_fluctuation(-k, k),
    ...       round(energy_fluctuation(-0.9 * k, k), 12))
    0.0 0.1
    """
    u = np.asarray(potential, dtype=float)
    k = np.asarray(kinetic, dtype=float)
    return float(np.std(u + k) / np.std(k))


def energy_drift(
    times: ArrayLike, potential: ArrayLike, kinetic: ArrayLike
) -> float:
    """The rate at which the mean total energy changes, in eV/fs.

    The mean of K + U over the second half of the run less that over the
    first, divided by the time between the centres of the two halves.

    >>> t = np.arange(5.0)
    >>> print(round(energy_drift(t, 0.1 * t, np.zeros(5)), 12))
    0.1
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(potential, dtype=float) + np.asarray(kinetic, dtype=float)
    if len(t) < 2:
        raise ValueError("a drift needs at least two frames")
    half = len(t) // 2
    first, second = slice(0, half), slice(len(t) - half, len(t))
    gap = np.mean(t[second]) - np.mean(t[first])
    return float((np.mean(e[second]) - np.mean(e[first])) / gap)


def closest_approach(
    positions: ArrayLike,
    cell: ArrayLike,
    skip: ArrayLike | None = None,
    reach: float = 3.0,
) -> float:
    """The shortest distance between two atoms, in Å.

    Pairs listed in ``skip`` (such as bonds) are left out, and only pairs
    within ``reach`` Å are searched; returns ``reach`` if none is left.

    >>> r = [[0, 0, 0], [1.0, 0, 0], [5, 5, 5]]
    >>> print(closest_approach(r, 10 * np.eye(3)),
    ...       closest_approach(r, 10 * np.eye(3), skip=[[0, 1]]))
    1.0 3.0
    """
    i, j, _, dist = all_pairs(positions, cell, reach)
    if skip is not None and len(i):
        pairs = np.sort(np.asarray(skip, dtype=int).reshape(-1, 2), axis=1)
        n = len(np.asarray(positions))
        bonded = np.isin(i * n + j, pairs[:, 0] * n + pairs[:, 1])
        dist = dist[~bonded]
    return float(dist.min()) if len(dist) else float(reach)


def largest_force(forces: ArrayLike) -> float:
    """The largest |Fᵢ| over the atoms, in eV/Å.

    >>> print(largest_force([[3.0, 4.0, 0.0], [0.0, 1.0, 0.0]]))
    5.0
    """
    f = np.asarray(forces, dtype=float)
    return float(np.sqrt(np.max(np.sum(f * f, axis=-1))))


def centre_of_mass_path(masses: ArrayLike, positions: ArrayLike) -> NDArray:
    """The distance of the centre of mass from its start, frame by frame.

    ``positions`` must be unwrapped, shape (frames, N, 3).

    >>> r = np.array([[[0.0, 0, 0], [1, 0, 0]], [[0.5, 0, 0], [1.5, 0, 0]]])
    >>> centre_of_mass_path([1.0, 1.0], r)
    array([0. , 0.5])
    """
    m = np.asarray(masses, dtype=float)
    r = np.asarray(positions, dtype=float)
    centre = np.einsum("i,tix->tx", m, r) / m.sum()
    return np.linalg.norm(centre - centre[0], axis=1)


def largest_jump(positions: ArrayLike, cell: ArrayLike) -> float:
    """The largest move of an atom between frames, in cell widths.

    The move is divided by the smallest perpendicular width of the cell.

    An atom wrapped back into the cell between two frames seems to jump
    by most of a width; in an unwrapped record of a healthy run every
    move is a small fraction.

    >>> r = np.array([[[0.5, 0, 0]], [[9.5, 0, 0]]])
    >>> print(round(largest_jump(r, 10 * np.eye(3)), 12))
    0.9
    """
    r = np.asarray(positions, dtype=float)
    moves = np.linalg.norm(np.diff(r, axis=0), axis=-1)
    return float(moves.max() / perpendicular_widths(cell).min())


def rms_displacement(positions: ArrayLike) -> NDArray:
    """The root-mean-square distance of the atoms from their starts.

    ``positions`` must be unwrapped, shape (frames, N, 3); the result has
    one value per frame, in Å.

    >>> r = np.array([[[0.0, 0, 0], [0, 0, 0]], [[3.0, 4, 0], [0, 0, 0]]])
    >>> print(rms_displacement(r).round(4))
    [0.     3.5355]
    """
    r = np.asarray(positions, dtype=float)
    return np.sqrt(np.mean(np.sum((r - r[0]) ** 2, axis=-1), axis=-1))


def unwrap(positions: ArrayLike, cell: ArrayLike) -> NDArray:
    """Frames of wrapped positions made continuous again.

    Each move between frames is replaced by its nearest copy, which
    recovers the path if no atom moves half a width between frames
    (Section 10.1).

    >>> r = np.array([[[9.8, 0, 0]], [[0.1, 0, 0]]])
    >>> unwrap(r, 10 * np.eye(3))[:, 0, 0].round(6)
    array([ 9.8, 10.1])
    """
    r = np.asarray(positions, dtype=float)
    moves = minimum_image(np.diff(r, axis=0).reshape(-1, 3), cell)
    moves = moves.reshape(r.shape[0] - 1, *r.shape[1:])
    return np.concatenate([r[:1], r[0] + np.cumsum(moves, axis=0)])


def fractional_spread(positions: ArrayLike, cell: ArrayLike) -> float:
    """The range of the fractional coordinates of a frame.

    It is at most 1 when the positions are wrapped into the cell, and
    grows as unwrapped atoms wander from it.

    >>> print(fractional_spread([[0.5, 0, 0], [25.0, 0, 0]], 10 * np.eye(3)))
    2.45
    """
    s = to_fractional(np.asarray(positions, dtype=float), cell)
    return float(np.max(np.ptp(s, axis=0)))


def excess_kurtosis(masses: ArrayLike, velocities: ArrayLike) -> float:
    """⟨u⁴⟩/⟨u²⟩² − 3 for the components u = √m v of all the velocities.

    Zero for velocities drawn from the Maxwell-Boltzmann distribution, in
    which each u is Gaussian with the same variance (Section 12.5); −1.2
    for components spread evenly over an interval.

    >>> rng = np.random.default_rng(1)
    >>> m = np.ones(20000)
    >>> print(round(excess_kurtosis(m, rng.normal(size=(20000, 3))), 1),
    ...       round(excess_kurtosis(m, rng.uniform(-1, 1, (20000, 3))), 1))
    0.0 -1.2
    """
    m = np.asarray(masses, dtype=float)
    u = np.sqrt(m)[..., :, None] * np.asarray(velocities, dtype=float)
    u = u.reshape(-1)
    return float(np.mean(u**4) / np.mean(u**2) ** 2 - 3)


@dataclass
class HealthReport:
    """The checks of ``health_report``: name → (value, limit, passed)."""

    checks: dict = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """Whether every check that could be run passed."""
        return all(passed for _, _, passed in self.checks.values())

    def __str__(self) -> str:
        lines = [f"  {'pass' if ok else 'FAIL'}  {name}: {value:.4g} "
                 f"(limit {limit:.4g})"
                 for name, (value, limit, ok) in self.checks.items()]
        lines.append("healthy" if self.healthy else "not healthy")
        return "\n".join(lines)


def health_report(
    positions: ArrayLike | None,
    cell: ArrayLike | None,
    masses: ArrayLike | None = None,
    velocities: ArrayLike | None = None,
    times: ArrayLike | None = None,
    potential: ArrayLike | None = None,
    kinetic: ArrayLike | None = None,
    symbols: list | None = None,
    temperature: float | None = None,
    n_free: int | None = None,
    ensemble: str = "nve",
    momentum_kept: bool = True,
) -> HealthReport:
    """Every check of Chapters 11 to 17 that the record allows.

    ``positions`` (frames, N, 3) in Å, as saved, wrapped or not; ``cell``
    with the lattice vectors as columns; ``masses`` in amu; ``velocities``
    (frames, N, 3) in Å/fs; ``times`` in fs; ``potential`` and ``kinetic``
    per frame in eV; ``symbols`` for the covalent radii; ``temperature``
    the target in K; ``ensemble`` 'nve' or 'nvt'. A check runs only when
    its inputs are given; ``positions`` and ``cell`` may be None for a
    record that kept only energies. The limits, and why:

    - continuity: no move between frames of half the smallest width or
      more, the most that ``unwrap`` can undo (Section 10.1);
    - centre of mass: it moves at most three times the root-mean-square
      displacement over √N, how far the independent motion of N atoms of
      equal mass would carry it (Exercise 17.17), for a run that keeps the
      total momentum, in which it should not move at all; with
      ``momentum_kept`` False, as under Langevin or Andersen dynamics,
      whose random forces move the centre as one free particle further
      than that (Section 13.8), the check is left out;
    - closest approach: no two atoms nearer than half the sum of their
      covalent radii, far inside any bond (Section 16.3), searched in the
      first and last frames out to twice the largest radius (1 is
      reported when no pair is that near);
    - energy, at fixed energy only: the spread of K + U below a twentieth
      of that of K, the fraction (ωδt)²/4 of Section 9.6 at about fourteen
      steps to the period of the fastest vibration, and no drift by the
      test of Chapter 15;
    - temperature: under a thermostat, the mean within three standard
      errors of the target, and, for one that claims the canonical
      ensemble, the spread of the kinetic temperature √(2/N_f) of the
      target, to three errors of a spread, √(g/2n) (Chapters 12 and 15); at
      fixed energy, where nothing holds the temperature at the target, the
      mean within a fifth of it, a rule of thumb between the 4% by which a
      liquid prepared at the target moves and the half that a start on a
      lattice loses (Section 12.9);
    - velocities: an excess kurtosis within 3√(24/n) of zero for the
      n = 3N mass-weighted components (Section 12.5) of the first frame and
      of the last, three times the scatter of a Gaussian sample's
      (Exercise 17.19): a start drawn wrongly shows in the first, and
      relaxes before the last.

    >>> rng = np.random.default_rng(0)
    >>> r = 10 * rng.random((1, 64, 3)) + np.cumsum(
    ...     rng.normal(0, 0.01, (50, 64, 3)), axis=0)
    >>> health_report(r, 30 * np.eye(3), np.ones(64)).healthy
    True
    """
    from mdlab.analysis import stats
    from mdlab.statmech import degrees_of_freedom
    from mdlab.units import KB

    report = HealthReport()
    r = np.zeros((0, 0, 3)) if positions is None else np.asarray(
        positions, dtype=float)
    h = None if cell is None else np.asarray(cell, dtype=float)
    n = r.shape[1] if positions is not None else (
        None if masses is None else len(masses))
    if len(r) > 1:
        jump = largest_jump(r, h)
        report.checks["largest move between frames / cell width"] = (
            jump, 0.5, jump < 0.5)
    if masses is not None and len(r) > 1 and momentum_kept:
        path = centre_of_mass_path(masses, unwrap(r, h))[-1]
        spread = rms_displacement(unwrap(r, h))[-1]
        limit = 3 * spread / np.sqrt(n)
        report.checks["path of the centre of mass / Å"] = (
            path, limit, path <= limit)
    if symbols is not None and len(r):
        from mdlab.bonds import covalent_radii

        radii = covalent_radii(symbols)
        worst = 1.0  # what is reported when no pair is within reach
        for frame in (r[0], r[-1]):
            i, j, _, dist = all_pairs(frame, h, 2 * radii.max())
            if len(dist):
                worst = min(worst, np.min(dist / (radii[i] + radii[j])))
        report.checks["closest approach / sum of covalent radii"] = (
            worst, 0.5, worst >= 0.5)
    if potential is not None and kinetic is not None and ensemble == "nve":
        u = np.asarray(potential, dtype=float)
        k = np.asarray(kinetic, dtype=float)
        ratio = energy_fluctuation(u, k)
        report.checks["spread of K + U / spread of K"] = (
            ratio, 0.05, ratio < 0.05)
        if times is not None:
            _, _, z = stats.drift_test(np.asarray(times, dtype=float), u + k)
            report.checks["drift of K + U / its error"] = (
                abs(z), stats.DRIFT_LIMIT, abs(z) < stats.DRIFT_LIMIT)
    if kinetic is not None and (n is not None or n_free is not None):
        nf = degrees_of_freedom(n) if n_free is None else n_free
        t = 2 * np.asarray(kinetic, dtype=float) / (nf * KB)
        mean, error, g = stats.standard_error(t)
        if temperature is not None and ensemble == "nve":
            off = abs(mean / temperature - 1)
            report.checks["mean temperature / target, less 1"] = (
                off, 0.2, off < 0.2)
        elif temperature is not None:
            miss = abs(mean - temperature) / error
            report.checks["miss of the mean temperature / its error"] = (
                miss, 3.0, miss < 3.0)
            if ensemble == "nvt":
                ratio = np.std(t, ddof=1) / (np.sqrt(2 / nf) * temperature)
                limit = 3 * np.sqrt(g / (2 * len(t)))
                report.checks["spread of T / canonical spread, less 1"] = (
                    abs(ratio - 1), limit, abs(ratio - 1) <= limit)
    if masses is not None and velocities is not None:
        v = np.asarray(velocities, dtype=float)
        excess = max(excess_kurtosis(masses, v[0]),
                     excess_kurtosis(masses, v[-1]), key=abs)
        limit = 3 * np.sqrt(24 / (3 * n))
        report.checks["excess kurtosis of the velocities"] = (
            abs(excess), limit, abs(excess) <= limit)
    return report
