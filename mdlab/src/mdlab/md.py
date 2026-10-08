"""A molecular dynamics code: Chapter 10.

The pieces of the earlier chapters put together for atoms in a periodic
cell:

- ``starting_velocities``: random velocities with the drift of the centre
  of mass removed, scaled to a chosen kinetic energy (Section 10.5);
- ``PairModel``: the energy and forces of a pair potential in a cell,
  from a Verlet neighbour list (Sections 10.4 and 10.6), with the virial
  tensor, a cell that may change and, if asked, a pair potential for
  each pair of atom types (Chapter 14);
- ``NoForces``: no interactions at all, an ideal gas (Section 14.5);
- ``EwaldModel``: the energy and forces of point charges by the Ewald sum
  (Section 10.3);
- ``sum_of``: the energy and forces of several models added;
- ``minimise``: steepest descent to the nearest minimum of the energy,
  for a start whose atoms overlap (Section 15.9);
- ``run``: the loop, velocity Verlet over a model, recording the energies
  and, if asked, writing the frames to an extxyz file (Section 10.6) and
  handing each recorded frame to observers that analyse it as the run
  goes (Section 16.4).

Each model is called with the positions and returns the potential energy
and the forces, the form ``integrators.integrate`` expects too.

Units
-----
Positions in Å, velocities in Å/fs, masses in amu, energies in eV, times
in fs; or reduced units with ``force_to_accel = mv2_to_energy = 1``.
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.cell import wrap
from mdlab.ewald import ewald_energy_forces
from mdlab.io import write_extxyz
from mdlab.neighbours import VerletList
from mdlab.units import FORCE_TO_ACCEL, MV2_TO_EV

EnergyForces = Callable[[NDArray], tuple[float, NDArray]]


def starting_velocities(
    masses: ArrayLike,
    kinetic_energy: float,
    rng: np.random.Generator,
    mv2_to_energy: float = MV2_TO_EV,
) -> NDArray:
    """Random velocities with no drift and the given kinetic energy.

    Each component is drawn from a normal distribution of width 1/√mᵢ, so
    that light and heavy atoms carry the same kinetic energy on average;
    the velocity of the centre of mass is then subtracted, and all are
    scaled so that ½ Σ mᵢ|vᵢ|² is ``kinetic_energy``.

    >>> m = np.array([1.0, 2.0, 3.0, 4.0])
    >>> v = starting_velocities(m, 6.0, np.random.default_rng(1), 1.0)
    >>> print(np.round(m @ v, 12) + 0.0, round(0.5 * m @ (v * v).sum(1), 12))
    [0. 0. 0.] 6.0
    """
    m = np.asarray(masses, dtype=float)
    v = rng.normal(size=(len(m), 3)) / np.sqrt(m)[:, None]
    v -= (m @ v) / m.sum()
    kinetic = 0.5 * mv2_to_energy * float(m @ (v * v).sum(1))
    return v * np.sqrt(kinetic_energy / kinetic)


class PairModel:
    """A pair potential in a periodic cell, with a Verlet neighbour list.

    ``pair`` returns φ(r) and dφ/dr for an array of distances, as the
    functions of ``mdlab.potentials`` do; it should already be cut off at
    ``r_cut`` (``potentials.with_cutoff``). With ``types``, one whole
    number per atom, ``pair`` is instead a mapping from each pair of types
    (a, b), a ≤ b, to its pair function.

    Each call also stores the virial tensor W_αβ = Σᵢ r_iα F_iβ, written
    as −Σ_pairs d_α F_β with d = rⱼ − rᵢ and F the force on i from j, so
    that it needs no position, only separations (Section 14.4); its trace
    is the virial W of Section 8.4. It keeps, too, the pairs within r_c
    and their distances, ``last_pairs``, for an observer that analyses the
    frame without measuring them again (Section 16.4).
    """

    def __init__(
        self,
        pair: Callable[[NDArray], tuple[NDArray, NDArray]]
        | dict[tuple[int, int], Callable[[NDArray], tuple[NDArray, NDArray]]],
        cell: ArrayLike,
        r_cut: float,
        skin: float,
        types: ArrayLike | None = None,
    ):
        self.pair = pair
        self.types = None if types is None else np.asarray(types, dtype=int)
        self.neighbours = VerletList(cell, r_cut, skin)
        self.virial = np.zeros((3, 3))
        self.last_pairs: tuple[NDArray, NDArray, NDArray] | None = None

    @property
    def cell(self) -> NDArray:
        """The current cell, columns the lattice vectors."""
        return self.neighbours.cell

    def set_cell(self, cell: ArrayLike) -> None:
        """Change the cell, as a barostat does."""
        self.neighbours.set_cell(cell)

    def _pair(self, i: NDArray, j: NDArray, dist: NDArray):
        if self.types is None:
            return self.pair(dist)
        a, b = self.types[i], self.types[j]
        low, high = np.minimum(a, b), np.maximum(a, b)
        phi, dphi = np.zeros_like(dist), np.zeros_like(dist)
        for (ta, tb), function in self.pair.items():
            chosen = (low == min(ta, tb)) & (high == max(ta, tb))
            if chosen.any():
                phi[chosen], dphi[chosen] = function(dist[chosen])
        return phi, dphi

    def __call__(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """The potential energy and the forces at ``positions``."""
        r = np.asarray(positions, dtype=float)
        i, j, d, dist = self.neighbours.pairs(r)
        self.last_pairs = (i, j, dist)
        phi, dphi = self._pair(i, j, dist)
        push = (dphi / dist)[:, None] * d  # the force on i from j
        forces = np.zeros_like(r)
        np.add.at(forces, i, push)
        np.add.at(forces, j, -push)
        self.virial = -np.einsum("pa,pb->ab", d, push)
        return float(np.sum(phi)), forces


class NoForces:
    """Atoms that do not interact: U = 0, no forces, no virial."""

    def __init__(self, cell: ArrayLike):
        self.cell = np.asarray(cell, dtype=float)
        self.virial = np.zeros((3, 3))

    def set_cell(self, cell: ArrayLike) -> None:
        """Change the cell, which changes nothing else."""
        self.cell = np.asarray(cell, dtype=float)

    def __call__(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """Zero energy and zero forces."""
        return 0.0, np.zeros_like(np.asarray(positions, dtype=float))


class EwaldModel:
    """Point charges in a periodic cell, by the Ewald sum."""

    def __init__(
        self,
        charges: ArrayLike,
        cell: ArrayLike,
        alpha: float | None = None,
        accuracy: float = 1e-8,
        coulomb: float | None = None,
    ):
        self.charges = np.asarray(charges, dtype=float)
        self.cell = np.asarray(cell, dtype=float)
        self.options = {"alpha": alpha, "accuracy": accuracy}
        if coulomb is not None:
            self.options["coulomb"] = coulomb

    def __call__(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """The Ewald energy and the forces at ``positions``."""
        energy, forces, _ = ewald_energy_forces(
            self.charges, positions, self.cell, **self.options
        )
        return energy, forces


def sum_of(models: Sequence[EnergyForces]) -> EnergyForces:
    """A model whose energy and forces are those of ``models`` added."""

    def energy_forces(positions: ArrayLike) -> tuple[float, NDArray]:
        results = [model(positions) for model in models]
        return (
            float(sum(u for u, _ in results)),
            np.sum([f for _, f in results], axis=0),
        )

    return energy_forces


def minimise(
    model: EnergyForces,
    positions: ArrayLike,
    step: float = 0.1,
    force_tolerance: float = 1e-2,
    max_steps: int = 10000,
) -> tuple[NDArray, NDArray, NDArray]:
    """Steepest descent until no force exceeds ``force_tolerance``.

    Each trial moves every atom along its force, the atom with the largest
    force by ``step`` Å. A trial that lowers the energy is kept and the
    step grows by a fifth; one that does not is refused and the step
    halves. Returns the positions, the energy after each kept move, and
    the largest force after each.
    """
    r = np.array(positions, dtype=float)
    u, f = model(r)
    energies, largest = [u], [np.max(np.linalg.norm(f, axis=-1))]
    for _ in range(max_steps):
        if largest[-1] < force_tolerance:
            break
        trial = r + step * f / largest[-1]
        u_trial, f_trial = model(trial)
        if u_trial < u:
            r, u, f = trial, u_trial, f_trial
            energies.append(u)
            largest.append(np.max(np.linalg.norm(f, axis=-1)))
            step *= 1.2
        else:
            step *= 0.5
    return r, np.array(energies), np.array(largest)


def run(
    model: EnergyForces,
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    cell: ArrayLike,
    dt: float,
    n_steps: int,
    every: int = 1,
    path: str | None = None,
    symbols: Sequence[str] | None = None,
    force_to_accel: float = FORCE_TO_ACCEL,
    observers: Sequence[Callable[[float, NDArray, NDArray], None]] = (),
) -> dict[str, NDArray]:
    """Velocity Verlet over ``model``, recording every ``every`` steps.

    The positions are integrated without wrapping, so that each atom's
    path stays continuous; the frames written to ``path`` are wrapped
    into the cell, as files of periodic systems usually are. Each
    observer is called at every recorded frame with the time and the
    positions and velocities, which it must not change. Returns the
    recorded "times", "positions", "velocities", "potential" and
    "kinetic" energies.
    """
    m = np.asarray(masses, dtype=float)[:, None]
    r = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    to_energy = 1.0 if force_to_accel == 1 else MV2_TO_EV
    keys = ("times", "positions", "velocities", "potential", "kinetic")
    record: dict[str, list] = {key: [] for key in keys}
    u, f = model(r)
    for step in range(n_steps + 1):
        if step > 0:
            v += 0.5 * dt * force_to_accel * f / m
            r += dt * v
            u, f = model(r)
            v += 0.5 * dt * force_to_accel * f / m
        if step % every == 0:
            kinetic = 0.5 * to_energy * float(np.sum(m * v * v))
            for key, value in zip(
                keys, (step * dt, r.copy(), v.copy(), u, kinetic), strict=True
            ):
                record[key].append(value)
            for observe in observers:
                observe(step * dt, r, v)
    out = {key: np.array(value) for key, value in record.items()}
    if path is not None:
        h = np.asarray(cell, dtype=float)
        write_extxyz(
            path,
            symbols if symbols is not None else ["X"] * len(r),
            [wrap(x, h) for x in out["positions"]],
            h,
            velocities=out["velocities"],
            scalars={
                "energy": out["potential"],
                "kinetic_energy": out["kinetic"],
            },
        )
    return out
