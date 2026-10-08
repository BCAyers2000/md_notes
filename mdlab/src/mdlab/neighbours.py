"""Neighbour lists: Chapter 10.

Finding the pairs of atoms closer than a cutoff r_c in a periodic cell:

- ``all_pairs``: every pair tested, N(N − 1)/2 distances (Section 10.4);
- ``cell_list_pairs``: the atoms sorted into cells at least r_c wide,
  each tested only against those in its own and the 26 neighbouring
  cells, so that the work grows as N;
- ``VerletList``: the pairs within r_c + skin, kept from step to step and
  rebuilt only when some atom has moved more than half the skin; when a
  barostat changes the cell, the rule allows for the strain too
  (Section 14.5).

Each returns the pairs as two index arrays i < j, with the separation
vectors rⱼ − rᵢ reduced to their nearest image and their lengths. All of
them need r_c at most half the smallest perpendicular width of the cell,
so that each pair has at most one copy within r_c and rounding finds it.

Units
-----
Lengths in Å, or any consistent unit.
"""

import itertools

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.cell import minimum_image, perpendicular_widths, to_fractional

Pairs = tuple[NDArray, NDArray, NDArray, NDArray]


def _check(cell: NDArray, r_cut: float) -> None:
    if r_cut > 0.5 * perpendicular_widths(cell).min():
        raise ValueError(
            "the cutoff exceeds half the smallest width of the cell; "
            "use a larger cell"
        )


def _within(
    r: NDArray, cell: NDArray, i: NDArray, j: NDArray, r_cut: float
) -> Pairs:
    d = minimum_image(r[j] - r[i], cell)
    dist = np.linalg.norm(d, axis=1)
    keep = dist < r_cut
    return i[keep], j[keep], d[keep], dist[keep]


def all_pairs(positions: ArrayLike, cell: ArrayLike, r_cut: float) -> Pairs:
    """Every pair closer than r_cut, found by testing all N(N − 1)/2.

    >>> r = [[0.1, 0.1, 0.1], [9.9, 0.1, 0.1], [5.0, 5.0, 5.0]]
    >>> i, j, d, dist = all_pairs(r, 10 * np.eye(3), 1.0)
    >>> print(i, j, dist.round(6))
    [0] [1] [0.2]
    """
    r = np.asarray(positions, dtype=float)
    h = np.asarray(cell, dtype=float)
    _check(h, r_cut)
    i, j = np.triu_indices(len(r), k=1)
    return _within(r, h, i, j, r_cut)


def cell_list_pairs(
    positions: ArrayLike, cell: ArrayLike, r_cut: float
) -> Pairs:
    """Every pair closer than r_cut, found through a cell list.

    The cell is divided into n₁ × n₂ × n₃ small cells, with nₐ the whole
    number of times r_c fits across the width wₐ, so each small cell is at
    least r_c wide and a partner within r_c lies in the same small cell or
    one of its 26 neighbours. With fewer than three small cells along some
    direction the neighbours would repeat, and all pairs are tested
    instead.
    """
    r = np.asarray(positions, dtype=float)
    h = np.asarray(cell, dtype=float)
    _check(h, r_cut)
    counts = np.floor(perpendicular_widths(h) / r_cut).astype(int)
    if counts.min() < 3:
        return all_pairs(r, h, r_cut)
    s = to_fractional(r, h)
    index = np.floor((s - np.floor(s)) * counts).astype(int) % counts
    flat = np.ravel_multi_index(index.T, counts)
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(np.prod(counts) + 1))
    bounds = zip(starts[:-1], starts[1:], strict=True)
    members = [order[a:b] for a, b in bounds]
    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    found_i, found_j = [], []
    for home in itertools.product(*(range(c) for c in counts)):
        atoms = members[np.ravel_multi_index(home, counts)]
        if len(atoms) == 0:
            continue
        for offset in offsets:
            other = tuple((np.array(home) + offset) % counts)
            partners = members[np.ravel_multi_index(other, counts)]
            if len(partners) == 0:
                continue
            ii, jj = np.meshgrid(atoms, partners, indexing="ij")
            mask = ii < jj
            found_i.append(ii[mask])
            found_j.append(jj[mask])
    i = np.concatenate(found_i) if found_i else np.zeros(0, dtype=int)
    j = np.concatenate(found_j) if found_j else np.zeros(0, dtype=int)
    return _within(r, h, i, j, r_cut)


class VerletList:
    """The pairs within r_c + skin, reused until an atom has moved far.

    Two atoms that have each moved less than half the skin since the list
    was built have closed their distance by less than the skin, so every
    pair now within r_c was within r_c + skin then and is on the list.

    If the cell has changed since the build, from h_b to h, the strain
    S = h h_b⁻¹ carries each separation d_b of the build to S d_b, which
    differs from d_b by at most e|d_b|, with e the largest stretch of
    S − I. With u the largest move of an atom from its strained reference
    position, a pair now within r_c had |d_b|(1 − e) ≤ r_c + 2u, so the
    list holds while (r_c + 2u)/(1 − e) ≤ r_c + skin; with the cell
    fixed, e = 0 and this is u ≤ skin/2.

    >>> rng = np.random.default_rng(0)
    >>> r = rng.uniform(0, 12, (100, 3))
    >>> nl = VerletList(12 * np.eye(3), 2.5, 0.5)
    >>> i, j, d, dist = nl.pairs(r)
    >>> step = np.array([1.0, 0.0, 0.0])
    >>> nl.needs_rebuild(r + 0.2 * step), nl.needs_rebuild(r + 0.3 * step)
    (False, True)
    """

    def __init__(self, cell: ArrayLike, r_cut: float, skin: float):
        self.cell = np.asarray(cell, dtype=float)
        self.built_cell = self.cell.copy()
        self.r_cut, self.skin = r_cut, skin
        self.reference: NDArray | None = None
        self.i = self.j = np.zeros(0, dtype=int)
        self.builds = 0

    def set_cell(self, cell: ArrayLike) -> None:
        """Change the cell; the list is kept while the strain allows."""
        self.cell = np.asarray(cell, dtype=float)

    def needs_rebuild(self, positions: ArrayLike) -> bool:
        """Whether a pair within r_c could be missing from the list."""
        if self.reference is None:
            return True
        if not np.array_equal(self.cell, self.built_cell):
            return self._strained_needs_rebuild(positions)
        moved = minimum_image(
            np.asarray(positions, dtype=float) - self.reference, self.cell
        )
        return bool(
            np.max(np.einsum("ix,ix->i", moved, moved))
            > (0.5 * self.skin) ** 2
        )

    def _strained_needs_rebuild(self, positions: ArrayLike) -> bool:
        """The rebuild test once the cell has changed since the build.

        A pair now within r_c was within (r_c + 2u)/(1 - e) at the build,
        with u the largest move beyond the strain's and e the strain's
        largest stretch of a separation.
        """
        strain = self.cell @ np.linalg.inv(self.built_cell)
        stretch = np.linalg.norm(strain - np.eye(3), 2)
        if stretch >= 1:
            return True
        moved = minimum_image(
            np.asarray(positions, dtype=float) - self.reference @ strain.T,
            self.cell,
        )
        largest = np.sqrt(np.max(np.einsum("ix,ix->i", moved, moved)))
        return bool(
            (self.r_cut + 2 * largest) / (1 - stretch)
            > self.r_cut + self.skin
        )

    def build(self, positions: ArrayLike) -> None:
        """Store the pairs within r_c + skin, by a cell list."""
        r = np.asarray(positions, dtype=float)
        self.i, self.j, _, _ = cell_list_pairs(
            r, self.cell, self.r_cut + self.skin
        )
        self.reference = r.copy()
        self.built_cell = self.cell.copy()
        self.builds += 1

    def pairs(self, positions: ArrayLike) -> Pairs:
        """The pairs now within r_c, rebuilding the list first if needed."""
        r = np.asarray(positions, dtype=float)
        if self.needs_rebuild(r):
            self.build(r)
        return _within(r, self.cell, self.i, self.j, self.r_cut)
