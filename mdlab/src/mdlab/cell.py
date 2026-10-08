"""Periodic cells: Chapter 10.

A cell is a 3 × 3 matrix h whose columns are the lattice vectors a, b and
c. Every position r has fractional coordinates s = h⁻¹r, and the copies
of an atom sit at r + h n for every vector n of whole numbers.

- ``cell_volume``: |a·(b × c)|, the volume of the cell (Section 10.1);
- ``to_fractional`` and ``to_cartesian``: s = h⁻¹r and r = h s;
- ``wrap``: the copy of each position inside the cell, 0 ≤ s < 1;
- ``perpendicular_widths``: the distances between opposite faces;
- ``reciprocal_vectors``: a*, b*, c*, the rows of 2π h⁻¹, with a·a* = 2π
  and a·b* = 0
  (Section 10.3);
- ``minimum_image``: separations reduced by rounding their fractional
  coordinates (Section 10.2);
- ``nearest_image``: the shortest copy of each separation, by a search
  over neighbouring images, the reference against which rounding is
  tested.

Units
-----
Lengths in Å, or any consistent unit.
"""

import itertools

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _cell(cell: ArrayLike) -> NDArray:
    h = np.asarray(cell, dtype=float)
    if h.shape != (3, 3):
        raise ValueError("a cell is a 3 x 3 matrix of lattice vectors")
    return h


def cell_volume(cell: ArrayLike) -> float:
    """The volume |a·(b × c)| of the cell whose columns are a, b and c.

    >>> round(cell_volume(np.diag([2.0, 3.0, 4.0])), 12)
    24.0
    """
    return float(abs(np.linalg.det(_cell(cell))))


def to_fractional(positions: ArrayLike, cell: ArrayLike) -> NDArray:
    """Fractional coordinates s = h⁻¹r of each row of ``positions``.

    >>> to_fractional([[1.0, 1.5, 3.0]], np.diag([2.0, 3.0, 4.0]))
    array([[0.5 , 0.5 , 0.75]])
    """
    r = np.asarray(positions, dtype=float)
    return np.linalg.solve(_cell(cell), r.T).T


def to_cartesian(fractional: ArrayLike, cell: ArrayLike) -> NDArray:
    """Positions r = h s from fractional coordinates."""
    return np.asarray(fractional, dtype=float) @ _cell(cell).T


def wrap(positions: ArrayLike, cell: ArrayLike) -> NDArray:
    """The copy of each position inside the cell, with 0 ≤ s < 1.

    >>> wrap([[2.5, -0.5, 9.0]], np.diag([2.0, 3.0, 4.0]))
    array([[0.5, 2.5, 1. ]])
    """
    s = to_fractional(positions, cell)
    return to_cartesian(s - np.floor(s), cell)


def perpendicular_widths(cell: ArrayLike) -> NDArray:
    """The distances between the three pairs of opposite faces.

    The face spanned by b and c has area |b × c|, so the width across it
    is V/|b × c|, and likewise for the other two.

    >>> h = [[2.0, -1.0, 0.0], [0.0, 3 ** 0.5, 0.0], [0.0, 0.0, 5.0]]
    >>> perpendicular_widths(h).round(6)
    array([1.732051, 1.732051, 5.      ])
    """
    h = _cell(cell)
    a, b, c = h.T
    volume = cell_volume(h)
    return volume / np.array(
        [
            np.linalg.norm(np.cross(b, c)),
            np.linalg.norm(np.cross(c, a)),
            np.linalg.norm(np.cross(a, b)),
        ]
    )


def reciprocal_vectors(cell: ArrayLike) -> NDArray:
    """The reciprocal vectors a*, b*, c* as rows: the rows of 2π h⁻¹.

    They satisfy a·a* = b·b* = c·c* = 2π, every other such product being
    zero, so that a wave cos(G·r), with G a whole-number combination of
    them, repeats with the cell.

    >>> np.round(reciprocal_vectors(np.diag([2.0, 1.0, 4.0])) / np.pi, 6)
    array([[1. , 0. , 0. ],
           [0. , 2. , 0. ],
           [0. , 0. , 0.5]])
    """
    return 2 * np.pi * np.linalg.inv(_cell(cell))


def minimum_image(separations: ArrayLike, cell: ArrayLike) -> NDArray:
    """Separations with their fractional coordinates rounded into [−½, ½].

    This gives the nearest copy for every separation shorter than half
    the smallest perpendicular width, in any cell; beyond that, in a
    skewed cell, it may not (Section 10.2).

    >>> minimum_image([[1.8, -2.9, 0.1]], np.diag([2.0, 3.0, 4.0])).round(6)
    array([[-0.2,  0.1,  0.1]])
    """
    s = to_fractional(separations, cell)
    return to_cartesian(s - np.round(s), cell)


def nearest_image(
    separations: ArrayLike, cell: ArrayLike, reach: int = 2
) -> NDArray:
    """The shortest copy of each separation, by a search of the images.

    The separations are first rounded, then compared with their copies up
    to ``reach`` cells away along each lattice vector.
    """
    h = _cell(cell)
    d = minimum_image(separations, h)
    shifts = np.array(
        list(itertools.product(range(-reach, reach + 1), repeat=3)),
        dtype=float,
    )
    candidates = d[:, None, :] + (shifts @ h.T)[None, :, :]
    best = np.argmin(np.einsum("ikx,ikx->ik", candidates, candidates), 1)
    return candidates[np.arange(len(d)), best]
