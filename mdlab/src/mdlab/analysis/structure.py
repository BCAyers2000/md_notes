"""How atoms arrange themselves: Chapter 16.

- ``rdf``: the radial distribution function g(r) of frames in a periodic
  cell, and the partial g_AB(r) between two sets of atoms, normalised as
  ASE's ``get_rdf`` (Section 16.1);
- ``running_coordination``: the mean number of partners within r of an
  atom, from g(r);
- ``structure_factor``: S(q) = |Σ_j e^{iq·r_j}|²/N on the wave vectors
  that repeat with the cell, averaged over frames and over the vectors of
  each length (Section 16.2);
- ``structure_factor_from_rdf``: 1 + 4πρ ∫ r² [g(r) − 1] sin(qr)/(qr) dr.

Units
-----
Lengths in Å, wave numbers in 1/Å.
"""

import itertools

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.cell import cell_volume, reciprocal_vectors
from mdlab.neighbours import cell_list_pairs


def _frames(positions: ArrayLike) -> NDArray:
    r = np.asarray(positions, dtype=float)
    return r[None] if r.ndim == 2 else r


def _chosen(n: int, index: ArrayLike | None) -> NDArray:
    """True for the atoms in ``index``, or for all of them if it is None."""
    chosen = np.zeros(n, dtype=bool)
    chosen[slice(None) if index is None else np.asarray(index)] = True
    return chosen


def shell_volumes(edges: ArrayLike) -> NDArray:
    """The volumes 4π(r_{k+1}³ − r_k³)/3 between neighbouring edges."""
    e = np.asarray(edges, dtype=float)
    return 4 * np.pi / 3 * (e[1:] ** 3 - e[:-1] ** 3)


def rdf(
    positions: ArrayLike,
    cell: ArrayLike,
    r_max: float,
    n_bins: int,
    centres: ArrayLike | None = None,
    partners: ArrayLike | None = None,
    per_frame: bool = False,
) -> tuple[NDArray, NDArray]:
    """The radial distribution function, averaged over frames.

    Every pair closer than ``r_max`` is found through a cell list, so
    ``r_max`` may be at most half the smallest width of the cell. The
    number of partners in each shell around each centre is divided by the
    number an even spread would put there, ρ_B × (shell volume), with
    ρ_B = N_B/V; for one kind of atom g(r) then tends to 1 − 1/N far away,
    as in ASE. ``centres`` and ``partners`` are index arrays for a partial
    g_AB(r); by default every atom is both.

    Returns the middle of each bin and g, or g for each frame if
    ``per_frame``.
    """
    frames = _frames(positions)
    h = np.asarray(cell, dtype=float)
    in_a = _chosen(frames.shape[1], centres)
    in_b = _chosen(frames.shape[1], partners)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    ideal = in_a.sum() * in_b.sum() / cell_volume(h) * shell_volumes(edges)
    out = np.empty((len(frames), n_bins))
    for k, r in enumerate(frames):
        i, j, _, dist = cell_list_pairs(r, h, r_max)
        # a pair counts once for each end that is a centre whose partner
        # is the other end
        weight = (in_a[i] & in_b[j]).astype(float) + (in_a[j] & in_b[i])
        out[k] = np.histogram(dist, edges, weights=weight)[0] / ideal
    middle = 0.5 * (edges[1:] + edges[:-1])
    return middle, (out if per_frame else out.mean(axis=0))


def running_coordination(
    r: ArrayLike, g: ArrayLike, density: float
) -> NDArray:
    """n(r_k): the mean number of partners out to the end of bin k.

    ``r`` holds the middles of equal bins starting at 0, ``density`` the
    partners' number density ρ_B. Each bin adds ρ_B g × (shell volume),
    the number it held.
    """
    middle = np.asarray(r, dtype=float)
    width = middle[1] - middle[0]
    edges = np.append(middle - 0.5 * width, middle[-1] + 0.5 * width)
    return np.cumsum(density * np.asarray(g) * shell_volumes(edges))


def structure_factor(
    positions: ArrayLike, cell: ArrayLike, m_max: int
) -> tuple[NDArray, NDArray]:
    """S(q) on the wave vectors q = m₁a* + m₂b* + m₃c*, |m_i| ≤ m_max, m ≠ 0.

    For each frame and each q, S = (C² + S²)/N with C = Σ cos(q·r_j) and
    S = Σ sin(q·r_j); q and −q give the same value, so half of them are
    used. Returns the distinct lengths |q| and S averaged over the frames
    and over the vectors of each length.
    """
    frames = _frames(positions)
    n = frames.shape[1]
    grid = np.array(
        list(itertools.product(range(-m_max, m_max + 1), repeat=3))
    )
    half = grid[(grid[:, 0] > 0)
                | ((grid[:, 0] == 0) & (grid[:, 1] > 0))
                | ((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] > 0))]
    q = half @ reciprocal_vectors(cell)
    total = np.zeros(len(q))
    for r in frames:
        phase = r @ q.T
        total += (np.cos(phase).sum(0) ** 2 + np.sin(phase).sum(0) ** 2) / n
    length = np.round(np.linalg.norm(q, axis=1), 10)
    values, which = np.unique(length, return_inverse=True)
    mean = np.bincount(which, total) / np.bincount(which) / len(frames)
    return values, mean


def structure_factor_from_rdf(
    r: ArrayLike, g: ArrayLike, density: float, q: ArrayLike
) -> NDArray:
    """S(q) = 1 + 4πρ ∫ r² [g(r) − 1] sin(qr)/(qr) dr, by the bin sums.

    ``r`` holds the middles of equal bins; the integral stops where g
    stops, at most half the cell.
    """
    x = np.asarray(r, dtype=float)
    width = x[1] - x[0]
    qr = np.outer(np.asarray(q, dtype=float), x)
    return 1 + 4 * np.pi * density * width * np.sum(
        x * x * (np.asarray(g) - 1) * np.sinc(qr / np.pi), axis=1
    )
