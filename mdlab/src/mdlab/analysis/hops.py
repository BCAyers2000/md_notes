"""Hops between sites, and rates that admit when unknown: Chapter 17.

An atom that rattles in a site and now and then jumps to the next is
followed by the site it is nearest to, frame by frame. A visit shorter
than a least stay is a passing excursion, not a hop, in the same way
that `mdlab.bonds.persistent` drops brief bonds (Section 16.3). The
hops counted over an exposure give a rate with the error √n of a
Poisson count (Section 16.4), but only once enough are seen; with too
few, the rate is reported as unresolved, with the exposure that would
resolve it.

- ``nearest_site``: the index of the nearest site, by the nearest image;
- ``hops``: the hops of one atom's sequence of sites;
- ``rate``: a rate with its error, or the exposure still needed.

Units
-----
Lengths in Å; frames counted in whole numbers; exposures in whatever
unit of time the caller uses, which sets the unit of the rate.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.cell import minimum_image

#: The fractional error at which a rate counts as resolved: a fifth, which
#: needs 25 hops (Section 16.4: a rate from n events is known to 1/√n).
RESOLVED = 0.2
#: −ln 0.05: the expected count above which a run would see no event at
#: all in fewer than 5% of runs.
NONE_SEEN = -np.log(0.05)


def nearest_site(
    positions: ArrayLike, sites: ArrayLike, cell: ArrayLike
) -> NDArray:
    """The index of the site nearest each position, by the nearest image.

    ``positions`` (..., 3) and ``sites`` (m, 3) in Å; ``cell`` with the
    lattice vectors as columns. Returns an integer array of shape (...).

    >>> nearest_site([[0.9, 0, 0], [9.6, 0, 0]], [[0, 0, 0], [2, 0, 0]],
    ...              10 * np.eye(3))
    array([0, 0])
    """
    r = np.asarray(positions, dtype=float)
    s = np.asarray(sites, dtype=float)
    d = r[..., None, :] - s
    shape = d.shape
    d = minimum_image(d.reshape(-1, 3), cell).reshape(shape)
    return np.argmin(np.sum(d * d, axis=-1), axis=-1)


def hops(sites: ArrayLike, least_stay: int = 1) -> NDArray:
    """The hops in one atom's sequence of sites, frame by frame.

    Runs of the same site shorter than ``least_stay`` frames are passing
    excursions and are joined to the visit before them; a hop is a change
    between the visits that remain. Returns rows (frame, from, to), the
    frame being the first of the new visit.

    >>> hops([0, 0, 1, 0, 0, 2, 2, 2], least_stay=2)
    array([[5, 0, 2]])
    """
    s = np.asarray(sites, dtype=int)
    change = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [len(s)]]))
    kept = [(starts[0], s[starts[0]])]
    for start, length in zip(starts[1:], lengths[1:], strict=True):
        site = s[start]
        if length >= least_stay and site != kept[-1][1]:
            kept.append((start, site))
    return np.array([(kept[k][0], kept[k - 1][1], kept[k][1])
                     for k in range(1, len(kept))], dtype=int).reshape(-1, 3)


def rate(n_hops: int, exposure: float, resolved: float = RESOLVED) -> dict:
    """The rate of hops, with its error, or the exposure still needed.

    With n hops over the exposure X the rate is n/X with the error √n/X.
    It counts as resolved when 1/√n ≤ ``resolved``. Otherwise "needed" is
    the exposure that would bring n to 1/resolved² at this rate; with no
    hop at all the rate is below NONE_SEEN/X with 95% confidence, and
    "needed" uses that bound, so it is the least exposure required.

    >>> rate(100, 50.0)["resolved"], rate(4, 2.0)["needed"]
    (True, 12.5)
    """
    want = int(np.ceil(1 / resolved**2 - 1e-9))  # hops needed, 25 for 0.2
    if n_hops == 0:
        bound = NONE_SEEN / exposure
        return {"rate": 0.0, "error": None, "upper": bound,
                "resolved": False, "needed": want / bound}
    k = n_hops / exposure
    return {"rate": k, "error": np.sqrt(n_hops) / exposure, "upper": None,
            "resolved": bool(n_hops >= want), "needed": want / k}
