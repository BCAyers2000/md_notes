"""Free energy from where a run spends its time: Chapter 16.

- ``free_energy``: F(x) = −k_BT ln P(x) from a histogram of one or more
  recorded coordinates, with error bars from the scatter of the
  histograms of consecutive blocks (Section 16.11).

Units
-----
Energies in eV, temperatures in K; the coordinates in any unit.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.units import KB


def free_energy(
    values: ArrayLike,
    bins: int | ArrayLike | list,
    temperature: float,
    n_blocks: int = 10,
    value_range: list | None = None,
) -> tuple[list[NDArray], NDArray, NDArray]:
    """F = −k_BT ln P on a grid of bins, its least value set to zero.

    ``values`` is (n,) for one coordinate or (n, d) for d of them. They
    are cut into ``n_blocks`` consecutive blocks in the order given: in
    time for one run, or copy after copy for many independent ones, which
    is the safer choice when barriers are crossed rarely;
    P in each bin is the mean of the blocks' densities, its error their
    standard error, and the error of F is k_BT δP/P, the first-order
    change of the logarithm. Bins never visited have F = ∞. Returns the
    middles of the bins along each coordinate, F and its error.
    """
    x = np.asarray(values, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    edges = np.histogramdd(x, bins=bins, range=value_range)[1]
    blocks = np.array_split(x, n_blocks)
    density = np.array(
        [np.histogramdd(b, bins=edges, density=True)[0] for b in blocks]
    )
    p = density.mean(axis=0)
    dp = density.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    kt = KB * temperature
    with np.errstate(divide="ignore", invalid="ignore"):
        f = -kt * np.log(p)
        df = kt * dp / p
    f -= np.min(f[np.isfinite(f)])
    middles = [0.5 * (e[1:] + e[:-1]) for e in edges]
    return middles, f, df
