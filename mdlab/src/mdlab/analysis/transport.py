"""How far atoms wander, and how fast charge and momentum move: Chapter 16.

- ``msd``: the mean squared displacement along each axis, averaged over
  the atoms and over every starting frame, with the centre of mass
  removed (Section 16.5);
- ``diffusion_coefficient``: a sixth (in three dimensions) of the slope
  of the MSD over a window, by least squares;
- ``correlation``: ⟨a(0)·b(t)⟩ averaged over starting frames, summed over
  the last axis, by the fast Fourier transform (Section 16.9);
- ``velocity_autocorrelation``: ⟨v(0)·v(t)⟩ per atom, mass-weighted if
  masses are given (Section 16.6);
- ``running_integral``: the integral of a correlation function from 0 to
  each t, by the trapezium rule, as in a Green-Kubo formula.

Units
-----
Positions in Å, velocities in Å/fs, times in fs, masses in amu.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def msd(
    positions: ArrayLike,
    max_lag: int | None = None,
    masses: ArrayLike | None = None,
    remove_drift: bool = True,
    select: ArrayLike | None = None,
    origin_step: int = 1,
    lags: ArrayLike | None = None,
) -> NDArray:
    """⟨(r(t + k) − r(t))²⟩ along x, y and z, for lags k = 0 … max_lag.

    ``positions`` (frames, N, 3) must be unwrapped, each atom's path
    continuous. With ``remove_drift`` each frame is first measured from
    its centre of mass (weighted by ``masses`` if given), so that a drift
    of the whole system does not count as diffusion. ``select`` picks the
    atoms averaged over, after the centre is removed. The starting frames
    t are every ``origin_step``-th; a step as long as the run keeps the
    first frame alone. Returns an array (max_lag + 1, 3), or one row for
    each of ``lags`` if they are given; the total MSD is its sum over the
    last axis.
    """
    r = np.asarray(positions, dtype=float)
    if remove_drift:
        w = (np.ones(r.shape[1]) if masses is None
             else np.asarray(masses, dtype=float))
        r = r - np.einsum("i,tix->tx", w / w.sum(), r)[:, None, :]
    if select is not None:
        r = r[:, np.asarray(select)]
    n = len(r)
    if lags is None:
        lags = np.arange((n - 1 if max_lag is None else max_lag) + 1)
    out = np.zeros((len(lags), 3))
    for row, k in enumerate(lags):
        if k == 0:
            continue
        start = np.arange(0, n - k, origin_step)
        d = r[start + k] - r[start]
        out[row] = np.mean(d * d, axis=(0, 1))
    return out


def diffusion_coefficient(
    times: ArrayLike, squared: ArrayLike, start: float, stop: float,
    dimensions: int = 3,
) -> float:
    """D = slope/(2d) of the MSD between ``start`` and ``stop``, in Å²/fs.

    ``squared`` is the MSD summed over the d directions that count.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(squared, dtype=float)
    keep = (t >= start) & (t <= stop)
    slope = np.polyfit(t[keep], y[keep], 1)[0]
    return float(slope / (2 * dimensions))


def correlation(
    a: ArrayLike, b: ArrayLike | None = None, max_lag: int | None = None
) -> NDArray:
    """⟨a(t)·b(t + k)⟩ averaged over t, for k = 0 … max_lag.

    ``a`` and ``b`` have time along the first axis; the product is summed
    over the last axis (the directions) and averaged over any axes
    between (the atoms). Each lag averages its n − k products. The sums
    over t are done at once by the fast Fourier transform of each series
    padded with n zeros, so that no product wraps round the end.
    """
    x = np.asarray(a, dtype=float)
    y = x if b is None else np.asarray(b, dtype=float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    n = len(x)
    fx = np.fft.rfft(x, 2 * n, axis=0)
    fy = fx if b is None else np.fft.rfft(y, 2 * n, axis=0)
    sums = np.fft.irfft(np.conj(fx) * fy, 2 * n, axis=0)[:n]
    per_lag = sums.reshape(n, -1, x.shape[-1]).sum(-1).mean(-1)
    c = per_lag / np.arange(n, 0, -1)
    return c if max_lag is None else c[: max_lag + 1]


def velocity_autocorrelation(
    velocities: ArrayLike,
    max_lag: int | None = None,
    masses: ArrayLike | None = None,
) -> NDArray:
    """⟨v(0)·v(t)⟩ per atom, or ⟨m v(0)·v(t)⟩ if ``masses`` are given.

    ``velocities`` is (frames, N, 3); the result is averaged over atoms
    and starting frames.
    """
    v = np.asarray(velocities, dtype=float)
    if masses is None:
        return correlation(v, max_lag=max_lag)
    root = np.sqrt(np.asarray(masses, dtype=float))[None, :, None]
    return correlation(v * root, max_lag=max_lag)


def running_integral(values: ArrayLike, dt: float) -> NDArray:
    """∫₀^t f, by the trapezium rule, at each sample t = 0, dt, 2dt, …."""
    f = np.asarray(values, dtype=float)
    steps = 0.5 * dt * (f[1:] + f[:-1])
    return np.concatenate([[0.0], np.cumsum(steps)])
