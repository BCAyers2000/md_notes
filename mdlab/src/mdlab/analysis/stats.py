"""Statistics of a time series: Chapter 15.

A run records a quantity A at equal intervals; these functions say how
well its mean is known and whether the run has settled.

- ``autocorrelation``: the normalised autocorrelation function C(k);
- ``statistical_inefficiency``: g = 1 + 2 Σ (1 − k/n) C(k), the sum
  stopped where C first reaches zero, as in ``pymbar.timeseries``;
- ``standard_error``: the mean and its standard error √(g s²/n);
- ``block_average``: the standard error from block means of growing
  length, with its own error (Flyvbjerg and Petersen 1989);
- ``detect_equilibration``: the start that leaves the largest effective
  number of samples (Chodera 2016);
- ``drift_test``: the least-squares slope and its error with correlated
  residuals;
- ``block_bootstrap``: the spread of any statistic over resampled blocks;
- ``convergence_report``: the criterion of Section 15.9, with its evidence.

Units
-----
Those of the series; ``dt``, the interval between samples, in fs.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Thresholds of the criterion of Section 15.9.
MIN_EFFECTIVE = 100  # twenty blocks of 5g: the error known to a sixth
BLOCK_FACTOR = 5  # blocks of 5g miss about 5% for an exponential memory
DRIFT_LIMIT = 2.0  # a slope within two of its errors is no drift


def autocorrelation(series: ArrayLike, max_lag: int | None = None) -> NDArray:
    """C(k) = ⟨δA_i δA_{i+k}⟩/⟨δA²⟩ for k = 0, 1, …, by the FFT.

    Each lag averages its n − k products, and C(0) = 1.

    >>> c = autocorrelation([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    >>> print(c.round(12))
    [ 1. -1.  1. -1.  1. -1.]
    """
    a = np.asarray(series, dtype=float)
    n = len(a)
    d = a - a.mean()
    f = np.fft.rfft(d, 2 * n)
    products = np.fft.irfft(f * np.conj(f), 2 * n)[:n]
    c = products / np.arange(n, 0, -1) / (products[0] / n)
    return c if max_lag is None else c[: max_lag + 1]


def statistical_inefficiency(series: ArrayLike, mintime: int = 3) -> float:
    """The statistical inefficiency g = 1 + 2 Σ_k (1 − k/n) C(k).

    The sum stops at the first C(k) ≤ 0 beyond ``mintime``. The variance
    of the mean of n correlated samples is g times that of n independent
    ones; g is at least 1.
    """
    a = np.asarray(series, dtype=float)
    n = len(a)
    if np.all(a == a[0]):
        raise ValueError("a constant series has no fluctuations")
    c = autocorrelation(a)[1 : n - 1]
    k = np.arange(1, n - 1)
    stop = np.nonzero((c <= 0) & (k > mintime))[0]
    end = stop[0] if len(stop) else len(c)
    g = 1 + 2 * np.sum(c[:end] * (1 - k[:end] / n))
    return float(max(g, 1.0))


def standard_error(
    series: ArrayLike, g: float | None = None
) -> tuple[float, float, float]:
    """The mean, its standard error √(g s²/n) and g.

    >>> mean, se, g = standard_error([1.0, 2.0, 3.0, 4.0], g=1.0)
    >>> print(mean, round(se, 6))
    2.5 0.645497
    """
    a = np.asarray(series, dtype=float)
    g = statistical_inefficiency(a) if g is None else g
    return float(a.mean()), float(np.sqrt(g * a.var(ddof=1) / len(a))), g


def block_average(series: ArrayLike, min_blocks: int = 4) -> dict:
    """Standard errors from block means, blocks of 1, 2, 4, … samples.

    For each length b the series is cut into n_b = n // b blocks (the
    remainder at its end dropped); the standard error is the spread of the
    block means over √n_b, and its own error that over √(2(n_b − 1)).
    Lengths stop where fewer than ``min_blocks`` blocks remain.
    """
    a = np.asarray(series, dtype=float)
    lengths, errors, error_errors, counts = [], [], [], []
    b = 1
    while len(a) // b >= min_blocks:
        n_b = len(a) // b
        means = a[: n_b * b].reshape(n_b, b).mean(axis=1)
        se = means.std(ddof=1) / np.sqrt(n_b)
        lengths.append(b)
        errors.append(se)
        error_errors.append(se / np.sqrt(2 * (n_b - 1)))
        counts.append(n_b)
        b *= 2
    return {"length": np.array(lengths), "error": np.array(errors),
            "error_error": np.array(error_errors),
            "blocks": np.array(counts)}


def detect_equilibration(
    series: ArrayLike, nskip: int = 1
) -> tuple[int, float, float]:
    """The start t₀ that maximises (n − t₀)/g(t₀), with that g and n_eff.

    Every ``nskip``-th start is tried; g(t₀) is the statistical
    inefficiency of the series from t₀ on.
    """
    a = np.asarray(series, dtype=float)
    n = len(a)
    best = (0, 1.0, 0.0)
    for t0 in range(0, n - 1, nskip):
        rest = a[t0:]
        if np.all(rest == rest[0]):
            g = float(n - t0)
        else:
            g = statistical_inefficiency(rest)
        if (n - t0) / g > best[2]:
            best = (t0, g, (n - t0) / g)
    return best


def drift_test(
    times: ArrayLike, series: ArrayLike, g: float | None = None
) -> tuple[float, float, float]:
    """The least-squares slope, its error and their ratio.

    The error is that for independent residuals, √(s_r²/Σ(t − t̄)²),
    multiplied by √g of the residuals, which the slope's weights, constant
    over a correlation time, inherit as the mean does.
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(series, dtype=float)
    dt_ = t - t.mean()
    slope = np.sum(dt_ * (a - a.mean())) / np.sum(dt_**2)
    residuals = a - a.mean() - slope * dt_
    g = statistical_inefficiency(residuals) if g is None else g
    s2 = np.sum(residuals**2) / (len(a) - 2)
    error = np.sqrt(g * s2 / np.sum(dt_**2))
    return float(slope), float(error), float(slope / error)


def block_bootstrap(
    arrays: Sequence[ArrayLike],
    statistic: Callable[..., float],
    block: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> NDArray:
    """The statistic of ``n_resamples`` resampled sets of the arrays.

    Each array is cut along its first axis into blocks of ``block``
    samples, and as many blocks are drawn from it with replacement; the
    arrays are resampled independently, as runs that share nothing.
    """
    cut = []
    for x in arrays:
        x = np.asarray(x)
        n_b = len(x) // block
        cut.append(x[: n_b * block].reshape(n_b, block, *x.shape[1:]))
    values = np.empty(n_resamples)
    for r in range(n_resamples):
        drawn = [c[rng.integers(0, len(c), len(c))].reshape(-1, *c.shape[2:])
                 for c in cut]
        values[r] = statistic(*drawn)
    return values


@dataclass
class Report:
    """The evidence for and against the convergence of one quantity.

    Times in fs; the error of the block estimate is its own standard
    error; ``checks`` maps each test of the criterion to whether it
    passed.
    """

    start: float
    mean: float
    error: float
    inefficiency: float
    correlation_time: float
    effective: float
    block_error: float | None
    block_error_error: float | None
    block_length: float | None
    slope: float
    slope_error: float
    target: float | None
    checks: dict

    @property
    def converged(self) -> bool:
        """Whether every check passed."""
        return all(self.checks.values())

    def __str__(self) -> str:
        lines = [
            f"mean {self.mean:.6g} ± {self.error:.2g} (one standard error)",
            f"equilibrated from {self.start:g} fs; g = "
            f"{self.inefficiency:.3g}, τ_int = {self.correlation_time:.4g} fs,"
            f" {self.effective:.1f} effective samples",
        ]
        if self.block_error is not None:
            lines.append(
                f"blocks of {self.block_length:g} fs: error "
                f"{self.block_error:.2g} ± {self.block_error_error:.1g}")
        lines.append(f"drift {self.slope:.3g} ± {self.slope_error:.2g} "
                     f"per fs")
        for name, passed in self.checks.items():
            lines.append(f"  {'pass' if passed else 'FAIL'}  {name}")
        lines.append("converged" if self.converged else "not converged")
        return "\n".join(lines)


def convergence_report(
    series: ArrayLike,
    dt: float,
    target: float | None = None,
    nskip: int | None = None,
) -> Report:
    """The criterion of Section 15.9 applied to one recorded quantity.

    Discards the start found by ``detect_equilibration`` (every ``nskip``-th
    start tried, by default about two hundred of them), which must lie in
    the first half of the run, then checks the production part: at least
    ``MIN_EFFECTIVE`` effective samples; the error from blocks of
    ``BLOCK_FACTOR`` g samples within twice its own error of √(g s²/n); a
    slope within ``DRIFT_LIMIT`` of its error; and, if ``target`` is
    given, an error no larger than it.
    """
    a = np.asarray(series, dtype=float)
    nskip = max(1, len(a) // 200) if nskip is None else nskip
    t0, _, _ = detect_equilibration(a, nskip)
    rest = a[t0:]
    mean, error, g = standard_error(rest)
    effective = len(rest) / g
    blocks = block_average(rest, min_blocks=2)
    wanted = BLOCK_FACTOR * g
    usable = np.nonzero((blocks["length"] >= wanted)
                        & (blocks["blocks"] >= 4))[0]
    times = dt * np.arange(len(rest))
    slope, slope_error, ratio = drift_test(times, rest)
    checks = {
        "start in the first half of the run": t0 < len(a) / 2,
        f"effective samples ≥ {MIN_EFFECTIVE}": effective >= MIN_EFFECTIVE,
    }
    if len(usable):
        i = usable[0]
        block_error = float(blocks["error"][i])
        block_error_error = float(blocks["error_error"][i])
        block_length = float(blocks["length"][i] * dt)
        checks["blocks of 5g agree with √(g s²/n)"] = bool(
            abs(block_error - error) <= 2 * block_error_error)
    else:
        block_error = block_error_error = block_length = None
        checks["blocks of 5g agree with √(g s²/n)"] = False
    checks[f"|slope| < {DRIFT_LIMIT:g} errors"] = bool(
        abs(ratio) < DRIFT_LIMIT)
    if target is not None:
        checks["error ≤ target"] = bool(error <= target)
    return Report(t0 * dt, mean, error, g, g * dt / 2, effective,
                  block_error, block_error_error, block_length, slope,
                  slope_error, target, checks)
