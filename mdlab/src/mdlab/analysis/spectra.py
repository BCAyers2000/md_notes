"""Frequencies in a recorded signal: Chapter 16.

- ``dft``: the discrete Fourier transform X_k = Σ_n x_n e^{−2πikn/N},
  summed directly, N² operations (Section 16.9);
- ``fft``: the same numbers by splitting the sum into its even and odd
  samples again and again, N log₂N operations, for N a power of 2;
- ``density_of_states``: the windowed cosine transform of a correlation
  function, scaled to area 3 per atom;
- ``vdos``: the vibrational density of states, that transform of the
  mass-weighted velocity autocorrelation function (Section 16.10);
- ``overlap_score``: 1 − ∫|f − g| / (∫f + ∫g), which is 1 for equal
  curves and 0 for curves that do not overlap (Schran et al. 2021).

Units
-----
Times in fs; frequencies in 1/fs (multiply by 1000 for THz, divide by
``units.C_CM_PER_FS`` for cm⁻¹).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.analysis.transport import velocity_autocorrelation


def dft(x: ArrayLike) -> NDArray:
    """X_k = Σ_n x_n e^{−2πikn/N} for k = 0 … N − 1, summed directly."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n) @ x


def fft(x: ArrayLike) -> NDArray:
    """The discrete Fourier transform by the radix-2 fast algorithm.

    With E and O the transforms of the even and odd samples, each of
    length N/2, X_k = E_k + e^{−2πik/N} O_k and X_{k+N/2} = E_k −
    e^{−2πik/N} O_k, for k < N/2.
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 1:
        return x
    if n % 2:
        raise ValueError("the length must be a power of 2")
    even, odd = fft(x[0::2]), fft(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + twiddle, even - twiddle])


def density_of_states(
    correlation: ArrayLike, dt: float, window: bool = True
) -> tuple[NDArray, NDArray]:
    """The windowed cosine transform of C(t), scaled to area 3 per atom.

    ``correlation`` holds C at the lags 0, dt, …, T. It is multiplied by
    the window cos²(πt/2T), which falls from 1 at t = 0 to 0 at T, and
    transformed as 2∫₀^T C(t) cos(2πνt) dt. Returns the frequencies ν in
    1/fs, from 0 to the Nyquist frequency 1/(2 dt), and the density.
    """
    c = np.asarray(correlation, dtype=float)
    last = len(c) - 1
    if window:
        c = c * np.cos(0.5 * np.pi * np.arange(last + 1) / last) ** 2
    even = np.concatenate([c, c[-2:0:-1]])  # C at t and at −t
    spectrum = np.fft.rfft(even).real * dt
    nu = np.fft.rfftfreq(len(even), dt)
    return nu, 3 * spectrum / np.trapezoid(spectrum, nu)


def vdos(
    velocities: ArrayLike,
    dt: float,
    max_lag: int,
    masses: ArrayLike | None = None,
    window: bool = True,
) -> tuple[NDArray, NDArray]:
    """The vibrational density of states from velocities every ``dt`` fs.

    The autocorrelation function ⟨m v(0)·v(t)⟩ to ``max_lag`` frames,
    through ``density_of_states``: area 3 per atom, as equipartition
    gives it, on frequencies from 0 to 1/(2 dt) in 1/fs.
    """
    c = velocity_autocorrelation(velocities, max_lag, masses)
    return density_of_states(c, dt, window)


def overlap_score(f: ArrayLike, g: ArrayLike) -> float:
    """1 − Σ|f − g| / (Σf + Σg), f and g at the same equally spaced points."""
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    return float(1 - np.sum(np.abs(f - g)) / (np.sum(f) + np.sum(g)))
