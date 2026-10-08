"""Fourier transforms, the VDOS, the overlap score and free energies."""

import numpy as np
import pytest

from mdlab import units
from mdlab.analysis import landscape, spectra


def test_dft_and_fft_against_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    assert spectra.dft(x) == pytest.approx(np.fft.fft(x), abs=1e-10)
    assert spectra.fft(x) == pytest.approx(np.fft.fft(x), abs=1e-10)
    with pytest.raises(ValueError):
        spectra.fft(np.ones(12))


def test_vdos_of_oscillators_peaks_at_their_frequencies():
    dt, n = 2.0, 20000
    t = dt * np.arange(n)
    nu = np.array([0.004, 0.011])  # 1/fs: 4 and 11 THz
    rng = np.random.default_rng(1)
    phase = rng.uniform(0, 2 * np.pi, (2, 3))
    v = np.cos(2 * np.pi * nu[None, :, None] * t[:, None, None] + phase)
    freq, g = spectra.vdos(v, dt, 2000)
    assert np.trapezoid(g, freq) == pytest.approx(3.0, rel=1e-12)
    low, high = freq < 0.0075, freq >= 0.0075
    assert freq[low][np.argmax(g[low])] == pytest.approx(0.004, abs=3e-4)
    assert freq[high][np.argmax(g[high])] == pytest.approx(0.011, abs=3e-4)


def test_overlap_score_limits():
    x = np.linspace(0, 1, 101)
    f = np.exp(-((x - 0.3) / 0.05) ** 2)
    assert spectra.overlap_score(f, f) == pytest.approx(1.0)
    far = np.exp(-((x - 0.9) / 0.01) ** 2)
    assert spectra.overlap_score(f, far) == pytest.approx(0.0, abs=1e-6)


def test_free_energy_of_a_harmonic_well():
    temperature, k = 300.0, 2.0  # K, eV/Å²
    kt = units.KB * temperature
    rng = np.random.default_rng(2)
    x = rng.normal(0, np.sqrt(kt / k), 400000)
    (middle,), f, df = landscape.free_energy(x, 41, temperature,
                                             value_range=[(-0.2, 0.2)])
    exact = 0.5 * k * middle**2
    inner = abs(middle) < 0.12
    shift = np.mean((f - exact)[inner])
    assert np.all(abs(f - exact - shift)[inner] < 4 * df[inner] + 1e-4)


def test_density_of_states_without_window_is_the_cosine_transform():
    """An exponential memory transforms to a Lorentzian of width 1/2πτ."""
    dt, tau = 1.0, 50.0
    t = dt * np.arange(4001)
    nu, g = spectra.density_of_states(np.exp(-t / tau), dt, window=False)
    lorentz = 2 * tau / (1 + (2 * np.pi * nu * tau) ** 2)
    lorentz *= 3 / np.trapezoid(lorentz, nu)
    assert g[:50] == pytest.approx(lorentz[:50], rel=2e-2)


def test_free_energy_in_two_dimensions():
    """A Gaussian cloud in a harmonic well gives F = ½k(x² + y²)."""
    temperature, k = 300.0, 2.0
    kt = units.KB * temperature
    rng = np.random.default_rng(6)
    xy = rng.normal(0, np.sqrt(kt / k), (400000, 2))
    (mx, my), f, df = landscape.free_energy(
        xy, [21, 21], temperature, value_range=[(-0.15, 0.15)] * 2)
    exact = 0.5 * k * (mx[:, None] ** 2 + my[None, :] ** 2)
    inner = (abs(mx[:, None]) < 0.1) & (abs(my[None, :]) < 0.1)
    shift = np.mean((f - exact)[inner])
    assert np.all(abs(f - exact - shift)[inner] < 4 * df[inner] + 1e-4)
