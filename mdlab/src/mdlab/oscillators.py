"""Oscillations: Chapter 4.

The motions of Chapter 4, in closed form where one exists, and the tools
for the rest:

- ``harmonic``: the spring of Section 2.8, x₀ cos ωt + (v₀/ω) sin ωt;
- ``morse``, ``lennard_jones`` and ``cosine_well``: three anharmonic wells
  (Section 4.5);
- ``period``: the period of a bound motion in any well, from its energy
  (Section 4.5);
- ``damped`` and ``driven_steady_state``: the oscillator with drag, and
  driven by a sinusoidal force (Sections 4.6-4.7);
- ``reduced_mass`` (Section 4.8) and ``normal_modes`` (Section 4.9).

Units
-----
Everything works in any consistent units. Where a force divided by a mass
must become an acceleration, the argument ``force_to_accel`` says how, as
in ``dynamics.solve_newton``: 1.0 in SI units, ``units.FORCE_TO_ACCEL``
for eV, Å, fs and amu.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq


def harmonic(
    t: ArrayLike, x0: float, v0: float, omega: float
) -> tuple[NDArray, NDArray]:
    """Position and velocity of a harmonic oscillator (Section 2.8).

    x(t) = x₀ cos ωt + (v₀/ω) sin ωt, measured from the resting place.

    Parameters
    ----------
    t : array_like
        Times.
    x0, v0 : float
        Position and velocity at t = 0.
    omega : float
        Angular frequency, √(k/m), in radians per unit of time.

    Returns
    -------
    x, v : ndarray
        Position and velocity at each time.

    Examples
    --------
    The block of Section 2.8, 4 cm out and released, a quarter period on:

    >>> x, v = harmonic(np.pi / 20, 0.04, 0.0, 10.0)
    >>> print(f"{float(x):.6f} m, {float(v):.4f} m/s")
    0.000000 m, -0.4000 m/s
    """
    t = np.asarray(t, dtype=float)
    phase = omega * t
    x = x0 * np.cos(phase) + v0 / omega * np.sin(phase)
    v = -x0 * omega * np.sin(phase) + v0 * np.cos(phase)
    return x, v


def damped(
    t: ArrayLike, x0: float, v0: float, omega0: float, gamma: float
) -> tuple[NDArray, NDArray]:
    """Position and velocity of a damped oscillator (Section 4.6).

    The equation of motion d²x/dt² = −ω₀² x − γ dx/dt is solved by
    x = e^(−γt/2) y, where y obeys d²y/dt² = −(ω₀² − γ²/4) y: a sinusoid
    when ω₀ > γ/2, a straight line when ω₀ = γ/2, and a sum of growing
    and shrinking exponentials when ω₀ < γ/2.

    Parameters
    ----------
    t : array_like
        Times.
    x0, v0 : float
        Position and velocity at t = 0.
    omega0 : float
        Undamped angular frequency, √(k/m).
    gamma : float
        Drag rate b/m.

    Returns
    -------
    x, v : ndarray
        Position and velocity at each time.
    """
    t = np.asarray(t, dtype=float)
    y0, w0 = x0, v0 + 0.5 * gamma * x0  # y and dy/dt at t = 0
    rate = omega0**2 - 0.25 * gamma**2
    if rate > 0.0:  # under-damped: y is a sinusoid
        wd = np.sqrt(rate)
        y = y0 * np.cos(wd * t) + w0 / wd * np.sin(wd * t)
        dy = -y0 * wd * np.sin(wd * t) + w0 * np.cos(wd * t)
    elif rate < 0.0:  # over-damped: cosh and sinh in place of cos and sin
        kappa = np.sqrt(-rate)
        y = y0 * np.cosh(kappa * t) + w0 / kappa * np.sinh(kappa * t)
        dy = y0 * kappa * np.sinh(kappa * t) + w0 * np.cosh(kappa * t)
    else:  # critical: y is a straight line
        y = y0 + w0 * t
        dy = np.full_like(t, w0)
    decay = np.exp(-0.5 * gamma * t)
    return decay * y, decay * (dy - 0.5 * gamma * y)


def driven_steady_state(
    omega: ArrayLike, omega0: float, gamma: float, force_per_mass: float
) -> tuple[NDArray, NDArray]:
    """Amplitude and phase lag of a driven, damped oscillator (Section 4.7).

    For d²x/dt² = −ω₀² x − γ dx/dt + f cos ωt the motion settles to
    x = A cos(ωt − δ) with

        A = f / √((ω₀² − ω²)² + γ² ω²),   tan δ = γω / (ω₀² − ω²).

    Parameters
    ----------
    omega : array_like
        Driving angular frequencies.
    omega0 : float
        Undamped angular frequency.
    gamma : float
        Drag rate b/m.
    force_per_mass : float
        f, the amplitude of the driving force divided by the mass.

    Returns
    -------
    amplitude, lag : ndarray
        A, and δ between 0 and π.
    """
    omega = np.asarray(omega, dtype=float)
    detuning = omega0**2 - omega**2
    amplitude = force_per_mass / np.hypot(detuning, gamma * omega)
    lag = np.arctan2(gamma * omega, detuning)
    return amplitude, lag


def morse(
    r: ArrayLike, depth: float, width: float, r0: float
) -> tuple[NDArray, NDArray]:
    """The Morse well, U = D (1 − e^(−a(r − r₀)))², and its force.

    Parameters
    ----------
    r : array_like
        Distances.
    depth : float
        D, the energy needed to pull the pair apart from r₀.
    width : float
        a, an inverse length: the larger a, the narrower the well.
    r0 : float
        The distance of the minimum.

    Returns
    -------
    energy, force : ndarray
        U, and F = −dU/dr.
    """
    r = np.asarray(r, dtype=float)
    e = np.exp(-width * (r - r0))
    energy = depth * (1.0 - e) ** 2
    force = -2.0 * depth * width * e * (1.0 - e)
    return energy, force


def lennard_jones(
    r: ArrayLike, epsilon: float, sigma: float
) -> tuple[NDArray, NDArray]:
    """The Lennard-Jones well, U = 4ε((σ/r)¹² − (σ/r)⁶), and its force.

    Parameters
    ----------
    r : array_like
        Distances.
    epsilon : float
        ε, the depth of the well.
    sigma : float
        σ, the distance at which U = 0.

    Returns
    -------
    energy, force : ndarray
        U, and F = −dU/dr.
    """
    r = np.asarray(r, dtype=float)
    s6 = (sigma / r) ** 6
    energy = 4.0 * epsilon * (s6**2 - s6)
    force = 24.0 * epsilon / r * (2.0 * s6**2 - s6)
    return energy, force


def cosine_well(
    x: ArrayLike, depth: float, wavelength: float
) -> tuple[NDArray, NDArray]:
    """The cosine well, U = (D/2)(1 − cos 2πx/Λ), and its force.

    The pendulum, with x = Lθ the distance along its arc and Λ = 2πL, and
    the hop coordinate of the model surface of Section 3.13, with Λ the
    spacing of the hollows, are both of this form (Section 4.5).

    Parameters
    ----------
    x : array_like
        Positions.
    depth : float
        D, the height of the barrier above the bottom of the well.
    wavelength : float
        Λ, the distance from one well to the next.

    Returns
    -------
    energy, force : ndarray
        U, and F = −dU/dx.
    """
    x = np.asarray(x, dtype=float)
    q = 2.0 * np.pi / wavelength
    energy = 0.5 * depth * (1.0 - np.cos(q * x))
    force = -0.5 * depth * q * np.sin(q * x)
    return energy, force


def turning_point(
    potential: Callable[[float], float],
    energy: float,
    start: float,
    step: float,
    max_steps: int = 10_000,
) -> float:
    """Where U first rises to E, walking out from ``start`` by ``step``.

    Parameters
    ----------
    potential : callable
        ``potential(x)`` returns U(x).
    energy : float
        E, above U(start).
    start : float
        A point inside the allowed region, usually the bottom of the well.
    step : float
        The stride of the walk; its sign sets the direction.
    max_steps : int
        How far to walk before giving up.

    Returns
    -------
    float
        The turning point, found to machine precision by bisection.

    Raises
    ------
    ValueError
        If U does not reach E within ``max_steps`` strides.
    """
    a = start
    for _ in range(max_steps):
        b = a + step
        if potential(b) >= energy:
            return brentq(lambda x: potential(x) - energy, a, b, xtol=1e-14)
        a = b
    raise ValueError("the well does not rise to this energy")


def period(
    potential: Callable[[float], float],
    energy: float,
    mass: float,
    bottom: float,
    step: float,
    *,
    force_to_accel: float,
    nodes: int = 200,
) -> float:
    """Period of a bound motion from its energy (Section 4.5).

    The time to cross the well is ∫ dx / v with v = √(2(E − U)/m), and the
    period is twice that. Writing x = c − η cos ϑ, with c and η the centre
    and half-width of the allowed region, cancels the 1/√ spike of the
    integrand at the turning points, and Gauss-Legendre quadrature then
    converges fast.

    Parameters
    ----------
    potential : callable
        ``potential(x)`` returns U(x) for a single position.
    energy : float
        E, between the bottom of the well and its rim.
    mass : float
        m.
    bottom : float
        A position inside the well, usually its minimum.
    step : float
        Positive stride for finding the turning points either side.
    force_to_accel : float
        Factor turning (energy / mass) into (length / time)²: 1.0 in SI
        units, ``units.FORCE_TO_ACCEL`` for eV, amu, Å and fs.
    nodes : int
        Number of quadrature points.

    Returns
    -------
    float
        The period, in the time unit of ``force_to_accel``.

    Examples
    --------
    A spring of k = 50 N/m with 0.5 kg has period 2π/10 at any energy:

    >>> def spring(x):
    ...     return 25.0 * x**2
    >>> t = period(spring, 0.04, 0.5, 0.0, 0.01, force_to_accel=1.0)
    >>> print(f"{t:.6f}")
    0.628319
    """
    left = turning_point(potential, energy, bottom, -step)
    right = turning_point(potential, energy, bottom, step)
    centre, half = 0.5 * (left + right), 0.5 * (right - left)
    s, w = np.polynomial.legendre.leggauss(nodes)
    theta = 0.5 * np.pi * (s + 1.0)  # Gauss-Legendre nodes moved to (0, π)
    x = centre - half * np.cos(theta)
    gap = np.array([energy - potential(xi) for xi in x])
    speed = np.sqrt(2.0 * np.clip(gap, 0.0, None) * force_to_accel / mass)
    crossing = 0.5 * np.pi * np.sum(w * half * np.sin(theta) / speed)
    return float(2.0 * crossing)


def reduced_mass(m1: float, m2: float) -> float:
    """m_r = m₁ m₂ / (m₁ + m₂), the mass of the relative motion (Section 4.8).

    Examples
    --------
    >>> reduced_mass(1.0, 3.0)
    0.75
    """
    return m1 * m2 / (m1 + m2)


def normal_modes(
    hessian: ArrayLike, masses: ArrayLike, *, force_to_accel: float
) -> tuple[NDArray, NDArray]:
    """Squared angular frequencies and patterns of the normal modes.

    Near a minimum the equations of motion are m_a d²q_a/dt² =
    −Σ_b Φ_ab q_b, with Φ the Hessian, the matrix of second derivatives of
    U. A normal mode q_a = u_a cos(ωt + φ) needs Φ u = ω² M u; writing
    w_a = √m_a u_a turns this into the eigenvalue problem of the
    mass-weighted Hessian Φ_ab / √(m_a m_b) (Section 4.9).

    Parameters
    ----------
    hessian : array_like, shape (n, n)
        Second derivatives of U, symmetric.
    masses : array_like, shape (n,)
        The mass belonging to each coordinate (repeat an atom's mass for
        its x, y and z).
    force_to_accel : float
        1.0 in SI units, ``units.FORCE_TO_ACCEL`` for eV, Å, fs and amu.

    Returns
    -------
    omega_squared : ndarray, shape (n,)
        ω² for each mode, in increasing order; zero for a free
        translation.
    modes : ndarray, shape (n, n)
        Column k is the displacement pattern u of mode k.

    Examples
    --------
    Two 1 kg carts between walls, three springs of 1 N/m:

    >>> w2, _ = normal_modes([[2.0, -1.0], [-1.0, 2.0]], [1.0, 1.0],
    ...                      force_to_accel=1.0)
    >>> w2.round(6)
    array([1., 3.])
    """
    hessian = np.asarray(hessian, dtype=float)
    root_m = np.sqrt(np.asarray(masses, dtype=float))
    weighted = hessian / np.outer(root_m, root_m)  # Φ_ab / √(m_a m_b)
    omega_squared, vectors = np.linalg.eigh(weighted)
    modes = vectors / root_m[:, np.newaxis]  # w_a / √m_a back to u_a
    return omega_squared * force_to_accel, modes
