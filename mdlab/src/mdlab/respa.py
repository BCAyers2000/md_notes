"""Multiple time steps: Chapter 11.

The forces are split into a fast part, cheap and stiff, and a slow part,
costly and smooth. The reversible reference system propagator algorithm
(RESPA) of Tuckerman, Berne and Martyna (1992) splits the Liouville
operator in the same way as velocity Verlet (Section 9.4), one level
deeper: half a kick with the slow forces, n velocity Verlet steps of
δt/n with the fast forces alone, and half a kick with the slow forces.

- ``respa_step``: one outer step;
- ``run``: the step repeated, recording the energies and counting the
  calls of each part;
- ``outer_step_matrix``: for one coordinate pulled by a fast spring and a
  slow one, the matrix that carries (q, v) through one outer step, whose
  trace decides whether the computed motion stays bounded (Section 11.5).

With n = 1 the step is velocity Verlet with the forces added.

Units
-----
Positions in Å, velocities in Å/fs, masses in amu, energies in eV, times
in fs; or reduced units with ``force_to_accel = 1``.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.units import FORCE_TO_ACCEL, MV2_TO_EV

EnergyForces = Callable[[NDArray], tuple[float, NDArray]]


def respa_step(
    positions: ArrayLike,
    velocities: ArrayLike,
    slow_forces: ArrayLike,
    fast_forces: ArrayLike,
    masses: ArrayLike,
    fast: EnergyForces,
    slow: EnergyForces,
    dt: float,
    n_inner: int,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> tuple[NDArray, NDArray, float, NDArray, float, NDArray]:
    """One outer step δt of RESPA with ``n_inner`` inner steps.

    Parameters
    ----------
    positions, velocities : array_like, shape (N, d)
        In Å and Å/fs.
    slow_forces, fast_forces : array_like, shape (N, d)
        The two parts of the force at ``positions``, in eV/Å.
    masses : array_like, shape (N,)
        In amu.
    fast, slow : callable
        Each returns (energy in eV, forces in eV/Å) of its part.
    dt : float
        The outer step, in fs.
    n_inner : int
        Inner steps per outer step.
    force_to_accel : float
        Converts eV/Å per amu to Å/fs².

    Returns
    -------
    positions, velocities : ndarray
    u_fast : float
    fast_forces : ndarray
    u_slow : float
    slow_forces : ndarray

    Examples
    --------
    With one inner step, the step of a spring split into two halves is
    velocity Verlet's (Section 9.2):

    >>> half = lambda r: (0.25 * float(np.sum(r * r)), -0.5 * r)
    >>> r, v, *_ = respa_step([[1.0]], [[0.0]], [[-0.5]], [[-0.5]], [1.0],
    ...                       half, half, 0.1, 1, force_to_accel=1)
    >>> print(r.round(6).item(), v.round(6).item())
    0.995 -0.09975
    """
    m = np.asarray(masses, dtype=float)[:, None]
    r = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    f_fast = np.asarray(fast_forces, dtype=float)
    h = dt / n_inner
    v += 0.5 * dt * force_to_accel * np.asarray(slow_forces) / m
    u_fast = 0.0
    for _ in range(n_inner):
        v += 0.5 * h * force_to_accel * f_fast / m
        r += h * v
        u_fast, f_fast = fast(r)
        v += 0.5 * h * force_to_accel * f_fast / m
    u_slow, f_slow = slow(r)
    v += 0.5 * dt * force_to_accel * f_slow / m
    return r, v, u_fast, f_fast, u_slow, f_slow


def run(
    fast: EnergyForces,
    slow: EnergyForces,
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    dt: float,
    n_inner: int,
    n_steps: int,
    every: int = 1,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> dict[str, NDArray]:
    """``n_steps`` outer steps of RESPA, recording every ``every``.

    Returns the recorded "times", "positions", "velocities", "potential"
    (fast plus slow) and "kinetic" energies, and the totals "fast_calls"
    and "slow_calls", counting the calls at the start.
    """
    m = np.asarray(masses, dtype=float)
    r = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    to_energy = 1.0 if force_to_accel == 1 else MV2_TO_EV
    keys = ("times", "positions", "velocities", "potential", "kinetic")
    record: dict[str, list] = {key: [] for key in keys}
    u_fast, f_fast = fast(r)
    u_slow, f_slow = slow(r)
    for step in range(n_steps + 1):
        if step > 0:
            r, v, u_fast, f_fast, u_slow, f_slow = respa_step(
                r, v, f_slow, f_fast, m, fast, slow, dt, n_inner,
                force_to_accel,
            )
        if step % every == 0:
            kinetic = 0.5 * to_energy * float(m @ (v * v).sum(1))
            for key, value in zip(
                keys,
                (step * dt, r.copy(), v.copy(), u_fast + u_slow, kinetic),
                strict=True,
            ):
                record[key].append(value)
    out = {key: np.array(value) for key, value in record.items()}
    out["fast_calls"] = np.array(1 + n_steps * n_inner)
    out["slow_calls"] = np.array(1 + n_steps)
    return out


def outer_step_matrix(
    dt: float,
    omega_fast: float,
    omega_slow: float,
    n_inner: int | None = None,
) -> NDArray:
    """The matrix of one outer step for a fast and a slow spring.

    One coordinate q with acceleration −(ω_f² + ω_s²) q: the slow part
    kicks for δt/2 at each end of the step, and in between the fast part
    moves (q, v) either exactly (``n_inner`` None), by the rotation
    q' = q cos θ + (v/ω_f) sin θ, v' = −ω_f q sin θ + v cos θ with
    θ = ω_f δt, or by ``n_inner`` velocity Verlet steps.

    Parameters
    ----------
    dt : float
        The outer step.
    omega_fast, omega_slow : float
        ω_f and ω_s, in radians per unit of ``dt``.
    n_inner : int, optional

    Returns
    -------
    ndarray, shape (2, 2)
        M with (q, v) after the step = M (q, v) before.

    Examples
    --------
    The trace is 2 cos θ − (ω_s² δt/ω_f) sin θ (Section 11.5):

    >>> m = outer_step_matrix(0.3, 2.0, 0.5)
    >>> print(round(np.trace(m), 12),
    ...       round(2 * np.cos(0.6) - 0.25 * 0.3 / 2.0 * np.sin(0.6), 12))
    1.629497137067 1.629497137067
    """

    def kick(omega2: float, h: float) -> NDArray:
        return np.array([[1.0, 0.0], [-omega2 * h, 1.0]])

    if n_inner is None:
        c, s = np.cos(omega_fast * dt), np.sin(omega_fast * dt)
        inner = np.array([[c, s / omega_fast], [-omega_fast * s, c]])
    else:
        h = dt / n_inner
        drift = np.array([[1.0, h], [0.0, 1.0]])
        half = kick(omega_fast**2, h / 2)
        inner = np.linalg.matrix_power(half @ drift @ half, n_inner)
    slow = kick(omega_slow**2, dt / 2)
    return slow @ inner @ slow
