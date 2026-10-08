"""Methods of integrating Newton's equations step by step: Chapter 9.

Each step function advances positions and velocities of N atoms by one
step δt, given a function that returns the forces:

- ``euler_step``: forward Euler, first order (Section 9.1);
- ``velocity_verlet_step``: velocity Verlet, second order, reversible and
  symplectic, one force evaluation per step (Sections 9.2 to 9.5);
- ``position_verlet_step`` and ``leapfrog_step``: the Störmer-Verlet and
  leapfrog forms, which give the same positions (Section 9.2);
- ``rk4_step``: the classical fourth-order Runge-Kutta method, four force
  evaluations per step and not symplectic (Section 9.7);
- ``integrate``: a driver that runs any of them and records the energy.

Units
-----
Positions in Å, velocities in Å/fs, forces in eV/Å, masses in amu, times
in fs, with ``force_to_accel`` = ``units.FORCE_TO_ACCEL`` (the default);
or any consistent units with ``force_to_accel=1``.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.units import FORCE_TO_ACCEL, MV2_TO_EV

ForceFunction = Callable[[NDArray], NDArray]


def _accel(forces: NDArray, masses: NDArray, force_to_accel: float):
    return force_to_accel * forces / masses[:, None]


def euler_step(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: ArrayLike,
    force: ForceFunction,
    dt: float,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> tuple[NDArray, NDArray]:
    """One forward Euler step: rates at the start held for the whole step.

    r' = r + δt v,   v' = v + δt a(r).

    Parameters
    ----------
    positions, velocities : array_like, shape (N, d)
        In Å and Å/fs.
    masses : array_like, shape (N,)
        In amu.
    force : callable
        ``force(positions)`` returns the forces, shape (N, d), in eV/Å.
    dt : float
        The step δt, in fs.
    force_to_accel : float
        Converts eV/Å per amu to Å/fs².

    Returns
    -------
    positions, velocities : ndarray
    """
    r = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    a = _accel(force(r), m, force_to_accel)
    return r + dt * v, v + dt * a


def velocity_verlet_step(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: ArrayLike,
    force: ForceFunction,
    dt: float,
    forces: ArrayLike | None = None,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> tuple[NDArray, NDArray, NDArray]:
    """One velocity Verlet step: half kick, drift, half kick.

    v½ = v + (δt/2) a(r),  r' = r + δt v½,  v' = v½ + (δt/2) a(r').

    Parameters
    ----------
    positions, velocities, masses, force, dt, force_to_accel
        As for ``euler_step``.
    forces : array_like, optional
        The forces at ``positions``, if known from the previous step, so
        that each step evaluates the forces once.

    Returns
    -------
    positions, velocities, forces : ndarray
        The new state and the forces there, to pass to the next step.

    Examples
    --------
    A spring with m = k = 1 (ω = 1) from (1, 0), a step of 0.1:

    >>> r, v, f = velocity_verlet_step([[1.0]], [[0.0]], [1.0],
    ...                                lambda r: -r, 0.1, force_to_accel=1)
    >>> print(r.round(6).item(), v.round(6).item())
    0.995 -0.09975
    """
    r = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    f = force(r) if forces is None else np.asarray(forces, dtype=float)
    v_half = v + 0.5 * dt * _accel(f, m, force_to_accel)
    r_new = r + dt * v_half
    f_new = force(r_new)
    v_new = v_half + 0.5 * dt * _accel(f_new, m, force_to_accel)
    return r_new, v_new, f_new


def position_verlet_step(
    previous: ArrayLike,
    positions: ArrayLike,
    masses: ArrayLike,
    force: ForceFunction,
    dt: float,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> NDArray:
    """One Störmer-Verlet step: r(t + δt) = 2r(t) − r(t − δt) + δt² a(t).

    Parameters
    ----------
    previous, positions : array_like, shape (N, d)
        r(t − δt) and r(t), in Å.
    masses, force, dt, force_to_accel
        As for ``euler_step``.

    Returns
    -------
    ndarray
        r(t + δt).
    """
    r0 = np.asarray(previous, dtype=float)
    r = np.asarray(positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    return 2 * r - r0 + dt * dt * _accel(force(r), m, force_to_accel)


def leapfrog_step(
    positions: ArrayLike,
    half_velocities: ArrayLike,
    masses: ArrayLike,
    force: ForceFunction,
    dt: float,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> tuple[NDArray, NDArray]:
    """One leapfrog step, the velocities half a step out of phase.

    v(t + δt/2) = v(t − δt/2) + δt a(r(t)),  r(t + δt) = r(t) + δt v(t + δt/2).

    Parameters
    ----------
    positions : array_like, shape (N, d)
        r(t), in Å.
    half_velocities : array_like, shape (N, d)
        v(t − δt/2), in Å/fs.
    masses, force, dt, force_to_accel
        As for ``euler_step``.

    Returns
    -------
    positions, half_velocities : ndarray
        r(t + δt) and v(t + δt/2).
    """
    r = np.asarray(positions, dtype=float)
    vh = np.asarray(half_velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    vh_new = vh + dt * _accel(force(r), m, force_to_accel)
    return r + dt * vh_new, vh_new


def rk4_step(
    positions: ArrayLike,
    velocities: ArrayLike,
    masses: ArrayLike,
    force: ForceFunction,
    dt: float,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> tuple[NDArray, NDArray]:
    """One step of the classical fourth-order Runge-Kutta method.

    Four evaluations of the rates (v, a) are combined with the weights
    1/6, 2/6, 2/6, 1/6 (Section 9.7).

    Parameters
    ----------
    positions, velocities, masses, force, dt, force_to_accel
        As for ``euler_step``.

    Returns
    -------
    positions, velocities : ndarray
    """
    r = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)

    def rates(r_, v_):
        return v_, _accel(force(r_), m, force_to_accel)

    k1r, k1v = rates(r, v)
    k2r, k2v = rates(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v)
    k3r, k3v = rates(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v)
    k4r, k4v = rates(r + dt * k3r, v + dt * k3v)
    r_new = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
    v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r_new, v_new


def integrate(
    energy_forces: Callable[[NDArray], tuple[float, NDArray]],
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    dt: float,
    n_steps: int,
    method: str = "velocity_verlet",
    every: int = 1,
    force_to_accel: float = FORCE_TO_ACCEL,
) -> dict[str, NDArray]:
    """Run ``n_steps`` steps of a method and record the motion and energy.

    Parameters
    ----------
    energy_forces : callable
        ``energy_forces(positions)`` returns (U in eV, forces in eV/Å).
    masses : array_like, shape (N,)
        In amu.
    positions, velocities : array_like, shape (N, d)
        The starting state, in Å and Å/fs.
    dt : float
        The step, in fs.
    n_steps : int
        Number of steps.
    method : {"euler", "velocity_verlet", "rk4"}
    every : int
        Record every ``every`` steps, and the start.
    force_to_accel : float
        As for ``euler_step``; with 1, the kinetic energy is ½mv² too.

    Returns
    -------
    dict
        "times" (fs), "positions", "velocities", "potential" and
        "kinetic" energies (eV) at the recorded steps, and "force_calls".
    """
    m = np.asarray(masses, dtype=float)
    r = np.asarray(positions, dtype=float).copy()
    v = np.asarray(velocities, dtype=float).copy()
    ke_factor = 1.0 if force_to_accel == 1 else MV2_TO_EV
    calls = [0]

    def force(x):
        calls[0] += 1
        return energy_forces(x)[1]

    record = {
        "times": [],
        "positions": [],
        "velocities": [],
        "potential": [],
        "kinetic": [],
    }

    def save(step):
        record["times"].append(step * dt)
        record["positions"].append(r.copy())
        record["velocities"].append(v.copy())
        record["potential"].append(energy_forces(r)[0])
        record["kinetic"].append(0.5 * ke_factor * float(m @ (v * v).sum(1)))

    save(0)
    f = None
    for step in range(1, n_steps + 1):
        if method == "velocity_verlet":
            r, v, f = velocity_verlet_step(
                r, v, m, force, dt, f, force_to_accel
            )
        elif method == "euler":
            r, v = euler_step(r, v, m, force, dt, force_to_accel)
        elif method == "rk4":
            r, v = rk4_step(r, v, m, force, dt, force_to_accel)
        else:
            raise ValueError(f"unknown method {method!r}")
        if step % every == 0:
            save(step)
    out = {key: np.array(val) for key, val in record.items()}
    out["force_calls"] = np.array(calls[0])
    return out
