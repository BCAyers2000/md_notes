"""Thermostats: Chapter 13.

One loop, ``run``, integrates velocity Verlet written as B·A·A·B, a half
kick, two half drifts and a half kick, and lets a thermostat act on the
velocities before the step, between the two half drifts, and after it:

- ``Thermostat``: does nothing, so the run keeps its energy;
- ``Rescale``: the velocities scaled to the target temperature before
  every step (Section 13.2);
- ``Berendsen``: weak coupling, λ² = 1 + (δt/τ)(T₀/T − 1) (Section 13.2);
- ``Andersen``: each atom's velocity redrawn from the Maxwell-Boltzmann
  distribution with probability 1 − e^{−νδt} after each step (Section
  13.3);
- ``Langevin``: friction γ and noise between the half drifts, the O of
  BAOAB (Section 13.4);
- ``NoseHooverChain``: a chain of M thermostat variables, integrated by
  Trotter half steps before and after the step with the weights of
  Suzuki and Yoshida; M = 1 is Nosé-Hoover (Section 13.5);
- ``CSVR``: stochastic velocity rescaling, after Bussi, Donadio and
  Parrinello, before each step (Section 13.6).

Every run records the heat, the kinetic energy the thermostat has added
since the start; the effective energy E − heat is conserved by the exact
dynamics of every method, and its drift measures the integration error.

Shapes
------
Positions and velocities are (N, d), or (R, N, d) for R independent
replicas, each with its own thermostat; d is 1, 2 or 3. The model returns
the potential energy, () or (R,), and the forces, shaped as the positions.

Units
-----
Masses in amu, positions in Å, time in fs, energies in eV, temperatures
in K; or reduced units with ``kb = mv2_to_energy = force_to_accel = 1``.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.units import FORCE_TO_ACCEL, KB, MV2_TO_EV

#: Weights of the fourth-order Suzuki-Yoshida composition.
SUZUKI_YOSHIDA = (
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
)


def kinetic(
    m: NDArray, v: NDArray, mv2_to_energy: float = MV2_TO_EV
) -> NDArray:
    """½ Σ m v² over the atoms and directions of each replica."""
    return 0.5 * mv2_to_energy * np.einsum("i,...ix,...ix->...", m, v, v)


class Thermostat:
    """No thermostat: velocity Verlet at fixed energy.

    Subclasses act on the velocities ``v`` (shape (..., N, d)) of atoms of
    masses ``m`` (N,) over a step ``dt``, and return the new velocities.
    """

    kb = KB
    mv2_to_energy = MV2_TO_EV

    def before(self, v, m, dt, rng):
        """Act before the first half kick; here, nothing."""
        return v

    def middle(self, v, m, dt, rng):
        """Act between the two half drifts; here, nothing."""
        return v

    def after(self, v, m, dt, rng):
        """Act after the second half kick; here, nothing."""
        return v

    def _kt(self, temperature):
        return self.kb * temperature


class Rescale(Thermostat):
    """Scale the velocities to the target kinetic energy before each step."""

    def __init__(self, temperature, n_free, kb=KB, mv2_to_energy=MV2_TO_EV):
        self.temperature, self.n_free = temperature, n_free
        self.kb, self.mv2_to_energy = kb, mv2_to_energy

    def before(self, v, m, dt, rng):
        """Scale every velocity by one factor to the target K."""
        target = 0.5 * self.n_free * self._kt(self.temperature)
        k = kinetic(m, v, self.mv2_to_energy)
        return v * np.sqrt(target / k)[..., None, None]


class Berendsen(Thermostat):
    """Weak coupling: T relaxes towards T₀ with the time constant τ."""

    def __init__(self, temperature, tau, n_free, kb=KB,
                 mv2_to_energy=MV2_TO_EV):
        self.temperature, self.tau, self.n_free = temperature, tau, n_free
        self.kb, self.mv2_to_energy = kb, mv2_to_energy

    def before(self, v, m, dt, rng):
        """Scale the velocities by λ for one step of relaxation."""
        k = kinetic(m, v, self.mv2_to_energy)
        t_now = 2 * k / (self.n_free * self.kb)
        scale = np.sqrt(1 + dt / self.tau * (self.temperature / t_now - 1))
        return v * scale[..., None, None]


class Andersen(Thermostat):
    """Redraw each atom's velocity at the rate ν (per unit time)."""

    def __init__(self, temperature, rate, kb=KB, mv2_to_energy=MV2_TO_EV):
        self.temperature, self.rate = temperature, rate
        self.kb, self.mv2_to_energy = kb, mv2_to_energy

    def after(self, v, m, dt, rng):
        """Redraw the velocities of the atoms that collide."""
        chosen = rng.random(v.shape[:-1]) < -math.expm1(-self.rate * dt)
        sigma = np.sqrt(self._kt(self.temperature) / (m * self.mv2_to_energy))
        fresh = rng.standard_normal(v.shape) * sigma[:, None]
        return np.where(chosen[..., None], fresh, v)


class Langevin(Thermostat):
    """Friction γ and matching noise, the exact O step of BAOAB."""

    def __init__(self, temperature, friction, kb=KB, mv2_to_energy=MV2_TO_EV):
        self.temperature, self.friction = temperature, friction
        self.kb, self.mv2_to_energy = kb, mv2_to_energy

    def middle(self, v, m, dt, rng):
        """Damp the velocities and add noise over the step."""
        c = math.exp(-self.friction * dt)
        sigma = np.sqrt((1 - c * c) * self._kt(self.temperature)
                        / (m * self.mv2_to_energy))
        return c * v + rng.standard_normal(v.shape) * sigma[:, None]


class NoseHooverChain(Thermostat):
    """A chain of M thermostat variables (M = 1: Nosé-Hoover).

    The first variable's mass is Q₁ = N_f k_BT τ², the others' k_BT τ²; the
    chain is integrated for half a step before and after each step, each
    half split into ``loops`` pieces and composed with the weights of
    Suzuki and Yoshida (Martyna, Tuckerman, Tobias and Klein 1996).
    """

    def __init__(self, temperature, tau, n_free, chain=3, loops=1, kb=KB,
                 mv2_to_energy=MV2_TO_EV):
        self.temperature, self.tau, self.n_free = temperature, tau, n_free
        self.chain, self.loops = chain, loops
        self.kb, self.mv2_to_energy = kb, mv2_to_energy
        kt = self._kt(temperature)
        self.q = np.full(chain, kt * tau * tau)
        self.q[0] *= n_free
        self.eta = self.p_eta = None

    def energy(self) -> NDArray:
        """The thermostat's part of the conserved extended energy."""
        kt = self._kt(self.temperature)
        return (self.n_free * kt * self.eta[..., 0]
                + kt * self.eta[..., 1:].sum(-1)
                + (0.5 * self.p_eta**2 / self.q).sum(-1))

    def _force(self, v, m, j):
        kt = self._kt(self.temperature)
        if j == 0:
            return 2 * kinetic(m, v, self.mv2_to_energy) - self.n_free * kt
        return self.p_eta[..., j - 1] ** 2 / self.q[j - 1] - kt

    def _kick(self, v, m, j, delta):
        last = j == self.chain - 1
        if not last:
            self.p_eta[..., j] *= np.exp(
                -0.25 * delta * self.p_eta[..., j + 1] / self.q[j + 1])
        self.p_eta[..., j] += 0.5 * delta * self._force(v, m, j)
        if not last:
            self.p_eta[..., j] *= np.exp(
                -0.25 * delta * self.p_eta[..., j + 1] / self.q[j + 1])

    def _half(self, v, m, dt):
        if self.p_eta is None:
            shape = v.shape[:-2] + (self.chain,)
            self.eta, self.p_eta = np.zeros(shape), np.zeros(shape)
        for _ in range(self.loops):
            for weight in SUZUKI_YOSHIDA:
                delta = weight * 0.5 * dt / self.loops
                for j in reversed(range(self.chain)):
                    self._kick(v, m, j, delta)
                self.eta += delta * self.p_eta / self.q
                v = v * np.exp(-delta * self.p_eta[..., 0] / self.q[0])[
                    ..., None, None]
                for j in range(self.chain):
                    self._kick(v, m, j, delta)
        return v

    def before(self, v, m, dt, rng):
        """Integrate the chain over half a step."""
        return self._half(v, m, dt)

    def after(self, v, m, dt, rng):
        """Integrate the chain over half a step."""
        return self._half(v, m, dt)


class CSVR(Thermostat):
    """Canonical sampling through velocity rescaling, before each step.

    K is moved by the exact solution over δt of the stochastic equation
    dK = (K̄ − K) dt/τ + 2 √(K K̄/(N_f τ)) dW, K̄ = ½ N_f k_BT, by one
    factor α for all velocities (Bussi, Donadio and Parrinello 2007).
    """

    def __init__(self, temperature, tau, n_free, stride=1, kb=KB,
                 mv2_to_energy=MV2_TO_EV):
        self.temperature, self.tau, self.n_free = temperature, tau, n_free
        self.stride, self._steps = stride, 0
        self.kb, self.mv2_to_energy = kb, mv2_to_energy

    def before(self, v, m, dt, rng):
        """Scale the velocities by the stochastic factor α.

        With ``stride`` > 1 it acts once every ``stride`` steps, over
        their whole length, as TrajCast's acts once per learned stride.
        """
        self._steps += 1
        if (self._steps - 1) % self.stride:
            return v
        dt = dt * self.stride
        k = kinetic(m, v, self.mv2_to_energy)
        target = 0.5 * self.n_free * self._kt(self.temperature)
        c1 = math.exp(-dt / self.tau)
        c2 = (1 - c1) * target / (k * self.n_free)
        r1 = rng.standard_normal(k.shape)
        rest = (2 * rng.standard_gamma(0.5 * (self.n_free - 1), k.shape)
                if self.n_free > 1 else np.zeros(k.shape))
        alpha2 = c1 + c2 * (rest + r1 * r1) + 2 * r1 * np.sqrt(c1 * c2)
        return v * np.sqrt(alpha2)[..., None, None]


def run(
    model: Callable[[NDArray], tuple],
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    dt: float,
    n_steps: int,
    thermostat: Thermostat | None = None,
    rng: np.random.Generator | None = None,
    every: int = 1,
    keep: tuple[str, ...] = ("positions", "velocities"),
    force_to_accel: float = FORCE_TO_ACCEL,
    observers: Sequence[Callable[[float, NDArray, NDArray], None]] = (),
) -> dict[str, NDArray]:
    """Velocity Verlet with a thermostat, recording every ``every`` steps.

    Returns "times", "potential", "kinetic" and "heat", and the arrays
    named in ``keep``; the effective energy is potential + kinetic − heat.
    Each observer is called at every recorded step with the time, the
    positions and the velocities, as in ``md.run`` (Section 16.4).
    """
    m = np.asarray(masses, dtype=float)
    r = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    th = Thermostat() if thermostat is None else thermostat
    rng = np.random.default_rng() if rng is None else rng
    mv2 = 1.0 if force_to_accel == 1 else MV2_TO_EV
    accel = force_to_accel / m[:, None]
    record: dict[str, list] = {
        key: [] for key in ("times", "potential", "kinetic", "heat", *keep)}
    u, f = model(r)
    heat = np.zeros(v.shape[:-2])

    def act(stage, v):
        nonlocal heat
        k0 = kinetic(m, v, mv2)
        v = stage(v, m, dt, rng)
        heat = heat + kinetic(m, v, mv2) - k0
        return v

    for step in range(n_steps + 1):
        if step > 0:
            v = act(th.before, v)
            v = v + 0.5 * dt * accel * f
            r = r + 0.5 * dt * v
            v = act(th.middle, v)
            r = r + 0.5 * dt * v
            u, f = model(r)
            v = v + 0.5 * dt * accel * f
            v = act(th.after, v)
        if step % every == 0:
            record["times"].append(step * dt)
            record["potential"].append(u)
            record["kinetic"].append(kinetic(m, v, mv2))
            record["heat"].append(heat)
            if "positions" in keep:
                record["positions"].append(r)
            if "velocities" in keep:
                record["velocities"].append(v)
            for observe in observers:
                observe(step * dt, r, v)
    return {key: np.array(value) for key, value in record.items()}
