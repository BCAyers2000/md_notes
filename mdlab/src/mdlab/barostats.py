"""Barostats: Chapter 14.

Two first-order barostats act once per step, at its start, on the cell,
the positions and, for cell rescaling, the velocities, inside the loop
``run``, which is Chapter 13's velocity Verlet with its thermostat hooks:

- ``Barostat``: does nothing, so the cell is fixed;
- ``Berendsen``: each length scaled by 1 − (κδt/3τ_P)(P₀ − P), the
  first-order form of Berendsen's cube root, from the mean pressure
  (isotropic), from the mean of P_xx and P_yy and from P_zz
  (semi-isotropic), or from each diagonal component (anisotropic)
  (Section 14.6);
- ``StochasticCellRescaling``: ln V moved by the drift of Berendsen's
  barostat and a noise that gives the right volume fluctuations, after
  Bernetti and Bussi (2020), isotropic or semi-isotropic (Section 14.6).

The piston of Martyna, Tobias and Klein has its own integrator,
``run_mtk``: the volume as a coordinate with a mass, isotropic, with
Nosé-Hoover chains on the atoms and on the piston, the Trotter step of
ASE's ``IsotropicMTKNPT`` (Section 14.7).

The model is called with the positions, returns the energy and forces,
stores its virial tensor in ``model.virial`` and takes a new cell through
``model.set_cell``, as ``md.PairModel`` does. Anisotropic and
semi-isotropic coupling scale the lattice vectors, so they mean x, y and
z only for a cell whose vectors lie along those axes.

Units
-----
Masses in amu, positions in Å, time in fs, energies in eV, temperatures
in K, pressures in eV/Å³, compressibilities in Å³/eV.
"""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import exprel

from mdlab.cell import cell_volume
from mdlab.thermostats import SUZUKI_YOSHIDA, NoseHooverChain, Thermostat
from mdlab.thermostats import kinetic as kinetic_energy
from mdlab.units import FORCE_TO_ACCEL, KB
from mdlab.virial import pressure_tensor

COUPLINGS = ("isotropic", "semi-isotropic", "anisotropic")


def _scale_cell(r: NDArray, v: NDArray, h: NDArray, stretch: NDArray,
                velocities: bool) -> tuple[NDArray, NDArray, NDArray]:
    """Lattice vector k scaled by stretch[k], fractional coordinates kept."""
    h_new = h * stretch[None, :]
    r_new = r @ (h_new @ np.linalg.inv(h)).T
    if velocities:
        v = v @ (h @ np.linalg.inv(h_new)).T
    return r_new, v, h_new


class Barostat:
    """No barostat: the cell stays as it is."""

    pressure = 0.0

    def scale(self, r, v, h, tensor, dt, rng):
        """Return the positions, velocities and cell unchanged."""
        return r, v, h


class Berendsen(Barostat):
    """Weak coupling: P relaxes towards P₀ with the time constant τ_P.

    ``compressibility`` κ is an estimate of the system's; with ``couple``
    "isotropic" every length is scaled by one factor, "semi-isotropic"
    scales x and y together and z alone, "anisotropic" each alone.
    """

    def __init__(self, pressure, tau, compressibility, couple="isotropic"):
        if couple not in COUPLINGS:
            raise ValueError(f"couple must be one of {COUPLINGS}")
        self.pressure, self.tau = pressure, tau
        self.compressibility, self.couple = compressibility, couple

    def target(self, tensor: NDArray) -> NDArray:
        """The pressure each length responds to."""
        diagonal = np.diag(tensor)
        if self.couple == "isotropic":
            return np.full(3, diagonal.mean())
        if self.couple == "semi-isotropic":
            plane = 0.5 * (diagonal[0] + diagonal[1])
            return np.array([plane, plane, diagonal[2]])
        return diagonal

    def scale(self, r, v, h, tensor, dt, rng):
        """Scale each length by 1 − (κδt/3τ_P)(P₀ − P)."""
        c = self.compressibility * dt / (3 * self.tau)
        stretch = 1 - c * (self.pressure - self.target(tensor))
        return _scale_cell(r, v, h, stretch, velocities=False)


class StochasticCellRescaling(Barostat):
    """Berendsen's drift on ln V with the noise that samples NPT.

    Isotropic: d ln V = −(κ/τ_P)(P₀ − P) dt + √(2k_BTκ/(Vτ_P)) dW, every
    length scaled by e^{d ln V/3} and every velocity by e^{−d ln V/3}.
    Semi-isotropic: ln A, the area across z, and ln L_z, the height, each
    with their own drift and noise, Eq. 9 of Bernetti and Bussi (2020).
    Integrated by Euler's rule, one change of ln V per time step.
    """

    def __init__(self, pressure, temperature, tau, compressibility,
                 couple="isotropic", kb=KB):
        if couple not in COUPLINGS[:2]:
            raise ValueError("couple must be isotropic or semi-isotropic")
        self.pressure, self.temperature, self.tau = pressure, temperature, tau
        self.compressibility, self.couple = compressibility, couple
        self.kb = kb

    def scale(self, r, v, h, tensor, dt, rng):
        """Scale the lengths, and the velocities by the inverse factors."""
        kappa, tau, p0 = self.compressibility, self.tau, self.pressure
        kt_v = self.kb * self.temperature / cell_volume(h)
        diagonal = np.diag(tensor)
        if self.couple == "isotropic":
            d_log_v = (-kappa / tau * (p0 - diagonal.mean()) * dt
                       + math.sqrt(2 * kt_v * kappa * dt / tau)
                       * rng.standard_normal())
            stretch = np.full(3, math.exp(d_log_v / 3))
        else:
            plane = 0.5 * (diagonal[0] + diagonal[1])
            d_log_area = (-2 * kappa / (3 * tau) * (p0 - plane) * dt
                          + math.sqrt(4 * kt_v * kappa * dt / (3 * tau))
                          * rng.standard_normal())
            d_log_height = (-kappa / (3 * tau) * (p0 - diagonal[2]) * dt
                            + math.sqrt(2 * kt_v * kappa * dt / (3 * tau))
                            * rng.standard_normal())
            stretch = np.exp([0.5 * d_log_area, 0.5 * d_log_area,
                              d_log_height])
        return _scale_cell(r, v, h, stretch, velocities=True)


def run(
    model: Callable[[NDArray], tuple],
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    dt: float,
    n_steps: int,
    barostat: Barostat | None = None,
    thermostat: Thermostat | None = None,
    rng: np.random.Generator | None = None,
    every: int = 1,
    keep: tuple[str, ...] = (),
) -> dict[str, NDArray]:
    """Velocity Verlet with a thermostat and a first-order barostat.

    At the start of each step the thermostat acts, then the barostat
    scales the cell from the pressure tensor at that moment, the forces
    are recomputed in the new cell, and the step proceeds as in
    ``thermostats.run``. Returns "times", "potential", "kinetic", "heat",
    "work" (the change of K + U + P₀V made by the barostat), "volume",
    "cell" and "pressure" (the instantaneous tensor), and the arrays named
    in ``keep`` ("positions", "velocities"). The effective energy
    K + U + P₀V − heat − work changes only through the integration error.
    """
    m = np.asarray(masses, dtype=float)
    r = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    th = Thermostat() if thermostat is None else thermostat
    baro = Barostat() if barostat is None else barostat
    rng = np.random.default_rng() if rng is None else rng
    accel = FORCE_TO_ACCEL / m[:, None]
    h = np.array(model.cell, dtype=float)
    record: dict[str, list] = {key: [] for key in (
        "times", "potential", "kinetic", "heat", "work", "volume", "cell",
        "pressure", *keep)}
    u, f = model(r)
    heat = work = 0.0

    def act(stage, v):
        nonlocal heat
        k0 = kinetic_energy(m, v)
        v = stage(v, m, dt, rng)
        heat += float(kinetic_energy(m, v) - k0)
        return v

    def enthalpy():
        return u + float(kinetic_energy(m, v)) + baro.pressure * cell_volume(h)

    for step in range(n_steps + 1):
        if step > 0:
            v = act(th.before, v)
            if barostat is not None:
                before = enthalpy()
                tensor = pressure_tensor(m, v, model.virial, cell_volume(h))
                r, v, h = baro.scale(r, v, h, tensor, dt, rng)
                model.set_cell(h)
                u, f = model(r)
                work += enthalpy() - before
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
            record["kinetic"].append(float(kinetic_energy(m, v)))
            record["heat"].append(heat)
            record["work"].append(work)
            record["volume"].append(cell_volume(h))
            record["cell"].append(h.copy())
            record["pressure"].append(
                pressure_tensor(m, v, model.virial, cell_volume(h)))
            if "positions" in keep:
                record["positions"].append(r.copy())
            if "velocities" in keep:
                record["velocities"].append(v.copy())
    return {key: np.array(value) for key, value in record.items()}


class _PistonChain:
    """Nosé-Hoover chain on the piston's momentum p_ε, as ASE's.

    The first link has the mass 9k_BT τ_P² and the others k_BT τ_P²; the
    first is driven by p_ε²/M_ε − k_BT.
    """

    def __init__(self, kt, tau, chain, loops, mass):
        self.kt, self.chain, self.loops, self.mass = kt, chain, loops, mass
        self.r = np.full(chain, kt * tau * tau)
        self.r[0] *= 9
        self.xi, self.p_xi = np.zeros(chain), np.zeros(chain)

    def energy(self) -> float:
        """The chain's part of the conserved energy."""
        return float(np.sum(0.5 * self.p_xi**2 / self.r)
                     + self.kt * np.sum(self.xi))

    def _kick(self, p_eps, j, delta):
        last = j == self.chain - 1
        if not last:
            self.p_xi[j] *= math.exp(-0.25 * delta * self.p_xi[j + 1]
                                     / self.r[j + 1])
        g = (p_eps**2 / self.mass if j == 0
             else self.p_xi[j - 1] ** 2 / self.r[j - 1]) - self.kt
        self.p_xi[j] += 0.5 * delta * g
        if not last:
            self.p_xi[j] *= math.exp(-0.25 * delta * self.p_xi[j + 1]
                                     / self.r[j + 1])

    def half(self, p_eps, dt):
        """Integrate the chain over half a step; return the new p_ε."""
        for _ in range(self.loops):
            for weight in SUZUKI_YOSHIDA:
                delta = weight * 0.5 * dt / self.loops
                for j in reversed(range(self.chain)):
                    self._kick(p_eps, j, delta)
                self.xi += delta * self.p_xi / self.r
                p_eps *= math.exp(-delta * self.p_xi[0] / self.r[0])
                for j in range(self.chain):
                    self._kick(p_eps, j, delta)
        return p_eps


def run_mtk(
    model: Callable[[NDArray], tuple],
    masses: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
    dt: float,
    n_steps: int,
    temperature: float,
    pressure: float,
    tdamp: float,
    pdamp: float,
    tchain: int = 3,
    pchain: int = 3,
    tloop: int = 1,
    ploop: int = 1,
    every: int = 1,
    keep: tuple[str, ...] = (),
) -> dict[str, NDArray]:
    """The isotropic piston of Martyna, Tobias and Klein, with chains.

    The cell is h₀e^ε, V = V₀e^{3ε}; the piston has the mass
    M_ε = (3N + 3)k_BT τ_P² and the momentum p_ε. Each step is the Trotter
    sequence of ASE's ``IsotropicMTKNPT``: half steps of the piston's
    chain, the atoms' chain (Q₁ = 3N k_BT τ_T²), the piston's momentum
    and the atoms' momenta; a whole step of the positions and of ε; then
    the half steps again in reverse. Returns "times", "potential",
    "kinetic", "volume", "pressure" (the instantaneous scalar),
    "p_eps" and "conserved", and the arrays named in ``keep``.
    """
    m = np.asarray(masses, dtype=float)
    r = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    n = len(m)
    kt = KB * temperature
    accel = FORCE_TO_ACCEL / m[:, None]
    h0 = np.array(model.cell, dtype=float)
    v0 = cell_volume(h0)
    piston_mass = (3 * n + 3) * kt * pdamp**2
    atoms_chain = NoseHooverChain(temperature, tdamp, 3 * n, chain=tchain,
                                  loops=tloop)
    piston_chain = _PistonChain(kt, pdamp, pchain, ploop, piston_mass)
    eps = p_eps = 0.0
    record: dict[str, list] = {key: [] for key in (
        "times", "potential", "kinetic", "volume", "pressure", "p_eps",
        "conserved", *keep)}

    def volume():
        return v0 * math.exp(3 * eps)

    def current_pressure():
        tensor = pressure_tensor(m, v, model.virial, volume())
        return float(np.trace(tensor) / 3)

    def kick_piston(delta):
        nonlocal p_eps
        twice_k = 2 * float(kinetic_energy(m, v))
        p_eps += delta * (3 * volume() * (current_pressure() - pressure)
                          + twice_k / n)

    def kick_atoms(delta):
        nonlocal v
        x = (1 + 1 / n) * p_eps * delta / piston_mass
        v = v * math.exp(-x) + delta * accel * f * exprel(-x)

    model.set_cell(h0)
    u, f = model(r)
    atoms_chain.before(v, m, 0.0, None)  # create the chain's variables
    for step in range(n_steps + 1):
        if step > 0:
            p_eps = piston_chain.half(p_eps, dt)
            v = atoms_chain.before(v, m, dt, None)
            kick_piston(0.5 * dt)
            kick_atoms(0.5 * dt)
            x = dt * p_eps / piston_mass
            r = r * math.exp(x) + v * dt * exprel(x)
            eps += dt * p_eps / piston_mass
            model.set_cell(h0 * math.exp(eps))
            u, f = model(r)
            kick_atoms(0.5 * dt)
            kick_piston(0.5 * dt)
            v = atoms_chain.after(v, m, dt, None)
            p_eps = piston_chain.half(p_eps, dt)
        if step % every == 0:
            k = float(kinetic_energy(m, v))
            record["times"].append(step * dt)
            record["potential"].append(u)
            record["kinetic"].append(k)
            record["volume"].append(volume())
            record["pressure"].append(current_pressure())
            record["p_eps"].append(p_eps)
            record["conserved"].append(
                u + k + float(atoms_chain.energy()) + piston_chain.energy()
                + 0.5 * p_eps**2 / piston_mass + pressure * volume())
            if "positions" in keep:
                record["positions"].append(r.copy())
            if "velocities" in keep:
                record["velocities"].append(v.copy())
    return {key: np.array(value) for key, value in record.items()}
