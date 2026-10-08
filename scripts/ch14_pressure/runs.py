"""The runs of Chapter 14, cached in data/ch14_pressure/runs.

Run from scripts/ch14_pressure/: python runs.py [--force]

Between walls (Sections 14.1 and 14.2), steps of 5 fs, the push on the
walls and the virial recorded at every step:

- wall_ideal: 200 argon atoms that do not interact, in a cube of 40 Å
  between walls of stiffness 5 eV/Å², 100 ps at fixed energy;
- wall_lj: 256 Lennard-Jones atoms in a cube of 23.26 Å (0.8σ⁻³), under
  CSVR at 135 K (τ 1 ps), 120 ps, the first 20 discarded.

The liquid (Sections 14.5 to 14.7 and 14.9), 256 atoms from the three
starts of Chapter 13 (s0, s1, s2), steps of 10 fs, 100 ps, CSVR at 135 K
with τ_T = 1 ps for the first-order barostats, P₀ = 0.1 GPa unless
stated, κ = 2.5 GPa⁻¹ in the barostat (a rough guess), τ_P = 1 ps;
energies, volume and pressure every 100 fs, positions every ps:

- npt_berendsen, npt_scr (stochastic cell rescaling), npt_mtk (piston
  and chains, τ_T = τ_P = 1 ps), each from three starts, 300 ps;
- npt_mtk_pd0.3, _pd3 and _pd10: pistons with τ_P of 0.3, 3 and 10 ps,
  the first recorded every 20 fs;
- nvt_v0.963, nvt_v0.978, nvt_v0.992, each from three starts: at fixed
  volume, the box scaled to 0.963, 0.978 and 0.992 of Chapter 12's
  volume, bracketing the mean volume at 0.1 GPa, 100 ps, CSVR;
- fault_kinetic_s0: cell rescaling given only the virial's pressure,
  without the kinetic part; fault_kappa_s0: Berendsen with κ entered as
  2.5 × 10⁻⁴, the value in bar⁻¹, where GPa⁻¹ was meant.

The ideal gas (Section 14.5): 20 argon atoms at 135 K and P₀ = 10⁻⁴
eV/Å³, Langevin friction 0.01/fs, τ_P = 0.2 ps, 2 × 10⁵ steps of 10 fs,
the volume every 100 fs: ideal_berendsen, ideal_scr, ideal_mtk (τ_T
0.1 ps).

The layered solid (Section 14.8), steps of 5 fs, 40 K, P₀ = 0, κ =
1 GPa⁻¹, τ_P = 1 ps: layered_relaxed, the sheets alone relaxed under
anisotropic coupling for 30 ps; then layered_{isotropic,
semi-isotropic,anisotropic}: guests placed between the sheets and their
reach σ grown from 1.6 Å, at which each sits in its hollow, to 2.5 Å over
20 pieces of 0.5 ps, then held for 60 more, the cell and the mean pressure
tensor saved per piece.

About 10 minutes on 8 cores.
"""

import sys
from multiprocessing import Pool

import ch14
import numpy as np
from ch14 import (
    DT,
    GPA,
    KAPPA_LIQUID,
    N_ATOMS,
    P_LIQUID,
    RUNS,
    T_LIQUID,
    argon_model,
    liquid_start,
)

from mdlab import (
    barostats,
    cell,
    md,
    potentials,
    statmech,
    thermostats,
    virial,
)

N_FREE = 3 * N_ATOMS - 3


def _save(name, out, **extra):
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **out, **extra)


class Walled:
    """Atoms between soft walls; records the push and virial each call."""

    def __init__(self, length, stiffness, interacting):
        self.length, self.stiffness = length, stiffness
        self.interacting = interacting
        self.pair = ch14.argon_pair()
        self.push, self.virial = [], []

    def __call__(self, r):
        """Energy and forces of the walls and the pairs."""
        u, f, push = virial.harmonic_walls(r, [self.length] * 3,
                                           self.stiffness)
        w = 0.0
        if self.interacting:
            up, fp, w = potentials.pair_energy_forces(r, self.pair)
            u, f = u + up, f + fp
        self.push.append(push)
        self.virial.append(w)
        return u, f


def wall_job(name, n, length, interacting, ps, seed=0):
    rng = np.random.default_rng(seed)
    m = np.full(n, ch14.MASS)
    if interacting:
        r0, _, h, _ = liquid_start(0)
        r = cell.wrap(r0, h) * (length / h[0, 0])
        thermostat = thermostats.CSVR(T_LIQUID, 1000.0, 3 * n)
    else:
        r = rng.uniform(0.5, length - 0.5, (n, 3))
        thermostat = None
    v = statmech.thermal_velocities(m, T_LIQUID, rng, remove_drift=False)
    model = Walled(length, 5.0, interacting)
    n_steps = int(round(ps * 1000 / 5.0))
    out = thermostats.run(model, m, r, v, 5.0, n_steps, thermostat,
                          np.random.default_rng(seed + 1), every=1, keep=())
    _save(name, {"times": out["times"], "kinetic": out["kinetic"],
                 "potential": out["potential"],
                 "push": np.array(model.push),
                 "virial": np.array(model.virial)},
          length=length, stiffness=5.0, n=n)


def _thin(out, every_positions=10):
    out = dict(out)
    if "positions" in out:
        out["positions"] = out["positions"][::every_positions]
    return out


def npt_job(name, kind, seed=0, pressure=P_LIQUID, pdamp=1000.0,
            kappa=KAPPA_LIQUID, kinetic=True, ps=100.0, every=10):
    r, v, h, m = liquid_start(seed)
    n_steps = int(round(ps * 1000 / DT))
    rng = np.random.default_rng(seed + 10)
    if kind == "mtk":
        out = barostats.run_mtk(argon_model(h), m, r, v, DT, n_steps,
                                T_LIQUID, pressure, 1000.0, pdamp,
                                every=every, keep=("positions",))
    else:
        if kind == "berendsen":
            baro = barostats.Berendsen(pressure, pdamp, kappa)
        else:
            baro = barostats.StochasticCellRescaling(
                pressure, T_LIQUID, pdamp, kappa)
        if not kinetic:
            baro = WithoutKinetic(baro, m)
        out = barostats.run(argon_model(h), m, r, v, DT, n_steps, baro,
                            thermostats.CSVR(T_LIQUID, 1000.0, N_FREE), rng,
                            every=10, keep=("positions",))
    _save(name, _thin(out, 100 // every), masses=m,
          pressure_target=pressure)


class WithoutKinetic:
    """A barostat handed the virial's pressure alone, a fault."""

    def __init__(self, inner, masses):
        self.inner, self.masses = inner, masses
        self.pressure = inner.pressure

    def scale(self, r, v, h, tensor, dt, rng):
        """Remove the kinetic part, then let the barostat act."""
        kin = virial.kinetic_tensor(self.masses, v) / cell.cell_volume(h)
        return self.inner.scale(r, v, h, tensor - kin, dt, rng)


def nvt_job(name, scale, seed=0, ps=50.0):
    r, v, h, m = liquid_start(seed)
    h, r = h * scale, r * scale
    model = argon_model(h)
    out = thermostats.run(model, m, r, v, DT, int(round(ps * 1000 / DT)),
                          thermostats.CSVR(T_LIQUID, 1000.0, N_FREE),
                          np.random.default_rng(seed + 20), every=10,
                          keep=("positions",))
    w = []
    for x in out["positions"]:
        model(x)
        w.append(np.trace(model.virial))
    _save(name, {"times": out["times"], "kinetic": out["kinetic"],
                 "potential": out["potential"], "virial": np.array(w)},
          volume=cell.cell_volume(h))


def ideal_job(name, kind, seed=0, n=20, pressure=1e-4, steps=200000):
    rng = np.random.default_rng(seed)
    side = ch14.ideal_side(n, T_LIQUID, pressure)
    m = np.full(n, ch14.MASS)
    r = rng.uniform(0, side, (n, 3))
    v = statmech.thermal_velocities(m, T_LIQUID, rng, remove_drift=False)
    model = md.NoForces(side * np.eye(3))
    if kind == "mtk":
        out = barostats.run_mtk(model, m, r, v, DT, steps, T_LIQUID,
                                pressure, 100.0, 200.0, every=10)
    else:
        baro = (barostats.Berendsen(pressure, 200.0, 1 / pressure)
                if kind == "berendsen" else
                barostats.StochasticCellRescaling(pressure, T_LIQUID, 200.0,
                                                  1 / pressure))
        out = barostats.run(model, m, r, v, DT, steps, baro,
                            thermostats.Langevin(T_LIQUID, 0.01),
                            np.random.default_rng(seed + 1), every=10)
    _save(name, {"times": out["times"], "volume": out["volume"],
                 "kinetic": out["kinetic"]}, n=n, pressure=pressure)


def _layered_cycle(model_for, m, r, v, h, couple, sigmas, rng):
    """Pieces of 100 steps, each with its own guest reach σ."""
    steps, cells, tensors = 100, [], []
    v_start = cell.cell_volume(h)
    for sigma in sigmas:
        out = barostats.run(
            model_for(sigma, h), m, r, v, ch14.DT_LAYERED, steps,
            barostats.Berendsen(0.0, 1000.0, 1.0 * GPA, couple),
            thermostats.CSVR(ch14.T_LAYERED, 200.0, 3 * len(m) - 3), rng,
            every=10, keep=("positions", "velocities"))
        r, v, h = (out["positions"][-1], out["velocities"][-1],
                   out["cell"][-1])
        if cell.cell_volume(h) > 2 * v_start:
            raise RuntimeError("the layered solid came apart")
        cells.append(np.diag(h))
        tensors.append(out["pressure"][1:].mean(0))
    return r, v, h, np.array(cells), np.array(tensors)


def layered_relax_job(name, seed=0):
    r, types, h = ch14.layered_sheets()
    m = np.full(len(r), ch14.MASS)
    rng = np.random.default_rng(seed)
    v = statmech.thermal_velocities(m, ch14.T_LAYERED, rng)

    def model_for(sigma, cell_now):
        return md.PairModel(ch14.layered_table(), cell_now, ch14.R_CUT, 0.5,
                            types=types)

    r, v, h, cells, tensors = _layered_cycle(
        model_for, m, r, v, h, "anisotropic", [None] * 60, rng)
    _save(name, {"positions": r, "velocities": v, "cell": h, "types": types,
                 "cells": cells, "tensors": tensors})


def layered_insert_job(name, couple, seed=1):
    base = np.load(RUNS / "layered_relaxed.npz")
    h = base["cell"]
    guests = ch14.guest_sites(h)
    r = cell.wrap(np.vstack([base["positions"], guests]), h)
    types = np.concatenate([base["types"], np.full(len(guests), 3)])
    m = np.concatenate([np.full(len(base["types"]), ch14.MASS),
                        np.full(len(guests), ch14.GUEST_MASS)])
    rng = np.random.default_rng(seed)
    v = statmech.thermal_velocities(m, ch14.T_LAYERED, rng)
    sigmas = [ch14.SIG_GROW + (ch14.SIG_GUEST - ch14.SIG_GROW)
              * min(1.0, (k + 1) / 20) for k in range(80)]

    def model_for(sigma, cell_now):
        return md.PairModel(ch14.layered_table(sigma), cell_now, ch14.R_CUT,
                            0.5, types=types)

    r, v, h, cells, tensors = _layered_cycle(model_for, m, r, v, h, couple,
                                             sigmas, rng)
    _save(name, {"positions": r, "cell": h, "types": types,
                 "cells": cells, "tensors": tensors,
                 "sigmas": np.array(sigmas)}, start_cell=base["cell"])


def _jobs():
    jobs = [
        ("wall_ideal", wall_job, (200, 40.0, False, 100.0)),
        ("wall_lj", wall_job, (N_ATOMS, 23.255672874803075, True, 120.0)),
    ]
    for kind in ("berendsen", "scr", "mtk"):
        for seed in range(3):
            jobs.append((f"npt_{kind}_s{seed}", npt_job,
                         (kind, seed, P_LIQUID, 1000.0, KAPPA_LIQUID, True,
                          300.0)))
    for pd in (0.3, 3.0, 10.0):
        jobs.append((f"npt_mtk_pd{pd:g}_s0", npt_job,
                     ("mtk", 0, P_LIQUID, pd * 1000, KAPPA_LIQUID, True,
                      100.0, 2 if pd < 1 else 10)))
    for scale in (0.963, 0.978, 0.992):
        for seed in range(3):
            jobs.append((f"nvt_v{scale:.3f}_s{seed}", nvt_job,
                         (scale ** (1 / 3), seed, 100.0)))
    jobs.append(("fault_kinetic_s0", npt_job,
                 ("scr", 0, P_LIQUID, 1000.0, KAPPA_LIQUID, False)))
    jobs.append(("fault_kappa_s0", npt_job,
                 ("berendsen", 0, P_LIQUID, 1000.0, 2.5e-4 * GPA)))
    for kind in ("berendsen", "scr", "mtk"):
        jobs.append((f"ideal_{kind}", ideal_job, (kind,)))
    return jobs


def _work(job):
    name, function, args = job
    function(name, *args)
    return name


if __name__ == "__main__":
    force = "--force" in sys.argv
    todo = [job for job in _jobs()
            if force or not (RUNS / f"{job[0]}.npz").exists()]
    print(f"{len(todo)} runs to compute", flush=True)
    with Pool(8) as pool:
        for name in pool.imap_unordered(_work, todo):
            print("done", name, flush=True)
    if force or not (RUNS / "layered_relaxed.npz").exists():
        layered_relax_job("layered_relaxed")
        print("done layered_relaxed", flush=True)
    couples = [c for c in barostats.COUPLINGS
               if force or not (RUNS / f"layered_{c}.npz").exists()]
    with Pool(3) as pool:
        for name in pool.imap_unordered(
                _work, [(f"layered_{c}", layered_insert_job, (c,))
                        for c in couples]):
            print("done", name, flush=True)
