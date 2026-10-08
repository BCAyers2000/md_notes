"""The runs of Chapter 15, cached in data/ch15_convergence/runs.

Run from scripts/ch15_convergence/: python runs.py [--force] [name ...]

Liquid argon, 256 atoms at 0.8σ⁻³ unless stated, steps of 10 fs, CSVR at
135 K with τ_T = 1 ps unless stated; energies, the pressure and its shear
components at every step, positions (float32) every 500 fs:

- long_csvr: 2 ns from the first start of Chapter 13;
- long_nve: 2 ns at fixed energy, after 50 ps of CSVR from the second
  start with the velocities then scaled so that K + U is the mean of
  those 50 ps (so that the run sits near 135 K);
- melt_s0 to melt_s7: 100 ps from the face-centred cubic lattice at the
  liquid's density, velocities drawn at 135 K with seeds 0 to 7;
- step_<δt>: 1 ns at fixed energy from the third start, δt = 5, 10, 20,
  30, 35 and 40 fs, positions every ps;
- ensemble_<thermostat>_<T>: CSVR (τ_T = 1 ps) and Berendsen
  (τ_T = 0.1 ps) at 130 and 140 K, 1 ns each from the first start;
- size_<N>: 256, 500, 864, 1372 and 2048 atoms prepared as in Chapter 12
  (melted at 270 K, then four rounds of velocities drawn at 135 K), then
  1 ns, energies every step and positions every ps; size_<N>_b the same
  from another seed;
- protocol: 256 atoms placed at random in the liquid's cell, steepest
  descent, 50 ps of CSVR at fixed volume, then 1 ns of stochastic cell
  rescaling at 0.1 GPa (τ_P = 1 ps, κ = 2.5 GPa⁻¹);
- ase_bussi: ASE's Bussi thermostat and Lennard-Jones calculator (cut and
  shifted at 8.5 Å) on the first start, 20 ps, energies at every step.

About 40 minutes on 10 cores, the largest box the longest.
"""

import math
import sys
from multiprocessing import Pool

import numpy as np
from ch15 import (
    DT,
    KAPPA_LIQUID,
    N_FREE,
    P_LIQUID,
    RUNS,
    T_LIQUID,
    TAU_T,
    argon_model,
    ch12,
    chunked,
    liquid_start,
)

from mdlab import barostats, md, statmech, thermostats, units


def _save(name, out, **extra):
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **out, **extra)


def _csvr(temperature=T_LIQUID, n_free=N_FREE):
    return thermostats.CSVR(temperature, TAU_T, n_free)


def long_job(name, kind, seed, ps=2000.0):
    r, v, h, m = liquid_start(seed)
    rng = np.random.default_rng(seed + 100)
    model = argon_model(h)
    if kind == "nve":
        prep = chunked(model, m, r, v, 100, 50, _csvr(), rng=rng)
        late = prep["times"] >= 10000
        target = np.mean(prep["potential"][late] + prep["kinetic"][late])
        r = prep["positions"][-1].astype(float)
        v = prep["final_velocities"]
        u, _ = model(r)
        k = statmech.kinetic_energy(m, v)
        v = v * math.sqrt((target - u) / k)
        thermostat = None
    else:
        thermostat = _csvr()
    n_chunks = int(round(ps * 1000 / (50 * DT)))
    out = chunked(model, m, r, v, n_chunks, 50, thermostat, rng=rng)
    _save(name, out, cell=h, masses=m)


def melt_job(name, seed, ps=100.0):
    r, h = ch12.fcc(4, ch12.A_LIQUID)
    m = np.full(len(r), ch12.MASS)
    rng = np.random.default_rng(seed)
    v = statmech.thermal_velocities(m, T_LIQUID, rng)
    n_chunks = int(round(ps * 1000 / (50 * DT)))
    out = chunked(argon_model(h), m, r, v, n_chunks, 50, _csvr(), rng=rng)
    _save(name, out, cell=h, masses=m)


def step_job(name, dt, ps=1000.0):
    r, v, h, m = liquid_start(2)
    chunk = int(round(1000.0 / dt))
    n_chunks = int(round(ps))
    out = chunked(argon_model(h), m, r, v, n_chunks, chunk, None, dt=dt)
    _save(name, out, cell=h, masses=m, dt=dt)


def ensemble_job(name, kind, temperature, ps=1000.0):
    r, v, h, m = liquid_start(0)
    rng = np.random.default_rng(int(temperature))
    if kind == "csvr":
        thermostat = _csvr(temperature)
    else:
        thermostat = thermostats.Berendsen(temperature, 100.0, N_FREE)
    n_chunks = int(round(ps * 1000 / (50 * DT)))
    out = chunked(argon_model(h), m, r, v, n_chunks, 50, thermostat,
                  rng=rng)
    _save(name, out, cell=h, masses=m, temperature=temperature)


def size_job(name, n_atoms, ps=1000.0, seed=None):
    rng = np.random.default_rng(n_atoms if seed is None else seed)
    cells = round((n_atoms / 4) ** (1 / 3))  # n_atoms = 4 cells³
    r, v, h, m, model = ch12.liquid(cells, T_LIQUID, rng)
    n_free = statmech.degrees_of_freedom(len(m))
    n_chunks = int(round(ps))
    out = chunked(model, m, r, v, n_chunks, 100,
                  _csvr(n_free=n_free), rng=rng)
    _save(name, out, cell=h, masses=m)


def protocol_job(name, seed=3):
    rng = np.random.default_rng(seed)
    _, h = ch12.fcc(4, ch12.A_LIQUID)
    m = np.full(256, ch12.MASS)
    r = rng.uniform(0, h[0, 0], (256, 3))
    model = argon_model(h)
    u0, _ = model(r)
    r, energies, largest = md.minimise(model, r, force_tolerance=0.05)
    v = statmech.thermal_velocities(m, T_LIQUID, rng)
    nvt = chunked(model, m, r, v, 100, 50, _csvr(), rng=rng)
    r = nvt["positions"][-1].astype(float)
    v = nvt["final_velocities"]
    barostat = barostats.StochasticCellRescaling(
        P_LIQUID, T_LIQUID, 1000.0, KAPPA_LIQUID)
    npt = chunked(model, m, r, v, 2000, 50, _csvr(), barostat, rng=rng)
    out = {"min_energy": energies, "min_force": largest, "start_energy": u0}
    out.update({f"nvt_{k}": x for k, x in nvt.items()})
    out.update({f"npt_{k}": x for k, x in npt.items()})
    _save(name, out, cell=h, masses=m)


def ase_job(name, ps=20.0):
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.md.bussi import Bussi

    r, v, h, m = liquid_start(0)
    fs = 1 / math.sqrt(units.FORCE_TO_ACCEL)  # one ASE time unit in fs
    atoms = Atoms(f"Ar{len(r)}", positions=r, cell=h.T, pbc=True, masses=m)
    atoms.set_velocities(v * fs)  # Å per ASE time unit
    atoms.calc = LennardJones(sigma=ch12.SIG, epsilon=ch12.EPS, rc=8.5,
                              smooth=False)
    dyn = Bussi(atoms, DT / fs, temperature_K=T_LIQUID, taut=TAU_T / fs,
                rng=np.random.default_rng(1))
    record = {"potential": [], "kinetic": []}

    def note():
        record["potential"].append(atoms.get_potential_energy())
        record["kinetic"].append(atoms.get_kinetic_energy())

    dyn.attach(note, interval=1)
    dyn.run(int(round(ps * 1000 / DT)))
    _save(name, {k: np.array(x) for k, x in record.items()},
          dt=DT)


def _jobs():
    jobs = [("size_2048", size_job, (2048,)),
            ("size_1372", size_job, (1372,)),
            ("size_864", size_job, (864,)),
            ("long_csvr", long_job, ("csvr", 0)),
            ("long_nve", long_job, ("nve", 1))]
    jobs += [(f"step_{dt:g}", step_job, (dt,))
             for dt in (5.0, 10.0, 20.0, 30.0, 35.0, 40.0)]
    jobs += [(f"ensemble_{kind}_{t:g}", ensemble_job, (kind, t))
             for kind in ("csvr", "berendsen") for t in (130.0, 140.0)]
    jobs += [("size_500", size_job, (500,)), ("size_256", size_job, (256,))]
    jobs += [(f"size_{n}_b", size_job, (n, 1000.0, n + 1))
             for n in (2048, 1372, 864, 500, 256)]
    jobs += [(f"melt_s{s}", melt_job, (s,)) for s in range(8)]
    jobs += [("protocol", protocol_job, ()), ("ase_bussi", ase_job, ())]
    return jobs


def _work(job):
    name, function, args = job
    function(name, *args)
    return name


if __name__ == "__main__":
    force = "--force" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = [job for job in _jobs()
            if (not names or job[0] in names)
            and (force or names or not (RUNS / f"{job[0]}.npz").exists())]
    print(f"{len(todo)} runs to compute", flush=True)
    with Pool(10) as pool:
        for name in pool.imap_unordered(_work, todo):
            print("done", name, flush=True)
