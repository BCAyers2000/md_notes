"""The thermostatted runs of Chapter 13, cached in data/ch13_thermostats/runs.

Run from scripts/ch13_thermostats/: python runs.py [--force]

Liquid argon, 256 atoms from three frames of the equilibrated liquid of
Chapter 12 (seeds s0, s1, s2), steps of 10 fs, 100 ps unless stated,
target 135 K, energies saved every 100 fs and positions and velocities
every 500 fs:

- nve, berendsen_100 and _1000 (τ in fs), andersen_0.1, _1 and _10
  (collision rate per atom, per ps), langevin_0.1, _1 and _10 (friction,
  per ps), nh_100 and _1000 (Nosé-Hoover), nhc_100 and _1000 (chains of
  3), csvr_100 and _1000, each from three starts (suffix _s0, _s1, _s2);
- rescale_s0; csvr_wrong_s0 (τ 100 fs, N_f = 3N, as if the drift were not
  removed); csvr_stride_s0 (τ 300 fs, acting every 3 steps, as TrajCast's
  acts once per 30 fs stride);
- hot_*: started at 200 K, 20 ps, for Berendsen, CSVR and Nosé-Hoover
  chains with τ = 1 ps and Langevin with γ = 1/ps;
- ice_*: started with a drift of 0.0002 Å/fs in each component, 200 ps,
  under Berendsen (τ 100 fs), rescaling and CSVR (τ 100 fs).

Lithium on the model surface: 1000 independent atoms with positions drawn
from e^{−βU} and velocities from the Maxwell-Boltzmann distribution at
1000 K, steps of 5 fs, 100 ps, every
other step saved; Langevin friction from 0 to 0.3/fs, and CSVR,
Nosé-Hoover, chains, Berendsen and Andersen each acting on one atom; and
li_hollow_*: Langevin from 10⁻⁵ to 3 × 10⁻³/fs with every atom started at
the bottom of a hollow, with the kinetic energy of 1000 K but no
potential energy.

About 10 minutes on 8 cores.
"""

import sys
from multiprocessing import Pool

import numpy as np
from ch13 import (
    DT_LI,
    DT_LIQUID,
    LI_MASS,
    N_FREE,
    RUNS,
    T_LI,
    T_LIQUID,
    argon_model,
    canonical_positions,
    liquid_start,
    surface,
)

from mdlab import statmech, thermostats
from mdlab.thermostats import (
    CSVR,
    Andersen,
    Berendsen,
    Langevin,
    NoseHooverChain,
    Rescale,
    Thermostat,
)


def _save(name, out, **extra):
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **out, **extra)


def liquid_job(name, thermostat, ps=100.0, start_t=None, drift=0.0,
               seed=0):
    r, v, h, m = liquid_start(seed)
    if start_t is not None:
        v = v * np.sqrt(start_t / statmech.kinetic_temperature(m, v, N_FREE))
    v = v + drift
    n = int(round(ps * 1000 / DT_LIQUID))
    out = thermostats.run(argon_model(h), m, r, v, DT_LIQUID, n, thermostat,
                          np.random.default_rng(seed), every=10,
                          keep=("positions", "velocities"))
    out["positions"] = out["positions"][::5]
    out["velocities"] = out["velocities"][::5]
    _save(name, out, masses=m, cell=h)


def surface_job(name, thermostat, ps=100.0, replicas=1000, seed=0,
                hollow=False):
    rng = np.random.default_rng(seed)
    m = np.array([LI_MASS])
    r = (np.zeros((replicas, 1, 2)) if hollow
         else canonical_positions(replicas, T_LI, rng))
    v = statmech.thermal_velocities(np.full(replicas, LI_MASS), T_LI, rng,
                                    remove_drift=False)[:, None, :2]
    n = int(round(ps * 1000 / DT_LI))
    out = thermostats.run(surface, m, r, v, DT_LI, n, thermostat,
                          np.random.default_rng(seed + 1), every=2,
                          keep=("positions",))
    _save(name, {"times": out["times"], "potential": out["potential"],
                 "kinetic": out["kinetic"], "heat": out["heat"],
                 "positions": out["positions"][:, :, 0, :]})


def _jobs():
    t, nf = T_LIQUID, N_FREE
    compared = {
        "nve": lambda: Thermostat(),
        "berendsen_100": lambda: Berendsen(t, 100.0, nf),
        "berendsen_1000": lambda: Berendsen(t, 1000.0, nf),
        "nh_100": lambda: NoseHooverChain(t, 100.0, nf, chain=1),
        "nh_1000": lambda: NoseHooverChain(t, 1000.0, nf, chain=1),
        "nhc_100": lambda: NoseHooverChain(t, 100.0, nf),
        "nhc_1000": lambda: NoseHooverChain(t, 1000.0, nf),
        "csvr_100": lambda: CSVR(t, 100.0, nf),
        "csvr_1000": lambda: CSVR(t, 1000.0, nf),
    }
    for rate in (0.1, 1, 10):
        compared[f"andersen_{rate:g}"] = (
            lambda rate=rate: Andersen(t, rate / 1000))
    for friction in (0.1, 1, 10):
        compared[f"langevin_{friction:g}"] = (
            lambda friction=friction: Langevin(t, friction / 1000))
    jobs = [(f"{name}_s{seed}", liquid_job, (make(), 100.0, None, 0.0, seed))
            for name, make in compared.items() for seed in range(3)]
    jobs += [
        ("rescale_s0", liquid_job, (Rescale(t, nf),)),
        ("csvr_wrong_s0", liquid_job, (CSVR(t, 100.0, 3 * 256),)),
        ("csvr_stride_s0", liquid_job, (CSVR(t, 300.0, nf, stride=3),)),
    ]
    for label, make in (
        ("berendsen", lambda: Berendsen(t, 1000.0, nf)),
        ("csvr", lambda: CSVR(t, 1000.0, nf)),
        ("nhc", lambda: NoseHooverChain(t, 1000.0, nf)),
        ("langevin", lambda: Langevin(t, 0.001)),
    ):
        jobs.append((f"hot_{label}", liquid_job, (make(), 20.0, 200.0)))
    for label, make in (
        ("berendsen", lambda: Berendsen(t, 100.0, nf)),
        ("rescale", lambda: Rescale(t, nf)),
        ("csvr", lambda: CSVR(t, 100.0, nf)),
    ):
        jobs.append((f"ice_{label}", liquid_job,
                     (make(), 200.0, None, 0.0002)))
    tl = T_LI
    for friction in (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03,
                     0.1, 0.3):
        jobs.append((f"li_langevin_{friction:g}", surface_job,
                     (Langevin(tl, friction),)))
    for friction in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3):
        jobs.append((f"li_hollow_{friction:g}", surface_job,
                     (Langevin(tl, friction), 100.0, 1000, 0, True)))
    for label, make in (
        ("csvr", lambda: CSVR(tl, 100.0, 2)),
        ("nh", lambda: NoseHooverChain(tl, 100.0, 2, chain=1)),
        ("nhc", lambda: NoseHooverChain(tl, 100.0, 2)),
        ("berendsen", lambda: Berendsen(tl, 100.0, 2)),
        ("andersen", lambda: Andersen(tl, 0.001)),
    ):
        jobs.append((f"li_{label}", surface_job, (make(),)))
    return jobs


def _work(job):
    name, function, args = job
    function(name, *args)
    return name


if __name__ == "__main__":
    force = "--force" in sys.argv
    todo = [job for job in _jobs()
            if force or not (RUNS / f"{job[0]}.npz").exists()]
    todo.sort(key=lambda job: job[0].startswith("li_"))  # long ones first
    print(f"{len(todo)} runs to compute", flush=True)
    with Pool(8) as pool:
        for name in pool.imap_unordered(_work, todo):
            print("done", name, flush=True)
