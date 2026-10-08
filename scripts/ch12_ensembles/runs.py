"""The argon and water runs of Chapter 12, computed in parallel and cached.

Each run is saved to data/ch12_ensembles/runs/<name>.npz. Runs already
cached are skipped; pass --force to compute them again. The figure scripts
and Notebook 12 read the cache.

- liquid: 256 atoms near 130 K, 50 ps at 10 fs, velocities every 50 fs;
- heat_<T>: 256 atoms prepared near 100, 115, 130, 145 and 160 K, 50 ps;
- size_<N>: 108, 256, 500 and 864 atoms near 130 K, 50 ps;
- lattice: 256 atoms on ASE's argon lattice, a = 5.26 Å, velocities at
  40 K, 20 ps;
- drift: the liquid started with a drift of 0.0005 Å/fs left in, 20 ps;
- mixture_masses and mixture_massless: 256 atoms with argon's forces, half
  of mass 39.948 and half of 83.798 amu, started with velocities drawn at
  130 K with the masses, and with one spread for every atom, 20 ps;
- uniform: the liquid started with velocity components spread evenly over
  an interval instead of drawn from the Gaussian, 10 ps.

About 3 minutes on 8 cores.
"""

import sys
from multiprocessing import Pool

import numpy as np
from ch12 import A_ASE, MASS, RUNS, fcc, liquid, model

from mdlab import md, statmech, units

SAVE = ("times", "potential", "kinetic")


def _save(name, out, every_v=5, **extra):
    keep = {key: out[key] for key in SAVE}
    keep["velocities"] = out["velocities"][::every_v]
    keep["frame_times"] = out["times"][::every_v]
    keep["positions"] = out["positions"][::every_v]
    keep.update(extra)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **keep)


def run_liquid(name, n, temperature, seed, ps=50.0, skin=1.0, drift=0.0):
    rng = np.random.default_rng(seed)
    r, v, h, m, w = liquid(n, temperature, rng, skin=skin)
    v = v + drift
    out = md.run(w, m, r, v, h, 10.0, int(ps * 100), every=1)
    _save(name, out, masses=m, cell=h, asked=temperature)


def run_lattice(name, seed):
    rng = np.random.default_rng(seed)
    r, h = fcc(4, A_ASE)
    m = np.full(len(r), MASS)
    v = statmech.thermal_velocities(m, 40.0, rng)
    out = md.run(model(h), m, r, v, h, 10.0, 2000, every=1)
    _save(name, out, masses=m, cell=h, asked=40.0)


def run_mixture(name, seed, with_masses):
    rng = np.random.default_rng(seed)
    r, v, h, _, w = liquid(4, 130.0, rng)
    m = np.where(np.arange(len(r)) % 2 == 0, MASS, 83.798)
    if with_masses:
        v = statmech.thermal_velocities(m, 130.0, rng)
    else:
        v = rng.standard_normal(r.shape)  # one spread for every atom
        v -= (m @ v) / m.sum()
        n_free = statmech.degrees_of_freedom(len(m))
        v *= np.sqrt(n_free * 0.5 * units.KB * 130.0
                     / statmech.kinetic_energy(m, v))
    out = md.run(w, m, r, v, h, 10.0, 2000, every=1)
    _save(name, out, every_v=1, masses=m, cell=h, asked=130.0)


def run_uniform(name, seed):
    rng = np.random.default_rng(seed)
    r, _, h, m, w = liquid(4, 130.0, rng)
    v = rng.uniform(-1.0, 1.0, r.shape)
    v -= (m @ v) / m.sum()
    n_free = statmech.degrees_of_freedom(len(m))
    v *= np.sqrt(n_free * 0.5 * units.KB * 130.0
                 / statmech.kinetic_energy(m, v))
    out = md.run(w, m, r, v, h, 10.0, 1000, every=1)
    _save(name, out, every_v=1, masses=m, cell=h, asked=130.0)


def _jobs():
    jobs = [("liquid", run_liquid, (4, 130.0, 1))]
    for k, t in enumerate((100.0, 115.0, 130.0, 145.0, 160.0)):
        jobs.append((f"heat_{t:g}", run_liquid, (4, t, 10 + k)))
    for n, skin in ((3, 0.2), (4, 1.0), (5, 1.0), (6, 1.0)):
        jobs.append((f"size_{4 * n**3}", run_liquid,
                     (n, 130.0, 20 + n, 50.0, skin)))
    jobs.append(("lattice", run_lattice, (30,)))
    jobs.append(("drift", run_liquid, (4, 130.0, 1, 20.0, 1.0, 0.0005)))
    jobs.append(("mixture_masses", run_mixture, (40, True)))
    jobs.append(("mixture_massless", run_mixture, (40, False)))
    jobs.append(("uniform", run_uniform, (50,)))
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
