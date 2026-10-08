"""Every water run of Chapter 11, computed in parallel and cached.

Each run starts from the state of prepare_water.py and is saved to
data/ch11_constraints/runs/<name>.npz with its times, potential and
kinetic energies, the largest bond error where bonds are held, and the
positions every 50 fs. Runs already cached are skipped; pass --force to
compute them again. The figure scripts and Notebook 11 read the cache.

The runs:

- step scans (fig_step): rigid water by RATTLE, with ordinary and with
  repartitioned masses; water with only its O–H bonds held, both ways;
  flexible water by velocity Verlet;
- RESPA scan (fig_respa, fig_resonance): flexible water, springs fast and
  everything else slow, inner step 0.25 fs, outer steps 0.25 to 4.5 fs;
- a healthy rigid run of 10 ps and the same run broken in eight ways
  (fig_healthy, fig_faults, Section 11.8).

About 6 minutes on 12 cores.
"""

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ch11 import DATA, N_MOLECULES, PER_FREEDOM, SEED, draw, load, model

from mdlab import cell, constraints, md, respa, units, water
from mdlab.neighbours import all_pairs
from mdlab.units import FORCE_TO_ACCEL, MV2_TO_EV

RUNS = DATA / "runs"
HMR_H = 3.024  # three times the mass of hydrogen
POSITIONS_EVERY = 50.0  # fs between saved frames


def _hmr(masses):
    oxygen = np.arange(0, len(masses), 3)
    pairs = np.concatenate([np.stack([oxygen, oxygen + 1], 1),
                            np.stack([oxygen, oxygen + 2], 1)])
    return constraints.repartition_masses(masses, pairs, HMR_H)


def _save(name, out, dt, **extra):
    stride = max(1, int(round(POSITIONS_EVERY / dt)))
    keep = {
        "times": out["times"],
        "potential": out["potential"],
        "kinetic": out["kinetic"],
        "frames": out["positions"][::stride],
        "frame_times": out["times"][::stride],
        "dt": dt,
    }
    if "bond_error" in out:
        keep["bond_error"] = out["bond_error"]
    keep.update(extra)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **keep)


def _held(kind, s, masses, rng):
    """Start positions, velocities and bonds for rigid or O–H runs."""
    rigid = kind == "rigid"
    bonds, lengths = water.constraint_bonds(N_MOLECULES, rigid=rigid)
    r = s["positions"]
    if rigid and np.array_equal(masses, s["masses"]):
        v = s["velocities"]
    else:
        v = draw(masses, r, rng, bonds, lengths)
    return r, v, bonds, lengths


def run_held(name, kind, dt, t_f, hmr=False):
    s = load()
    h = s["cell"]
    m = _hmr(s["masses"]) if hmr else s["masses"]
    rng = np.random.default_rng(SEED + 1)
    r, v, bonds, lengths = _held(kind, s, m, rng)
    w = model(h, flexible=(kind == "bonds"))
    out = constraints.run(w, m, r, v, h, dt, int(round(t_f / dt)), bonds,
                          lengths)
    _save(name, out, dt, masses=m)


def run_flexible(name, dt, t_f):
    s = load()
    h = s["cell"]
    out = md.run(model(h, flexible=True), s["masses"],
                 s["flexible_positions"], s["flexible_velocities"], h, dt,
                 int(round(t_f / dt)))
    _save(name, out, dt)


def run_respa(name, dt, t_f, inner=0.25):
    s = load()
    h = s["cell"]
    w = model(h, flexible=True)
    out = respa.run(w.intramolecular, w.intermolecular, s["masses"],
                    s["flexible_positions"], s["flexible_velocities"], dt,
                    max(1, int(round(dt / inner))), int(round(t_f / dt)))
    _save(name, out, dt, slow_calls=out["slow_calls"],
          fast_calls=out["fast_calls"])


class TruncatedCoulomb:
    """The rigid model with Coulomb cut off atom by atom at r_cut."""

    def __init__(self, h, r_cut=6.0):
        self.base = model(h)
        self.h, self.r_cut = h, r_cut
        self.q = self.base.charges

    def __call__(self, r):
        """The energy and forces, Coulomb truncated."""
        u_lj, f = self.base.lennard_jones(r)
        i, j, d, dist = all_pairs(r, self.h, self.r_cut)
        other = (i // 3) != (j // 3)
        i, j, d, dist = i[other], j[other], d[other], dist[other]
        qq = units.COULOMB * self.q[i] * self.q[j]
        push = (qq / dist**3)[:, None] * d  # the force on j
        np.add.at(f, j, push)
        np.add.at(f, i, -push)
        return u_lj + float(np.sum(qq / dist)), f


def _loop_until_failure(step, r, v, f, u, n_steps, dt, m, record_bonds):
    """Steps until done or until the arithmetic fails."""
    out = {key: [] for key in ("times", "positions", "potential",
                               "kinetic", "bond_error")}
    for k in range(n_steps + 1):
        if k > 0:
            try:
                with np.errstate(all="raise"):
                    r, v, u, f = step(r, v, f)
            except (FloatingPointError, RuntimeError, ValueError):
                break
            if not np.all(np.isfinite(r)) or abs(u) > 1e6:
                break
        out["times"].append(k * dt)
        out["positions"].append(r.copy())
        out["potential"].append(u)
        out["kinetic"].append(0.5 * MV2_TO_EV * float(m @ (v * v).sum(1)))
        out["bond_error"].append(record_bonds(r))
    return {key: np.array(value) for key, value in out.items()}


def run_fault(name, t_f=2000.0):
    s = load()
    h, m = s["cell"], s["masses"]
    r, v = s["positions"].copy(), s["velocities"].copy()
    bonds, lengths = water.constraint_bonds(N_MOLECULES)
    w = model(h)
    dt = 2.0
    extra = {}
    if name == "fault_cutoff":
        w = TruncatedCoulomb(h)
    elif name == "fault_units":
        # Å/fs read as Å per ASE time unit, 1/√FORCE_TO_ACCEL fs
        v = v * math.sqrt(FORCE_TO_ACCEL)
        extra["intended"] = PER_FREEDOM
    elif name == "fault_drift":
        rng = np.random.default_rng(SEED + 2)
        v = rng.normal(size=r.shape) / np.sqrt(m)[:, None]  # drift kept
        v, _ = constraints.rattle_velocities(r, v, m, bonds, lengths, h,
                                             tolerance=1e-13)
        k = 0.5 * MV2_TO_EV * float(m @ (v * v).sum(1))
        v *= math.sqrt((6 * N_MOLECULES - 3) * PER_FREEDOM / k)
        extra["momentum"] = m @ v
    elif name == "fault_overlap":
        # molecule 1 moved so that its oxygen sits 1.0 Å from molecule 0's
        shift = r[0] + np.array([1.0, 0.0, 0.0]) - r[3]
        r[3:6] += shift
        t_f = 200.0
    if name == "fault_tolerance":
        tol = 1e-3
    else:
        tol = 1e-10

    def errors(x):
        return float(np.abs(constraints.bond_errors(x, bonds, lengths,
                                                    h)).max())

    if name == "fault_wrapped":
        def step(r_, v_, f_):
            r_, v_, u_, f_ = constraints.rattle_step(
                r_, v_, f_, m, w, dt, bonds, lengths, cell=None,
                tolerance=tol)
            return cell.wrap(r_, h), v_, u_, f_

        def errors_raw(x):  # bonds measured as a code without copies sees
            return float(np.abs(constraints.bond_errors(x, bonds,
                                                        lengths)).max())

        u, f = w(r)
        out = _loop_until_failure(step, r, v, f, u, int(t_f / dt), dt, m,
                                  errors_raw)
    else:
        v, _ = constraints.rattle_velocities(r, v, m, bonds, lengths, h,
                                             tolerance=tol)

        def step(r_, v_, f_):
            return constraints.rattle_step(r_, v_, f_, m, w, dt, bonds,
                                           lengths, cell=h, tolerance=tol)

        u, f = w(r)
        extra["first_forces"] = f
        out = _loop_until_failure(step, r, v, f, u, int(t_f / dt), dt, m,
                                  errors)
    _save(name, out, dt, **extra)


def _jobs():
    jobs = []
    for dt in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        jobs.append((f"rigid_{dt:g}", run_held, ("rigid", dt, 2000.0)))
    for dt in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        jobs.append((f"rigid_hmr_{dt:g}", run_held,
                     ("rigid", dt, 2000.0, True)))
    for dt in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        jobs.append((f"bonds_{dt:g}", run_held, ("bonds", dt, 1000.0)))
    for dt in (1.0, 2.0, 3.0, 4.0, 5.0):
        jobs.append((f"bonds_hmr_{dt:g}", run_held,
                     ("bonds", dt, 1000.0, True)))
    for dt in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        jobs.append((f"flexible_{dt:g}", run_flexible, (dt, 1000.0)))
    jobs.append(("fault_step", run_flexible, (2.0, 2000.0)))
    for dt in np.arange(0.25, 4.51, 0.25):
        jobs.append((f"respa_{dt:g}", run_respa, (float(dt), 1000.0)))
    jobs.append(("respa_2_inner0.125", run_respa, (2.0, 1000.0, 0.125)))
    jobs.append(("rigid_healthy", run_held, ("rigid", 2.0, 10000.0)))
    for fault in ("cutoff", "units", "wrapped", "overlap", "tolerance",
                  "drift"):
        jobs.append((f"fault_{fault}", run_fault, ()))
    return jobs


def _work(job):
    name, function, args = job
    if function is run_fault:
        function(name)
    else:
        function(name, *args)
    return name


if __name__ == "__main__":
    force = "--force" in sys.argv
    todo = [job for job in _jobs()
            if force or not Path(RUNS / f"{job[0]}.npz").exists()]
    print(f"{len(todo)} runs to compute")
    with Pool(12) as pool:
        for name in pool.imap_unordered(_work, todo):
            print("done", name, flush=True)
