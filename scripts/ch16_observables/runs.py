"""The runs of Chapter 16, cached in data/ch16_observables/runs.

Run from scripts/ch16_observables/:
python runs.py [--force] [--workers=N] [name ...]

- salt_s0 to salt_s7: the model molten salt (64 ions, 13.4 Å cell, 2 fs
  steps), melted from its lattice at 4000 K for 10 ps, held at 1500 K by
  CSVR (τ_T = 0.1 ps) for 40 ps, then 1 ns at fixed energy; at every step
  the potential and kinetic energies, the charge displacement Σ qᵢrᵢ
  (positions unwrapped) and the charge current Σ qᵢvᵢ; positions
  (float32) every 100 fs; salt_s0 also keeps velocities every 10 fs over
  its first 100 ps;
- carbon_fast and carbon_slow: 216 carbon atoms at 2.0 g/cm³ (Tersoff),
  0.5 fs steps, CSVR (τ_T = 50 fs) at 6000 K for 2 ps, its target then
  lowered steadily to 300 K over 10 ps (fast) or 40 ps (slow), then 2 ps
  at 300 K; energies and positions every 10 fs;
- argon_nve: the liquid argon of Chapter 13's first start, 20 ps of CSVR
  at 135 K, then 100 ps at fixed energy with 10 fs steps, velocities
  (float32) at every step; argon_langevin: the same 100 ps under Langevin
  friction of 10 ps⁻¹ instead; positions every 500 fs;
- crystal_nve and crystal_langevin: the fcc crystal at the model's own
  lattice constant, velocities drawn at 40 K, 10 ps of CSVR at 20 K, then
  50 ps at fixed energy or under Langevin friction of 2 ps⁻¹, velocities
  every 10 fs and positions every 100 fs;
- gas_<T>: the A-B gas (500 atoms, 1 fs steps) from 250 AB molecules,
  100 ps under Langevin friction of 1 ps⁻¹ (a stand-in for a buffer gas)
  at 1500 to 2500 K, the bonds followed every 10 fs by an observer, with
  the band of hysteresis and with a single cutoff at r_on; species
  counts, bond events and reactions, and the positions every 10 fs over
  the first 5 ps;
- surface_<T>: 1000 lithium atoms on the model surface under Langevin
  friction of 1 ps⁻¹ (5 fs steps), started from e^{−βU}, 200 ps,
  positions every 50 fs; the first 200 also with velocities every 5 fs
  over 20 ps; surface_1000_g<γ>: the same at 1000 K with friction of 3,
  10 and 30 ps⁻¹;
- layered_150: Chapter 14's layered solid with its guests, at its relaxed
  cell, 20 ps then 500 ps of CSVR at 150 K (τ_T = 1 ps), the guests'
  positions and the centre of mass every 100 fs;
- chain_<T>: 200 copies of the four-bead chain under Langevin friction of
  1 ps⁻¹ at 300 and 200 K, 1 fs steps, 2 ns, ψ every 100 fs; chain_dense:
  20 copies at fixed energy after 10 ps at 300 K, velocities every 1 fs
  for 20 ps.

About 90 minutes on 10 cores, carbon_slow the longest.
"""

import sys
from multiprocessing import Pool

import ch16
import numpy as np
from ch16 import (
    CARBON_DT,
    CARBON_MASS,
    RUNS,
    SALT_DT,
    SALT_T,
    carbon_start,
    salt_lattice,
    salt_model,
    tersoff_carbon,
)

from mdlab import bonds, io, md, statmech, thermostats


def _save(name, out, **extra):
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{name}.npz", **out, **extra)


# ---------------------------------------------------------------- salt ---
def salt_job(name, seed, n_steps=500_000, chunk=50, every=50,
             velocity_steps=50_000, velocity_every=5):
    rng = np.random.default_rng(seed)
    r, q, m, h = salt_lattice()
    model = salt_model(q, h)
    n_free = statmech.degrees_of_freedom(len(q))
    v = statmech.thermal_velocities(m, 4000.0, rng)
    for temperature, steps in ((4000.0, 5000), (SALT_T, 20000)):
        out = thermostats.run(model, m, r, v, SALT_DT, steps,
                              thermostats.CSVR(temperature, 100.0, n_free),
                              rng, every=steps)
        r, v = out["positions"][-1], out["velocities"][-1]
    rec = {k: [] for k in ("potential", "kinetic", "charge_displacement",
                           "charge_current")}
    frames, vel_frames = [r.astype(np.float32)], []
    keep_velocities = seed == 0
    if keep_velocities:
        vel_frames.append(v.astype(np.float32))
    for c in range(n_steps // chunk):
        out = thermostats.run(model, m, r, v, SALT_DT, chunk, None, rng,
                              every=1)
        part = slice(0 if c == 0 else 1, None)
        rec["potential"].append(out["potential"][part])
        rec["kinetic"].append(out["kinetic"][part])
        rec["charge_displacement"].append(
            np.einsum("i,tix->tx", q, out["positions"][part]))
        rec["charge_current"].append(
            np.einsum("i,tix->tx", q, out["velocities"][part]))
        r, v = out["positions"][-1], out["velocities"][-1]
        step = (c + 1) * chunk
        if step % every == 0:
            frames.append(r.astype(np.float32))
        if keep_velocities and step <= velocity_steps:
            vel_frames += [x.astype(np.float32)
                           for x in out["velocities"][velocity_every::
                                                      velocity_every]]
    result = {k: np.concatenate(x) for k, x in rec.items()}
    result["times"] = SALT_DT * np.arange(len(result["potential"]))
    extra = {"positions": np.array(frames), "charges": q, "masses": m,
             "cell": h, "frame_dt": SALT_DT * every}
    if keep_velocities:
        extra["velocities"] = np.array(vel_frames)
        extra["velocity_dt"] = SALT_DT * velocity_every
    _save(name, result, **extra)


# -------------------------------------------------------------- carbon ---
def carbon_job(name, ramp_ps, seed=0, every=20):
    rng = np.random.default_rng(seed)
    r, h = carbon_start(seed)
    n = len(r)
    m = np.full(n, CARBON_MASS)
    model = io.AseModel(tersoff_carbon(), ["C"] * n, h)
    n_free = statmech.degrees_of_freedom(n)
    v = statmech.thermal_velocities(m, 6000.0, rng)
    ramp = int(ramp_ps * 1000 / CARBON_DT)
    hold = int(2000 / CARBON_DT)
    targets = np.concatenate([np.full(hold, 6000.0),
                              np.linspace(6000.0, 300.0, ramp),
                              np.full(hold, 300.0)])
    rec = {k: [] for k in ("times", "potential", "kinetic", "target",
                           "positions")}
    rec["times"].append(0.0)
    u0, _ = model(r)
    rec["potential"].append(u0)
    rec["kinetic"].append(thermostats.kinetic(m, v))
    rec["target"].append(6000.0)
    rec["positions"].append(r.astype(np.float32))
    for start in range(0, len(targets), every):
        csvr = thermostats.CSVR(float(targets[start]), 50.0, n_free)
        out = thermostats.run(model, m, r, v, CARBON_DT, every, csvr, rng,
                              every=every)
        r, v = out["positions"][-1], out["velocities"][-1]
        rec["times"].append((start + every) * CARBON_DT)
        rec["potential"].append(out["potential"][-1])
        rec["kinetic"].append(out["kinetic"][-1])
        rec["target"].append(float(targets[start]))
        rec["positions"].append(r.astype(np.float32))
    _save(name, {k: np.array(x) for k, x in rec.items()}, cell=h, masses=m)


# -------------------------------------------------------------- argon ---
def argon_job(name, friction=None, n_steps=10000):
    rng = np.random.default_rng(16)
    r, v, h, m = ch16.liquid_start(0)
    model = ch16.argon_model(h)
    n_free = statmech.degrees_of_freedom(len(m))
    out = thermostats.run(model, m, r, v, ch16.DT, 2000,
                          thermostats.CSVR(ch16.T_LIQUID, 1000.0, n_free),
                          rng, every=2000)
    r, v = out["positions"][-1], out["velocities"][-1]
    th = (None if friction is None
          else thermostats.Langevin(ch16.T_LIQUID, friction))
    out = thermostats.run(model, m, r, v, ch16.DT, n_steps, th, rng)
    _save(name, {"times": out["times"], "potential": out["potential"],
                 "kinetic": out["kinetic"],
                 "velocities": out["velocities"].astype(np.float32),
                 "positions": out["positions"][::50].astype(np.float32)},
          masses=m, cell=h)


def crystal_job(name, friction=None, n_steps=5000):
    rng = np.random.default_rng(17)
    r, h = ch16.ch12.fcc(4, ch16.crystal_a0())
    m = np.full(len(r), ch16.ch12.MASS)
    model = ch16.argon_model(h)
    n_free = statmech.degrees_of_freedom(len(m))
    v = statmech.thermal_velocities(m, 40.0, rng)
    out = thermostats.run(model, m, r, v, ch16.DT, 1000,
                          thermostats.CSVR(20.0, 500.0, n_free), rng,
                          every=1000)
    r, v = out["positions"][-1], out["velocities"][-1]
    th = None if friction is None else thermostats.Langevin(20.0, friction)
    out = thermostats.run(model, m, r, v, ch16.DT, n_steps, th, rng)
    _save(name, {"times": out["times"], "potential": out["potential"],
                 "kinetic": out["kinetic"],
                 "velocities": out["velocities"].astype(np.float32),
                 "positions": out["positions"][::10].astype(np.float32)},
          masses=m, cell=h)


# ---------------------------------------------------------------- gas ---
def gas_job(name, temperature, n_steps=100_000, chunk=1000, every=10,
            dense=5000):
    rng = np.random.default_rng(int(temperature))
    h = ch16.GAS_SIDE * np.eye(3)
    r = ch16.gas_molecules(rng)
    m = np.full(ch16.GAS_N, ch16.GAS_MASS)
    model = ch16.gas_model(h)
    v = statmech.thermal_velocities(m, temperature, rng)
    th = thermostats.Langevin(temperature, 0.001)
    logs = {"band": bonds.ReactionLog(ch16.gas_tracker()),
            "single": bonds.ReactionLog(ch16.gas_tracker(
                ch16.MORSE_ON, ch16.MORSE_ON))}
    clock = {"start": 0.0, "last": -1.0}
    frames = []

    def observe(t, x, _):
        time = clock["start"] + t
        if time <= clock["last"]:
            return  # the first frame of a chunk repeats the last one
        i, j, dist = model.last_pairs  # those of the force evaluation at x
        for log in logs.values():
            log.add(time, i, j, dist)
        if round(time / ch16.GAS_DT) <= dense:
            frames.append(x.astype(np.float32))
        clock["last"] = time

    kinetic = []
    for c in range(n_steps // chunk):
        out = thermostats.run(model, m, r, v, ch16.GAS_DT, chunk, th, rng,
                              every=every, observers=[observe])
        r, v = out["positions"][-1], out["velocities"][-1]
        kinetic.append(out["kinetic"][1:] if c else out["kinetic"])
        clock["start"] += chunk * ch16.GAS_DT
    result = {"kinetic": np.concatenate(kinetic),
              "positions": np.array(frames)}
    for key, log in logs.items():
        names = sorted({n for c in log.counts for n in c})
        result[f"{key}_species"] = np.array(names)
        result[f"{key}_counts"] = np.array(
            [[c.get(n, 0) for n in names] for c in log.counts])
        result[f"{key}_events"] = np.array(log.events)
        result[f"{key}_reaction_times"] = np.array(
            [t for t, _, _ in log.reactions])
        result[f"{key}_reactions"] = np.array(
            [" + ".join(a) + " -> " + " + ".join(b)
             for _, a, b in log.reactions])
    result["times"] = np.array(logs["band"].times)
    _save(name, result, cell=h, masses=m, frame_dt=every * ch16.GAS_DT)


# ------------------------------------------------------------ surface ---
def surface_job(name, temperature, friction=0.001, replicas=1000,
                n_steps=40_000, every=10, dense=4000, dense_atoms=200):
    ch13 = ch16.ch13
    rng = np.random.default_rng(int(temperature) + 1)
    r = ch13.canonical_positions(replicas, temperature, rng)
    m = np.array([ch13.LI_MASS])
    v = rng.normal(size=r.shape) * np.sqrt(
        ch16.units.KB * temperature / (ch13.LI_MASS * ch16.units.MV2_TO_EV))
    th = thermostats.Langevin(temperature, friction)
    out = thermostats.run(ch13.surface, m, r, v, ch13.DT_LI, dense, th, rng)
    velocities = out["velocities"][:, :dense_atoms, 0].astype(np.float32)
    frames = [x[:, 0].astype(np.float32) for x in out["positions"][::every]]
    r, v = out["positions"][-1], out["velocities"][-1]
    for _ in range((n_steps - dense) // every):
        out = thermostats.run(ch13.surface, m, r, v, ch13.DT_LI, every, th,
                              rng, every=every)
        r, v = out["positions"][-1], out["velocities"][-1]
        frames.append(r[:, 0].astype(np.float32))
    _save(name, {"positions": np.array(frames), "velocities": velocities},
          frame_dt=ch13.DT_LI * every, velocity_dt=ch13.DT_LI)


# ------------------------------------------------------------ layered ---
def layered_job(name, temperature, n_steps=100_000, every=20):
    ch14 = ch16.ch14
    rng = np.random.default_rng(14)
    base = np.load(ch14.RUNS / "layered_semi-isotropic.npz")
    r, h, types = base["positions"], base["cell"], base["types"]
    m = np.where(types == 3, ch14.GUEST_MASS, ch14.MASS)
    model = md.PairModel(ch14.layered_table(ch14.SIG_GUEST), h, ch14.R_CUT,
                         0.5, types)
    n_free = statmech.degrees_of_freedom(len(m))
    v = statmech.thermal_velocities(m, temperature, rng)
    csvr = thermostats.CSVR(temperature, 1000.0, n_free)
    out = thermostats.run(model, m, r, v, ch14.DT_LAYERED, 4000, csvr, rng,
                          every=4000)
    r, v = out["positions"][-1], out["velocities"][-1]
    guests = types == 3
    frames, centres, chunk = [], [], 1000
    for _ in range(n_steps // chunk):
        out = thermostats.run(model, m, r, v, ch14.DT_LAYERED, chunk, csvr,
                              rng, every=every)
        part = out["positions"][1:]
        frames.append(part[:, guests].astype(np.float32))
        centres.append(np.einsum("i,tix->tx", m / m.sum(), part))
        r, v = out["positions"][-1], out["velocities"][-1]
    _save(name, {"guests": np.concatenate(frames),
                 "centre": np.concatenate(centres)},
          frame_dt=ch14.DT_LAYERED * every, cell=h, guest_mass=m[guests][0])


# -------------------------------------------------------------- chain ---
def chain_job(name, temperature, copies=200, n_steps=2_000_000,
              every=100, chunk=10_000):
    rng = np.random.default_rng(int(temperature) + 2)
    model = ch16.chain_model()
    m = np.full(4, ch16.CHAIN_MASS)
    r = np.repeat(ch16.chain_trans()[None], copies, axis=0)
    v = rng.normal(size=r.shape) * np.sqrt(
        ch16.units.KB * temperature / (ch16.CHAIN_MASS
                                       * ch16.units.MV2_TO_EV))
    th = thermostats.Langevin(temperature, 0.001)
    psi = []
    for _ in range(n_steps // chunk):
        out = thermostats.run(model, m, r, v, ch16.CHAIN_DT, chunk, th, rng,
                              every=every)
        psi.append(ch16.chain_dihedrals(out["positions"][1:]))
        r, v = out["positions"][-1], out["velocities"][-1]
    _save(name, {"psi": np.concatenate(psi).astype(np.float32)},
          frame_dt=ch16.CHAIN_DT * every, temperature=temperature)


def chain_dense_job(name, copies=20, n_steps=20_000):
    rng = np.random.default_rng(5)
    model = ch16.chain_model()
    m = np.full(4, ch16.CHAIN_MASS)
    r = np.repeat(ch16.chain_trans()[None], copies, axis=0)
    v = rng.normal(size=r.shape) * np.sqrt(
        ch16.units.KB * 300.0 / (ch16.CHAIN_MASS * ch16.units.MV2_TO_EV))
    out = thermostats.run(model, m, r, v, ch16.CHAIN_DT, 10_000,
                          thermostats.Langevin(300.0, 0.001), rng,
                          every=10_000)
    r, v = out["positions"][-1], out["velocities"][-1]
    out = thermostats.run(model, m, r, v, ch16.CHAIN_DT, n_steps, None, rng)
    _save(name, {"velocities": out["velocities"].astype(np.float32),
                 "potential": out["potential"], "kinetic": out["kinetic"]},
          velocity_dt=ch16.CHAIN_DT)


def _jobs():
    jobs = [("carbon_slow", carbon_job, (40.0,)),
            ("carbon_fast", carbon_job, (10.0,))]
    jobs += [(f"salt_s{s}", salt_job, (s,)) for s in range(8)]
    jobs += [(f"chain_{t:g}", chain_job, (t,)) for t in (300.0, 200.0)]
    jobs += [(f"gas_{t:g}", gas_job, (t,))
             for t in (1500.0, 1750.0, 2000.0, 2250.0, 2500.0)]
    jobs += [(f"surface_{t:g}", surface_job, (t,)) for t in (1000.0, 600.0)]
    jobs += [(f"surface_1000_g{g:g}", surface_job, (1000.0, g / 1000))
             for g in (3.0, 10.0, 30.0)]
    jobs += [("layered_150", layered_job, (150.0,)),
             ("argon_nve", argon_job, ()),
             ("argon_langevin", argon_job, (0.01,)),
             ("crystal_nve", crystal_job, ()),
             ("crystal_langevin", crystal_job, (0.002,)),
             ("chain_dense", chain_dense_job, ())]
    return jobs


def _work(job):
    name, function, args = job
    function(name, *args)
    return name


if __name__ == "__main__":
    force = "--force" in sys.argv
    workers = next((int(a.split("=")[1]) for a in sys.argv
                    if a.startswith("--workers=")), 10)
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = [job for job in _jobs()
            if (not names or job[0] in names)
            and (force or names or not (RUNS / f"{job[0]}.npz").exists())]
    print(f"{len(todo)} runs to compute", flush=True)
    with Pool(workers) as pool:
        for name in pool.imap_unordered(_work, todo):
            print("done", name, flush=True)
