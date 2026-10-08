"""Shared set-up of the Chapter 15 scripts.

The liquid argon of Chapters 12 to 14 (256 atoms, 0.8σ⁻³, switched
Lennard-Jones, 10 fs steps), held at 135 K by CSVR with τ_T = 1 ps unless
a run says otherwise, and the series of known answer used to try each
method first: the Ornstein-Uhlenbeck step x' = cx + √(1 − c²)ξ, whose
correlation is c^k and whose statistical inefficiency is (1 + c)/(1 − c).
"""

import sys
from pathlib import Path

import numpy as np

from mdlab import barostats, statmech, units, viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch12_ensembles"))
sys.path.insert(0, str(HERE.parent / "ch13_thermostats"))
sys.path.insert(0, str(HERE.parent / "ch14_pressure"))
import ch12  # noqa: E402, F401  (re-exported for the scripts)
import ch13  # noqa: E402
import ch14  # noqa: E402

DATA = Path(viz.THEORY, "data", "ch15_convergence")
RUNS = DATA / "runs"
T_LIQUID = 135.0
N_ATOMS = 256
N_FREE = statmech.degrees_of_freedom(N_ATOMS)  # 765
DT = 10.0
TAU_T = 1000.0
KT = units.KB * T_LIQUID
GPA = units.EV_PER_A3_TO_GPA  # 1 eV/Å³ in GPa

P_LIQUID = ch14.P_LIQUID  # 0.1 GPa, in eV/Å³
KAPPA_LIQUID = ch14.KAPPA_LIQUID  # the barostats' rough 2.5 GPa⁻¹

liquid_start = ch13.liquid_start
argon_model = ch13.argon_model
diffusion = ch13.diffusion


def ou(n, c, rng, shape=()):
    """The OU process of unit variance over n steps, started settled."""
    x = np.empty((n, *shape))
    x[0] = rng.standard_normal(shape)
    kick = np.sqrt(1 - c**2) * rng.standard_normal((n, *shape))
    for k in range(1, n):
        x[k] = c * x[k - 1] + kick[k]
    return x


def chunked(model, masses, r, v, n_chunks, chunk, thermostat=None,
            barostat=None, rng=None, dt=DT):
    """Run ``barostats.run`` in chunks, keeping positions at each chunk's end.

    Every step's energies and pressure are kept. Returns times, potential,
    kinetic, heat (accumulated), volume, pressure (a third of the trace of
    the instantaneous tensor), shear (its xy, xz and yz components),
    positions (float32) and cells; stops early if the energy is no longer
    finite.
    """
    rng = np.random.default_rng() if rng is None else rng
    rec = {k: [] for k in ("times", "potential", "kinetic", "heat",
                           "volume", "pressure", "shear")}
    frames, cells = [r.astype(np.float32)], [np.array(model.cell)]
    clock, heat = 0.0, 0.0
    for c in range(n_chunks):
        out = barostats.run(model, masses, r, v, dt, chunk, barostat,
                            thermostat, rng, every=1,
                            keep=("positions", "velocities"))
        keep = slice(0 if c == 0 else 1, None)
        p = out["pressure"]
        rec["times"].append(out["times"][keep] + clock)
        rec["potential"].append(out["potential"][keep])
        rec["kinetic"].append(out["kinetic"][keep])
        rec["heat"].append(out["heat"][keep] + heat)
        rec["volume"].append(out["volume"][keep])
        rec["pressure"].append(np.trace(p, axis1=1, axis2=2)[keep] / 3)
        rec["shear"].append(p[:, [0, 0, 1], [1, 2, 2]][keep])
        r, v = out["positions"][-1], out["velocities"][-1]
        frames.append(r.astype(np.float32))
        cells.append(out["cell"][-1])
        clock += chunk * dt
        heat += out["heat"][-1]
        if not np.isfinite(out["potential"][-1]):
            break
    result = {k: np.concatenate(x) for k, x in rec.items()}
    result["positions"] = np.array(frames)
    result["cells"] = np.array(cells)
    result["final_velocities"] = v
    return result


def dashboard(axes, series, dt, report, unit="", colour="#00546D"):
    """Draw the four panels of a convergence dashboard on ``axes``.

    (a) the series against time in ps, with the start found (dotted) and
    the mean after it (dashed); (b) the running mean from the start with
    its error band, √(g s²/n) at each length; (c) the error of the mean
    from blocks against the block length in ps, with √(g s²/n) (dashed);
    (d) the autocorrelation function against the lag in ps, with the lag
    at which the estimate of g stops (dotted).
    """
    from mdlab.analysis import stats
    from mdlab.viz import REFERENCE, REFERENCE_STYLE, THRESHOLD_STYLE

    a = np.asarray(series, dtype=float)
    t = np.arange(len(a)) * dt / 1000
    start = int(round(report.start / dt))
    rest = a[start:]
    ax_a, ax_b, ax_c, ax_d = axes
    ax_a.plot(t, a, color=colour, lw=0.3)
    ax_a.axvline(report.start / 1000, **THRESHOLD_STYLE)
    ax_a.axhline(report.mean, **REFERENCE_STYLE, lw=0.8)
    ax_a.set_xlabel("time / ps")
    ax_a.set_ylabel(unit)

    lengths = np.unique(np.geomspace(max(20, report.inefficiency * 2),
                                     len(rest), 40).astype(int))
    means, errors = [], []
    for n in lengths:
        m, e, _ = stats.standard_error(rest[:n])
        means.append(m)
        errors.append(e)
    means, errors = np.array(means), np.array(errors)
    span = (start + lengths) * dt / 1000
    ax_b.fill_between(span, means - errors, means + errors,
                      color=colour, alpha=0.25, lw=0)
    ax_b.plot(span, means, color=colour, lw=0.9)
    ax_b.axhline(report.mean, **REFERENCE_STYLE, lw=0.8)
    ax_b.set_xlabel("time / ps")
    ax_b.set_ylabel("running mean")

    blocks = stats.block_average(rest, min_blocks=8)
    ax_c.errorbar(blocks["length"] * dt / 1000, blocks["error"],
                  yerr=blocks["error_error"], fmt="o", ms=2.5, lw=0.8,
                  color=colour, capsize=0)
    ax_c.axhline(report.error, **REFERENCE_STYLE, lw=0.8)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("block / ps")
    ax_c.set_ylabel("error of the mean")

    acf = stats.autocorrelation(rest)
    k = np.arange(1, len(acf) - 1)
    stop = np.nonzero((acf[1:-1] <= 0) & (k > 3))[0]
    cut = stop[0] + 1 if len(stop) else len(acf) - 1
    shown = min(len(acf), 3 * cut)
    ax_d.plot(np.arange(shown) * dt / 1000, acf[:shown], color=colour,
              lw=0.9)
    ax_d.axvline(cut * dt / 1000, **THRESHOLD_STYLE)
    ax_d.axhline(0, color=REFERENCE, lw=0.4)
    ax_d.set_xlabel("lag / ps")
    ax_d.set_ylabel(r"$\mathrm{corr}$")
