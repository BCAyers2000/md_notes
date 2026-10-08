"""Figure fig:pe-cutoff: three ways to cut off a pair potential.

The Lennard-Jones potential, in reduced units, cut at r_c = 2.5σ by
truncation, by truncation and shifting, and by switching smoothly to zero
between 2σ and 2.5σ. (a) The energy and (b) the force near the cutoff,
with the full potential dashed. (c) The total energy of a cluster of 32
atoms followed for 20τ by the accurate solver of Chapter 2 with each
potential, as its change from the start.

The runs are cached in data/ch08_potentials/cutoff_runs.npz for Notebook
08; delete the file to run them again (about two minutes).

Prints the numbers of Section 8.5.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, potentials, rotation, viz
from mdlab.viz import (
    ACCENT,
    GREEN,
    OCHRE,
    OXBLOOD,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

R_CUT, R_SWITCH = 2.5, 2.0
CACHE = Path(viz.THEORY, "data", "ch08_potentials", "cutoff_runs.npz")
SCHEMES = ("none", "truncate", "shift", "switch")


def pair_for(scheme):
    if scheme == "none":
        return potentials.lennard_jones
    return potentials.with_cutoff(
        potentials.lennard_jones, R_CUT, scheme, r_switch=R_SWITCH
    )


def relaxed_cluster(n=32, seed=1):
    """A cluster near a local minimum of the full potential."""
    rng = np.random.default_rng(seed)
    grid = np.array(
        [[i, j, k] for i in range(4) for j in range(3) for k in range(3)],
        dtype=float,
    )[:n]
    r = 1.12 * grid + 0.02 * rng.normal(size=grid.shape)
    for _ in range(3000):
        f = potentials.pair_energy_forces(r, potentials.lennard_jones)[1]
        r = r + 0.002 * np.clip(f, -50, 50)
    v = rng.normal(size=r.shape) * np.sqrt(0.1)
    v = rotation.remove_rigid_motion(np.ones(n), r, v)
    return r, v


def run_all():
    r0, v0 = relaxed_cluster()
    times = np.linspace(0.0, 20.0, 801)
    out = {"times": times}
    for scheme in SCHEMES:
        pair = pair_for(scheme)
        calls = [0]

        def force(r, pair=pair, calls=calls):
            calls[0] += 1
            return potentials.pair_energy_forces(r, pair)[1]

        pos, vel = dynamics.solve_newton(
            force, np.ones(len(r0)), r0, v0, times, force_to_accel=1.0
        )
        energy = np.array(
            [
                0.5 * np.sum(v * v) + potentials.pair_energy_forces(r, pair)[0]
                for r, v in zip(pos, vel, strict=True)
            ]
        )
        out[f"energy_{scheme}"] = energy
        out[f"calls_{scheme}"] = calls[0]
        out[f"spread_{scheme}"] = np.ptp(
            np.linalg.norm(pos[-1] - pos[-1].mean(axis=0), axis=1)
        )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)


if not CACHE.exists():
    run_all()
runs = np.load(CACHE)
phi_c = float(potentials.lennard_jones(R_CUT)[0])
f_c = float(-potentials.lennard_jones(R_CUT)[1])
print(f"phi(r_c) = {phi_c:.5f} eps, force at r_c = {f_c:.5f} eps/sigma")
for scheme in SCHEMES:
    e = runs[f"energy_{scheme}"]
    print(
        f"{scheme:9s}: E(0) = {e[0]:.5f}, largest change "
        f"{np.abs(e - e[0]).max():.2e} eps, force calls "
        f"{int(runs[f'calls_{scheme}'])}"
    )
print(
    f"force calls, shift/none = "
    f"{runs['calls_shift'] / runs['calls_none']:.1f}, truncate/none = "
    f"{runs['calls_truncate'] / runs['calls_none']:.1f}, switch/none = "
    f"{runs['calls_switch'] / runs['calls_none']:.1f}"
)
jumps = np.abs(np.diff(runs["energy_truncate"]))
print(
    f"truncated: largest change between records {jumps.max():.4f} eps "
    f"({jumps.max() / abs(phi_c):.1f} times |phi(r_c)|)"
)

viz.use_style(notebook=False)
fig, axes = plt.subplots(
    1, 3, figsize=(viz.FULL, 2.3), gridspec_kw=dict(wspace=0.5)
)
r = np.linspace(1.9, 2.8, 900)
colours = {
    "truncate": OXBLOOD,
    "shift": OCHRE,
    "switch": GREEN,
    "none": ACCENT,
}
full_phi, full_dphi = potentials.lennard_jones(r)
axes[0].plot(r, full_phi, **REFERENCE_STYLE)
axes[1].plot(r, -full_dphi, **REFERENCE_STYLE)
for scheme in ("truncate", "shift", "switch"):
    phi, dphi = pair_for(scheme)(r)
    for ax, y in ((axes[0], phi), (axes[1], -dphi)):
        y = np.where(np.abs(r - R_CUT) < 1.5e-3, np.nan, y)  # show the jump
        ax.plot(r, y, color=colours[scheme], label=scheme)
for ax in axes[:2]:
    ax.axvline(R_CUT, **THRESHOLD_STYLE)
    ax.set_xlabel(r"$r / \sigma$")
axes[0].set_ylabel(r"$\varphi / \varepsilon$")
axes[0].set_ylim(-0.08, 0.01)
axes[1].set_ylabel(r"force / $(\varepsilon/\sigma)$")
axes[1].set_ylim(-0.22, 0.02)
axes[0].legend(loc="upper left", fontsize=7)
t = runs["times"]
for scheme in SCHEMES:
    e = runs[f"energy_{scheme}"]
    axes[2].plot(t, e - e[0], color=colours[scheme], lw=1.0)
axes[2].set_xlabel(r"time / $\tau$")
axes[2].set_ylabel(r"$E - E(0)$ / $\varepsilon$")
for ax, tag in zip(axes, "abc", strict=True):
    viz.panel_tag(ax, tag)

print("wrote", viz.save(fig, figure_path("ch08_potentials", "cutoff.pdf")))
