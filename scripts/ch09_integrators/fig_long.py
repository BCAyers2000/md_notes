"""Figure fig:in-long: long runs, other methods and rough forces.

The 32-atom Lennard-Jones cluster of Section 8.5, in reduced units
(m = ε = σ = 1), relaxed and given small random velocities.
(a) The change of the total energy over 200τ with velocity Verlet,
δt = 0.005τ, and with RK4, δt = 0.02τ, so that both call for the forces
the same number of times. (b) Velocity Verlet with δt = 0.005τ and the
potential cut at 2.5σ by shifting (the force jumps) and by switching
between 2σ and 2.5σ (it does not).

The runs are cached in data/ch09_integrators/long_runs.npz for Notebook
09; delete the file to run them again (about a minute).

Prints the numbers of Section 9.7.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mdlab import integrators, potentials, rotation, viz
from mdlab.viz import ACCENT, OCHRE, figure_path

CACHE = Path(viz.THEORY, "data", "ch09_integrators", "long_runs.npz")
T_END = 200.0


def relaxed_cluster(n=32, seed=1):
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


def energy_forces_for(pair):
    def energy_forces(r):
        u, f, _ = potentials.pair_energy_forces(r, pair)
        return u, f

    return energy_forces


def run(pair, method, dt, every):
    r0, v0 = relaxed_cluster()
    out = integrators.integrate(
        energy_forces_for(pair),
        np.ones(len(r0)),
        r0,
        v0,
        dt,
        int(round(T_END / dt)),
        method,
        every,
        force_to_accel=1.0,
    )
    return (
        out["times"],
        out["potential"] + out["kinetic"],
        int(out["force_calls"]),
    )


if not CACHE.exists():
    lj = potentials.lennard_jones
    shift = potentials.with_cutoff(lj, 2.5, "shift")
    switch = potentials.with_cutoff(lj, 2.5, "switch", r_switch=2.0)
    data = {}
    for name, pair, method, dt, every in (
        ("vv", lj, "velocity_verlet", 0.005, 20),
        ("rk4", lj, "rk4", 0.02, 5),
        ("shift", shift, "velocity_verlet", 0.005, 20),
        ("switch", switch, "velocity_verlet", 0.005, 20),
    ):
        t, e, calls = run(pair, method, dt, every)
        data[f"t_{name}"], data[f"e_{name}"] = t, e
        data[f"calls_{name}"] = calls
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **data)
runs = np.load(CACHE)
for name in ("vv", "rk4", "shift", "switch"):
    e = runs[f"e_{name}"]
    half = len(e) // 2
    print(
        f"{name}: force calls {int(runs[f'calls_{name}'])}, E(0) = "
        f"{e[0]:.4f}, spread {np.ptp(e):.2e}, mean change first half "
        f"{np.mean(e[:half]) - e[0]:+.2e}, second half "
        f"{np.mean(e[half:]) - e[0]:+.2e}, end {e[-1] - e[0]:+.2e}"
    )
    print(
        f"  second-half mean less first-half mean "
        f"{np.mean(e[half:]) - np.mean(e[:half]):+.2e}"
    )

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.4)
)
for name, colour, label, ax, style in (
    ("vv", ACCENT, r"velocity Verlet, $\delta t = 0.005\tau$", left, "-"),
    ("rk4", OCHRE, r"RK4, $\delta t = 0.02\tau$", left, "--"),
    ("switch", ACCENT, "switched", right, "-"),
    ("shift", OCHRE, "shifted", right, "--"),
):
    e = runs[f"e_{name}"]
    ax.plot(runs[f"t_{name}"], e - e[0], color=colour, lw=0.8,
            linestyle=style, label=label)
for ax, tag in ((left, "a"), (right, "b")):
    ax.set_xlabel(r"time / $\tau$")
    ax.set_ylabel(r"$E - E(0)$ / $\varepsilon$")
    ax.legend(loc="best", fontsize=7)
    viz.panel_tag(ax, tag)

print("wrote", viz.save(fig, figure_path("ch09_integrators", "long.pdf")))
