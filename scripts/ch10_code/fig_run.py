"""Figure fig:md-run: two periodic runs of the code of Chapter 10.

(a) 500 Lennard-Jones argon atoms (ε = 0.01034 eV, σ = 3.4 Å, the
potential switched off between 2σ and 2.5σ), started on the model's own
fcc lattice (a0 = 1.54916σ, from numbers_chapter.py) with 0.02 eV of
kinetic energy per atom, followed for 20 ps with δt = 10 fs: kinetic,
potential and total energy per atom over the first 5 ps. (b) A model
salt in reduced units: 64 ions of charge ±1 and mass 1 on a rock-salt
lattice at the spacing that minimises its energy, with a purely repulsive
Lennard-Jones wall (cut at 2^(1/6)σ) and Coulomb energy k_e q q'/r with
k_e = 10 εσ, summed
by Ewald; 0.5ε of kinetic energy per ion, followed for 2τ: the change of
the total energy with two steps.

The runs are cached in data/ch10_code/runs.npz for Notebook 10; delete
the file to run them again (about 20 s).

Prints the numbers of Section 10.6.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from mdlab import cell, ewald, md, potentials, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, figure_path, tint

CACHE = Path(viz.THEORY, "data", "ch10_code", "runs.npz")
EPS, SIG, MASS = 0.01034, 3.4, 39.948
A0 = 1.54916 * SIG
K_E = 10.0
WCA = potentials.with_cutoff(potentials.lennard_jones, 2 ** (1 / 6), "shift")


def fcc(n, a):
    base = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    grid = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)],
        float,
    )
    s = (grid[:, None, :] + base[None, :, :]).reshape(-1, 3) / n
    h = n * a * np.eye(3)
    return cell.to_cartesian(s, h), h


def rock_salt(n, d):
    base = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    ions = np.vstack([base, base + [1.0, 0.0, 0.0]])
    grid = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)],
        float,
    )
    r = (ions[None, :, :] + 2 * grid[:, None, :]).reshape(-1, 3) * d
    q = np.tile([1.0] * 4 + [-1.0] * 4, len(grid))
    return r, q, 2 * n * d * np.eye(3)


def salt_energy_per_pair(d):
    r, q, h = rock_salt(2, d)
    i, j = np.triu_indices(len(r), 1)
    dist = np.linalg.norm(cell.minimum_image(r[j] - r[i], h), axis=1)
    wall = float(np.sum(WCA(dist)[0]))
    coulomb = ewald.ewald_energy_forces(q, r, h, coulomb=K_E)[0]
    return (wall + coulomb) / (len(q) / 2)


if not CACHE.exists():
    data = {}
    r, h = fcc(5, A0)
    m = np.full(len(r), MASS)
    pair = potentials.with_cutoff(
        lambda x: potentials.lennard_jones(x, EPS, SIG),
        2.5 * SIG,
        "switch",
        r_switch=2.0 * SIG,
    )
    model = md.PairModel(pair, h, 2.5 * SIG, 1.0)
    v = md.starting_velocities(m, 0.02 * len(r), np.random.default_rng(11))
    out = md.run(model, m, r, v, h, 10.0, 2000, every=2)
    for key in ("times", "potential", "kinetic"):
        data[f"argon_{key}"] = out[key]
    data["argon_builds"] = model.neighbours.builds
    data["argon_momentum"] = np.abs(m @ out["velocities"][-1]).max()

    d0 = minimize_scalar(
        salt_energy_per_pair,
        bounds=(0.9, 1.2),
        method="bounded",
        options={"xatol": 1e-8},
    ).x
    data["salt_d0"] = d0
    data["salt_energy"] = salt_energy_per_pair(d0)
    r, q, h = rock_salt(2, d0)
    m = np.ones(len(q))
    for dt in (0.002, 0.004):
        model = md.sum_of(
            [
                md.PairModel(WCA, h, 2 ** (1 / 6), 0.3),
                md.EwaldModel(q, h, accuracy=1e-8, coulomb=K_E),
            ]
        )
        v = md.starting_velocities(
            m, 0.5 * len(q), np.random.default_rng(12), 1.0
        )
        out = md.run(
            model,
            m,
            r,
            v,
            h,
            dt,
            int(round(2.0 / dt)),
            every=5,
            force_to_accel=1.0,
        )
        data[f"salt_{dt}_times"] = out["times"]
        data[f"salt_{dt}_total"] = out["potential"] + out["kinetic"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **data)
runs = np.load(CACHE)

n = 500
e = (runs["argon_potential"] + runs["argon_kinetic"]) / n
print(
    f"argon: E per atom {e[0]:.5f} eV, spread {np.ptp(e):.1e} eV per "
    f"atom ({np.ptp(e) / abs(e[0]):.1e} of it); K per atom from 0.0200 to "
    f"mean {runs['argon_kinetic'][500:].mean() / n:.4f} eV over the last "
    f"10 ps; {int(runs['argon_builds'])} list builds in 2000 steps; "
    f"largest momentum component {float(runs['argon_momentum']):.1e}"
)
t_fs = runs["argon_times"]
k_atom = runs["argon_kinetic"] / n
first = np.argmax(k_atom <= 0.5 * k_atom[0])
low = np.argmin(k_atom[t_fs <= 1000])
print(
    f"argon: kinetic energy per atom first at half its start at "
    f"{t_fs[first] / 1000:.2f} ps; lowest {1000 * k_atom[low]:.2f} meV at "
    f"{t_fs[low] / 1000:.2f} ps; mean over 0.5-1 ps "
    f"{1000 * k_atom[(t_fs >= 500) & (t_fs <= 1000)].mean():.2f} meV"
)
settled = t_fs >= 10000.0
e_tot = (runs["argon_potential"] + runs["argon_kinetic"])[settled]
print(
    f"argon over the last 10 ps: std of E / std of K = "
    f"{np.std(e_tot) / np.std(runs['argon_kinetic'][settled]):.4f}"
)
late = runs["argon_times"] >= 1000.0
kin_late = runs["argon_kinetic"][late] / n
print(
    f"argon after 1 ps: kinetic energy per atom varies by "
    f"{1000 * np.ptp(kin_late):.1f} meV peak to peak, standard deviation "
    f"{1000 * np.std(kin_late):.2f} meV"
)
print(
    f"salt: d0 = {float(runs['salt_d0']):.4f} sigma, energy per ion pair "
    f"{float(runs['salt_energy']):.3f} epsilon"
)
spreads = {dt: np.ptp(runs[f"salt_{dt}_total"]) for dt in (0.002, 0.004)}
for dt, s in spreads.items():
    print(f"salt dt = {dt}: total energy spread {s:.2e} epsilon")
print(f"ratio {spreads[0.004] / spreads[0.002]:.2f} for a doubled step")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.45)
)
t = runs["argon_times"] / 1000
kin = runs["argon_kinetic"] / n
pot = runs["argon_potential"] / n
left.plot(t, 1000 * kin, color=OCHRE, lw=0.8, label="kinetic")
left.plot(
    t,
    1000 * (pot - pot[0]),
    color=OXBLOOD,
    lw=0.8,
    label=r"potential $-\,U(0)$",
)
left.plot(
    t, 1000 * (e - pot[0]), color="black", lw=1.0, label=r"total $-\,U(0)$"
)
left.set_xlabel("time / ps")
left.set_ylabel("energy per atom / meV")
left.set_xlim(0, 5)
left.legend(loc="lower right", fontsize=7)
viz.panel_tag(left, "a")
for dt, colour in ((0.002, ACCENT), (0.004, tint(ACCENT, 0.5))):
    tot = runs[f"salt_{dt}_total"]
    right.plot(
        runs[f"salt_{dt}_times"],
        tot - tot[0],
        color=colour,
        lw=0.8,
        label=rf"$\delta t = {dt}\tau$",
    )
right.set_xlabel(r"time / $\tau$")
right.set_ylabel(r"$E - E(0)$ / $\varepsilon$")
right.legend(loc="upper right", fontsize=7)
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch10_code", "run.pdf")))
