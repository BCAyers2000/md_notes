"""Numbers quoted in Chapter 10 that no figure script prints.

Run from theory/: python scripts/ch10_code/numbers_chapter.py
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar

from mdlab import cell, ewald, potentials


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


heading("10.1 atoms at the surface of a cube of n x n x n")
for n in (5, 10, 22, 100, 1000):
    print(
        f"n = {n}: N = {n**3:.0e}, surface fraction "
        f"{1 - (1 - 2 / n) ** 3:.4f}, 6/n = {6 / n:.4f}"
    )

heading("10.5 the fcc lattice of the Lennard-Jones model")
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def fcc_energy_per_atom(a, pair, n=6):
    grid = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)],
        float,
    )
    s = (grid[:, None, :] + FCC[None, :, :]).reshape(-1, 3) / n
    h = n * a * np.eye(3)
    r = cell.to_cartesian(s, h)
    d = cell.minimum_image(r[1:] - r[0], h)
    phi, _ = pair(np.linalg.norm(d, axis=1))
    return 0.5 * float(np.sum(phi))


lj = potentials.lennard_jones
switched = potentials.with_cutoff(lj, 2.5, "switch", r_switch=2.0)
best = minimize_scalar(
    lambda a: fcc_energy_per_atom(a, switched),
    bounds=(1.45, 1.65),
    method="bounded",
    options={"xatol": 1e-9},
)
a0 = best.x
print(
    f"switched 2.0-2.5 sigma: a0 = {a0:.5f} sigma = {a0 * 3.4:.4f} Å for "
    f"sigma = 3.4 Å; nearest distance {a0 / math.sqrt(2):.5f} sigma; "
    f"energy per atom {best.fun:.4f} epsilon"
)
far = minimize_scalar(
    lambda a: fcc_energy_per_atom(a, lj, n=10),
    bounds=(1.45, 1.65),
    method="bounded",
    options={"xatol": 1e-9},
)
print(
    f"no cutoff (to half of a 10-cell box): a0 = {far.x:.5f} sigma, "
    f"nearest {far.x / math.sqrt(2):.5f} sigma, energy {far.fun:.4f}"
)

heading("10.4 neighbours within the cutoff")
rho = 0.8
for r_c in (2.5, 2.5 + 0.3):
    print(
        f"rho = {rho} sigma^-3, r = {r_c}: (4/3) pi r^3 rho = "
        f"{4 / 3 * math.pi * r_c**3 * rho:.1f} neighbours"
    )

heading("10.3 rock salt in three cells, and forces")
salt = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
pos = np.vstack([salt, salt + [1.0, 0.0, 0.0]])
chg = np.array([1.0] * 4 + [-1.0] * 4)
e1 = ewald.ewald_energy_forces(chg, pos, 2 * np.eye(3), coulomb=1.0)[0] / 4
shifts = [
    np.array([i, j, k]) for i in range(2) for j in range(2) for k in range(2)
]
big = np.vstack([pos + 2 * shift for shift in shifts])
e8 = (
    ewald.ewald_energy_forces(
        np.tile(chg, 8), big, 4 * np.eye(3), coulomb=1.0
    )[0]
    / 32
)
prim = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float).T
ep = ewald.ewald_energy_forces(
    [1.0, -1.0], [[0, 0, 0], [1, 0, 0]], prim, coulomb=1.0
)[0]
print(
    f"per ion pair: cubic cell {-e1:.10f}, 2x2x2 supercell {-e8:.10f}, "
    f"primitive cell {-ep:.10f}"
)
rng = np.random.default_rng(3)
r6 = rng.uniform(0, 5, size=(6, 3))
q6 = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
h6 = np.array([[5, 0, 0], [1.2, 4.5, 0], [0.3, 0.7, 5.5]], float).T
energy6, forces6, _ = ewald.ewald_energy_forces(q6, r6, h6)
numeric6 = potentials.finite_difference_forces(
    lambda x: ewald.ewald_energy_forces(q6, x, h6)[0], r6, 1e-5
)
print(
    f"skewed cell of six charges: forces differ from differences by "
    f"{np.abs(forces6 - numeric6).max():.1e} eV/Å; largest force "
    f"{np.abs(forces6).max():.2f} eV/Å; total force "
    f"{np.abs(forces6.sum(0)).max():.1e}"
)

heading("10.3 a charged cell needs a background")
rng = np.random.default_rng(4)
r = rng.uniform(0, 6, size=(5, 3))
q = np.array([1.0, 1.0, -1.0, 0.5, 0.2])
for a in (0.5, 0.8, 1.2):
    e, _, parts = ewald.ewald_energy_forces(q, r, 6 * np.eye(3), alpha=a)
    print(
        f"alpha = {a}: with background {e:.10f} eV, without "
        f"{e - parts['background']:.6f} eV"
    )

heading("10.6 the loop against ASE's VelocityVerlet")
from ase import Atoms  # noqa: E402
from ase.calculators.lj import LennardJones  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402
from ase.units import fs as ase_fs  # noqa: E402

from mdlab import md, units  # noqa: E402

# ASE's femtosecond comes from CODATA 2014, mdlab's from CODATA 2022 (via
# SciPy). Inside ASE no constant enters the dynamics, so converting with
# mdlab's femtosecond makes the two runs comparable to round-off.
fs = math.sqrt(units.FORCE_TO_ACCEL)
print(f"ASE's fs / mdlab's fs - 1 = {ase_fs / fs - 1:.1e}")

eps, sig, rc = 0.01034, 3.4, 8.5
grid = np.array(
    [[i, j, k] for i in range(4) for j in range(4) for k in range(4)], float
)
s = (grid[:, None, :] + FCC[None, :, :]).reshape(-1, 3) / 4
h = 4 * 5.26 * np.eye(3)
r = cell.to_cartesian(s, h)
m = np.full(len(r), 39.948)
v = md.starting_velocities(m, 0.02 * len(r), np.random.default_rng(8))
pair = potentials.with_cutoff(
    lambda x: potentials.lennard_jones(x, eps, sig), rc, "shift"
)
model = md.PairModel(pair, h, rc, 1.0)
out = md.run(model, m, r, v, h, 5.0, 200, every=200)
atoms = Atoms("Ar" * len(r), positions=r, cell=h.T, pbc=True)
atoms.set_masses(m)
atoms.set_velocities(v / fs)
atoms.calc = LennardJones(epsilon=eps, sigma=sig, rc=rc, smooth=False)
e0 = atoms.get_potential_energy()
plain = atoms.copy()
plain.set_velocities(v / ase_fs)
plain.calc = LennardJones(epsilon=eps, sigma=sig, rc=rc, smooth=False)
VelocityVerlet(plain, timestep=5.0 * ase_fs).run(200)
print(
    f"converted with ASE's own fs: positions differ by "
    f"{np.abs(plain.get_positions() - out['positions'][-1]).max():.1e} Å"
)
VelocityVerlet(atoms, timestep=5.0 * fs).run(200)
print(
    f"{len(r)} atoms, 200 steps of 5 fs: starting energies differ by "
    f"{abs(e0 - out['potential'][0]):.1e} eV; final positions by "
    f"{np.abs(atoms.get_positions() - out['positions'][-1]).max():.1e} Å, "
    f"velocities by "
    f"{np.abs(atoms.get_velocities() * fs - out['velocities'][-1]).max():.1e}"
    f" Å/fs"
)

heading("10.5 the share of kinetic energy left, against the energy given")
grid5 = np.array(
    [[i, j, k] for i in range(5) for j in range(5) for k in range(5)], float
)
s5 = (grid5[:, None, :] + FCC[None, :, :]).reshape(-1, 3) / 5
h5 = 5 * 1.54916 * 3.4 * np.eye(3)
r5 = cell.to_cartesian(s5, h5)
m5 = np.full(len(r5), 39.948)
switched_ar = potentials.with_cutoff(
    lambda x: potentials.lennard_jones(x, 0.01034, 3.4),
    8.5,
    "switch",
    r_switch=6.8,
)
for given in (0.002, 0.005, 0.02):
    v5 = md.starting_velocities(m5, given * len(r5), np.random.default_rng(11))
    run5 = md.run(
        md.PairModel(switched_ar, h5, 8.5, 1.0),
        m5,
        r5,
        v5,
        h5,
        10.0,
        2000,
        every=2,
    )
    share = run5["kinetic"][500:].mean() / (given * len(r5))
    print(
        f"{given} eV per atom: mean kinetic energy over the last 10 ps is "
        f"{share:.4f} of the starting kinetic energy"
    )

heading("checks")
print(
    f"10.3: r_c = {math.sqrt(-math.log(1e-8)) / 0.3:.2f} Å, G_c = "
    f"{2 * 0.3 * math.sqrt(-math.log(1e-8)):.3f} /Å"
)
print(
    f"10.4: pairs per atom within 2.8 sigma at 0.8: "
    f"{0.5 * 4 / 3 * math.pi * 2.8**3 * 0.8:.1f}"
)
print(f"10.5: 5x5x5 fcc: {4 * 125} atoms, side {5 * 5.267:.2f} Å")
print(
    f"10.7: 1 ns, 1e4 atoms, 0.2 µs per atom per call: "
    f"{1e6 * 1e4 * 0.2e-6:.0f} s = {1e6 * 1e4 * 0.2e-6 / 60:.0f} min"
)

heading("exercises")
n = 1
while 1 - (1 - 2 / n) ** 3 >= 0.01:
    n += 1
print(f"surface fraction first below 1% at n = {n}, N = {n**3:.2e}")
a_g, c_g = 2.464, 6.711
print(
    f"graphite: volume {math.sqrt(3) / 2 * a_g**2 * c_g:.2f} Å³, widths "
    f"{a_g * math.sqrt(3) / 2:.3f}, {c_g:.3f} Å; |b1| = "
    f"{4 * math.pi / (a_g * math.sqrt(3)):.4f} /Å, |b3| = "
    f"{2 * math.pi / c_g:.4f} /Å"
)
h_prim = 5.267 * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]).T
print(
    f"fcc primitive cell (a0 = 5.267 Å): volume "
    f"{cell.cell_volume(h_prim):.3f} Å³ = a0³/4 = {5.267**3 / 4:.3f}; widths "
    f"{cell.perpendicular_widths(h_prim).round(4)} Å = a0/√3 = "
    f"{5.267 / math.sqrt(3):.4f}; largest cutoff "
    f"{5.267 / (2 * math.sqrt(3)):.3f} Å"
)
reach = math.sqrt(-math.log(1e-6))
alpha = reach / 15
g_c = 2 * alpha * reach
m_max = int(g_c * 30 / (2 * math.pi)) + 1
grid_m = np.array(
    [
        [i, j, k]
        for i in range(-m_max, m_max + 1)
        for j in range(-m_max, m_max + 1)
        for k in range(-m_max, m_max + 1)
    ]
)
g_len = 2 * math.pi / 30 * np.linalg.norm(grid_m, axis=1)
count = int(np.sum((g_len > 0) & (g_len <= g_c)))
estimate = 4 / 3 * math.pi * g_c**3 / ((2 * math.pi) ** 3 / 30**3)
print(
    f"Ewald, 30 Å cube, eps 1e-6, r_c = 15 Å: alpha = {alpha:.4f} /Å, "
    f"G_c = {g_c:.3f} /Å, {count} waves (sphere estimate {estimate:.0f})"
)
print(
    f"water at 0.0334 per Å³ within 10 Å: "
    f"{4 / 3 * math.pi * 10**3 * 0.0334:.1f} molecules"
)
for side in (40, 30, 25):
    k = math.floor(side / 9.5)
    print(
        f"cube of {side} Å, r_c + skin = 9.5 Å: {k} small cells a side, "
        f"{k**3} in all"
    )
from mdlab.units import MV2_TO_EV  # noqa: E402

v_rms = math.sqrt(2 * 0.0104 / (39.948 * MV2_TO_EV))
print(
    f"argon at 0.0104 eV per atom: rms speed {v_rms:.5f} Å/fs, "
    f"{v_rms * 10:.4f} Å per 10 fs step, {0.5 / (v_rms * 10):.0f} steps to "
    f"0.5 Å; measured {2000 / 187:.1f} steps between builds"
)
v_ase = math.sqrt(2 * 0.02 / (39.948 * MV2_TO_EV))
print(
    f"ASE comparison: rms speed {v_ase:.4f} Å/fs; 1000 fs x 4.6e-9 = "
    f"{1000 * 4.6e-9:.1e} fs, times the speed {v_ase * 1000 * 4.6e-9:.1e} Å"
)

heading("10.6 how the difference from ASE grows with ASE's own fs")
for steps in (2, 10, 50, 200):
    run_s = md.run(model, m, r, v, h, 5.0, steps, every=steps)
    probe = Atoms("Ar" * len(r), positions=r, cell=h.T, pbc=True)
    probe.set_masses(m)
    probe.set_velocities(v / ase_fs)
    probe.calc = LennardJones(epsilon=eps, sigma=sig, rc=rc, smooth=False)
    VelocityVerlet(probe, timestep=5.0 * ase_fs).run(steps)
    gap = np.abs(probe.get_positions() - run_s["positions"][-1]).max()
    print(f"after {steps} steps: positions differ by {gap:.1e} Å")

heading("10.7 where a step spends its time (cProfile)")
import cProfile  # noqa: E402
import pstats  # noqa: E402

rng_p = np.random.default_rng(15)
k_p = 16
grid_p = np.array(
    [[a, b, c] for a in range(k_p) for b in range(k_p) for c in range(k_p)],
    float,
)[:4000]
side_p = (4000 / 0.8) ** (1 / 3)
r_p = grid_p / k_p * side_p + rng_p.normal(scale=0.05, size=(4000, 3))
h_p = side_p * np.eye(3)
lj_p = potentials.with_cutoff(potentials.lennard_jones, 2.5, "shift")
model_p = md.PairModel(lj_p, h_p, 2.5, 0.3)
v_p = md.starting_velocities(
    np.ones(4000), 0.5 * 4000, np.random.default_rng(16), 1.0
)
profile = cProfile.Profile()
profile.enable()
md.run(
    model_p,
    np.ones(4000),
    r_p,
    v_p,
    h_p,
    0.002,
    200,
    every=200,
    force_to_accel=1.0,
)
profile.disable()
stats = pstats.Stats(profile)
total = stats.total_tt
for key, row in stats.stats.items():
    name = key[2]
    if "'at' of 'numpy.ufunc'" in name:
        print(f"np.add.at: share {row[3] / total:.2f}")
    if name in ("pairs", "build", "_within", "minimum_image"):
        print(f"{name}: cumulative share {row[3] / total:.2f}")
        if name == "_within":
            for caller, crow in row[4].items():
                print(
                    f"   _within called from {caller[2]}: "
                    f"{crow[3] / total:.2f}"
                )
print(f"list builds in 200 steps: {model_p.neighbours.builds}")
