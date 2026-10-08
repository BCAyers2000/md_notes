"""Numbers quoted in Chapter 11 that no figure script prints.

Run from theory/: python scripts/ch11_constraints/numbers_chapter.py
"""

import math

import numpy as np
from ch11 import N_MOLECULES, load, model

from mdlab import (
    constraints,
    oscillators,
    rotation,
    units,
    water,
)


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


heading("11.1 one bond solved exactly: the quadratic for y")
# atoms of masses 1 (i) and 3 (j) drifted from 1 apart along x to 1.2
m_i, m_j = 1.0, 3.0
mu = 1 / (1 / m_i + 1 / m_j)
s = np.array([1.0, 0, 0])
d = np.array([1.2, 0, 0])
roots = np.sort(np.roots([s @ s, 2 * (d @ s), d @ d - 1.0]))
y = roots[-1]  # the smaller correction
print(f"reduced mass {mu}; y^2 + {2 * (d @ s):.1f} y + {d @ d - 1:.2f} = 0, "
      f"roots y = {roots.round(6)}")
print(f"small root y = {y:.4f}: atom j moves by mu y / m_j = "
      f"{mu * y / m_j:+.4f}, atom i by -mu y / m_i = {-mu * y / m_i:+.4f}")
print(f"other root: bond vector d + y s = {(d + roots[0] * s)[0]:+.2f}, "
      f"the bond turned round")
x = 0.5 * (1.0 - d @ d) / (s @ d)
print(f"first linear update (Newton): x = {x:.6f}, new length "
      f"{1.2 + x:.6f}")
x2 = 0.5 * (1.0 - (1.2 + x) ** 2) / (1.2 + x)
print(f"second: x = {x2:.3e}, new length {1.2 + x + x2:.10f}")

heading("11.1 when no move along the old bond can restore the length")
for stretch in (0.1, 0.2, 0.5):
    print(f"stretch {stretch:.0%}: the provisional bond may turn at most "
          f"{math.degrees(math.asin(1 / (1 + stretch))):.1f} degrees")

heading("11.1 toolbox: Newton's method for y^2 = 2 from y = 1")
y = 1.0
for n in range(5):
    print(f"y_{n} = {y:.13f}, error {abs(y - math.sqrt(2)):.2e}")
    y -= (y * y - 2) / (2 * y)

heading("11.1 figure: the provisional and corrected positions")
r_i, r_j = np.array([0.0, 0.0]), np.array([3.0, 0.0])
p_i, p_j = np.array([0.2, 0.9]), np.array([4.0, 1.5])
old, new = r_j - r_i, p_j - p_i
roots = np.roots([old @ old, 2 * (new @ old), new @ new - 9.0])
eta = roots.max()
m_r = 1 / (1 / 1.0 + 1 / 3.0)
f_i, f_j = p_i - eta * m_r / 1.0 * old, p_j + eta * m_r / 3.0 * old
print(f"eta {eta:.4f}; corrected i {f_i.round(4)}, j {f_j.round(4)}, "
      f"length {np.linalg.norm(f_j - f_i):.6f}")

heading("11.2 RATTLE reversed: 50 steps of 2 fs out and back")
start = load()
h0, m0 = start["cell"], start["masses"]
b0, l0 = water.constraint_bonds(N_MOLECULES)
r0, _ = constraints.shake(start["positions"], start["positions"], m0, b0,
                          l0, h0, tolerance=1e-14)  # bonds held to 1e-14
v0, _ = constraints.rattle_velocities(r0, start["velocities"], m0, b0, l0,
                                      h0, tolerance=1e-14)
out = constraints.run(model(h0), m0, r0, v0, h0, 2.0, 50, b0, l0, every=50,
                      tolerance=1e-14)
back = constraints.run(model(h0), m0, out["positions"][-1],
                       -out["velocities"][-1], h0, 2.0, 50, b0, l0,
                       every=50, tolerance=1e-14)
print(f"positions recovered to "
      f"{np.abs(back['positions'][-1] - r0).max():.1e} Å, velocities to "
      f"{np.abs(back['velocities'][-1] + v0).max():.1e} Å/fs")

heading("11.2 what the corrections cost beside the forces")
import time  # noqa: E402

w_rigid = model(h0)
f0 = w_rigid(r0)[1]
trial = r0 + 2.0 * (v0 + units.FORCE_TO_ACCEL * f0 / m0[:, None])
t0 = time.perf_counter()
for _ in range(5):
    constraints.shake(trial, r0, m0, b0, l0, h0, tolerance=1e-10)
t_shake = (time.perf_counter() - t0) / 5
t0 = time.perf_counter()
for _ in range(5):
    w_rigid(r0)
t_model = (time.perf_counter() - t0) / 5
print(f"one SHAKE to 1e-10 after a 2 fs step: {1e3 * t_shake:.1f} ms; one "
      f"call of the model {1e3 * t_model:.1f} ms; ratio "
      f"{t_shake / t_model:.2f}")

heading("11.2 velocity correction of one bond")
v, _ = constraints.rattle_velocities([[0.0, 0, 0], [1.0, 0, 0]],
                                     [[0.0, 0, 0], [0.3, 0.4, 0]],
                                     [1.0, 3.0], [[0, 1]], [1.0])
print(f"masses 1 and 3, v_j = (0.3, 0.4, 0): corrected {v.round(4).tolist()}")

heading("11.2 freedoms of 64 water molecules")
n = 64
print(f"flexible: 3N - 3 = {9 * n - 3}; O-H held: {9 * n - 3 - 2 * n}; "
      f"rigid: {9 * n - 3 - 3 * n}")
print(f"kinetic energy per freedom, (1/2) k_B (300 K) = "
      f"{0.5 * units.KB * 300:.5f} eV")
print(f"rigid box: {(6 * n - 3) * 0.5 * units.KB * 300:.3f} eV; flexible: "
      f"{(9 * n - 3) * 0.5 * units.KB * 300:.3f} eV")

heading("11.3 the box")
r, h, m, _ = water.box(4, 0.997, np.random.default_rng(11))
print(f"side {h[0, 0]:.4f} Å, half {h[0, 0] / 2:.4f} Å, "
      f"{len(m)} atoms")
print(f"number density {n / np.linalg.det(h):.5f} molecules per Å^3")
print(f"TIP3P epsilon {water.EPSILON_OO:.6f} eV, H-H {water.R_HH:.4f} Å")
lj = water.WaterModel(h, n).lj
ratio6 = (water.SIGMA_OO / 6) ** 6
print(f"Lennard-Jones at 5 Å: {lj(np.array(5.0))[0]:.2e} eV, at 6 Å "
      f"cut to 0; untouched value at 6 Å "
      f"{4 * water.EPSILON_OO * (ratio6**2 - ratio6):.2e} eV")
print(f"Coulomb of a molecule's own O and H at {water.R_OH} Å: "
      f"{units.COULOMB * water.CHARGE_O * water.CHARGE_H / water.R_OH:.3f}"
      f" eV")
print(f"Coulomb between O and H at 6 Å: "
      f"{units.COULOMB * water.CHARGE_O * water.CHARGE_H / 6:.3f} eV")

heading("11.6 the spring model of water with heavier hydrogens")
shape = water.molecule()
pairs = [(0, 1), (0, 2), (1, 2)]
hess = rotation.spring_hessian(shape, pairs,
                               [water.K_OH, water.K_OH, water.K_HH])
ordinary = np.array([water.MASS_O, water.MASS_H, water.MASS_H])
heavy = constraints.repartition_masses(ordinary, [[0, 1], [0, 2]], 3.024)
print(f"masses {ordinary} -> {heavy}, total {ordinary.sum():.3f} and "
      f"{heavy.sum():.3f}")
for label, masses in (("ordinary", ordinary), ("heavy H", heavy)):
    w2, _ = oscillators.normal_modes(hess, np.repeat(masses, 3),
                                     force_to_accel=units.FORCE_TO_ACCEL)
    w = np.sqrt(w2[-3:])
    print(f"{label}: omega {w.round(4)} rad/fs, periods "
          f"{(2 * math.pi / w).round(2)} fs, limits 2/omega "
          f"{(2 / w).round(3)} fs")
    centre = masses @ shape / masses.sum()
    moments = np.linalg.eigvalsh(rotation.inertia_tensor(masses,
                                                         shape - centre))
    print(f"   principal moments {moments.round(4)} amu Å^2")
for label, (mx, mh) in (("O-H", (15.999, 1.008)), ("O-H heavy", (11.967,
                                                                 3.024))):
    mr = oscillators.reduced_mass(mx, mh)
    print(f"{label} reduced mass {mr:.4f} amu")
ratio = (oscillators.reduced_mass(11.967, 3.024)
         / oscillators.reduced_mass(15.999, 1.008))
print(f"ratio of O-H reduced masses {ratio:.4f}, square root "
      f"{math.sqrt(ratio):.4f}")
# the bend alone with the O-H bonds held, as the O-H springs grow stiff
for label, masses in (("ordinary", ordinary), ("heavy H", heavy)):
    stiff = rotation.spring_hessian(shape, pairs, [1e7, 1e7, water.K_HH])
    w2, _ = oscillators.normal_modes(stiff, np.repeat(masses, 3),
                                     force_to_accel=units.FORCE_TO_ACCEL)
    print(f"{label}: bend with the O-H bonds held, period "
          f"{2 * math.pi / math.sqrt(w2[6]):.2f} fs")
print("C-H for comparison: reduced mass "
      f"{oscillators.reduced_mass(12.011, 1.008):.4f} -> "
      f"{oscillators.reduced_mass(12.011 - 2.016, 3.024):.4f} amu")

heading("11.7 factors")
print("see fig_step.py (constraints, heavier H) and fig_respa.py (RESPA)")
print(f"thirty times the flexible step at 1%: 30 x 0.41 = {30 * 0.41:.1f} fs")
print(f"a TrajCast stride of 5 fs is {5 / 0.5:.0f} steps of 0.5 fs; the "
      f"velocity Verlet limit of the O-H stretch (8.88 fs) is "
      f"{2 / (2 * math.pi / 8.88):.2f} fs")

heading("exercises")
print(f"11.1 methanol: {18 - 4} freedoms per molecule; 100 molecules with "
      f"drift removed {100 * 18 - 100 * 4 - 3}")
r_old, r_new = np.array([1.5, 0, 0]), np.array([1.6, 0.3, 0])
roots = np.sort(np.roots([r_old @ r_old, 2 * (r_new @ r_old),
                          r_new @ r_new - 1.5**2]))
eta = roots[-1]
m_r = 1 / (1 / 2.0 + 1 / 5.0)
move_j, move_i = eta * m_r / 5.0 * r_old, -eta * m_r / 2.0 * r_old
print(f"11.2 roots {roots.round(5)}, eta {eta:.5f}, m_r {m_r:.4f}, atom j "
      f"moves {move_j[0]:+.5f}, atom i {move_i[0]:+.5f}; new bond length "
      f"{np.linalg.norm(r_new + move_j - move_i):.6f}")
y = 2.0
for n in range(1, 4):
    y = y - (y * y - 3) / (2 * y)
    print(f"11.3 y_{n} = {y:.10f}, error {y - math.sqrt(3):.2e}")
m_o, m_h = 16.0, 1.0
bond = np.array([0.96, 0.0, 0.0])
v, _ = constraints.rattle_velocities([[0.0, 0, 0], bond],
                                     [[0.001, 0, 0], [0.02, 0.01, 0]],
                                     [m_o, m_h], [[0, 1]], [0.96],
                                     tolerance=1e-15)
zeta = -(0.02 - 0.001) * 0.96 / 0.96**2
print(f"11.7 zeta {zeta:.6f}; corrected v_O {v[0].round(7)}, v_H "
      f"{v[1].round(7)}; momentum x before {m_o * 0.001 + m_h * 0.02:.4f}, "
      f"after {m_o * v[0, 0] + m_h * v[1, 0]:.4f}")
k_before = 0.5 * (m_o * 0.001**2 + m_h * (0.02**2 + 0.01**2))
k_after = 0.5 * (m_o * v[0] @ v[0] + m_h * v[1] @ v[1])
mr = m_o * m_h / (m_o + m_h)
print(f"   kinetic energy (amu Å²/fs²) {k_before:.4e} -> {k_after:.4e}, "
      f"loss {k_before - k_after:.4e}, (1/2) m_r (v_ij.e)^2 "
      f"{0.5 * mr * 0.019**2:.4e}")
print(f"11.8 216 rigid molecules: {6 * 216 - 3} freedoms, kinetic energy "
      f"{(6 * 216 - 3) * 0.01293:.3f} eV at 12.93 meV each")
for label, calls_ps, fast_ps in (("velocity Verlet 0.41 fs", 1000 / 0.41,
                                  1000 / 0.41),
                                 ("RESPA 0.72 fs, 3 inner",
                                  1000 / 0.72, 3 * 1000 / 0.72)):
    seconds = 1000 * (calls_ps * 24e-3 + fast_ps * 0.04e-3)
    print(f"11.10 {label}: {seconds:.0f} s per ns at 24 ms and 0.04 ms "
          f"per call")
for b in (0.09,):
    print(f"11.13 b = {b}: band below T/2 from {(1 - b) / 2:.3f} to 0.5; "
          f"below T from {1 - b:.3f} to 1")
print(f"11.5 check: b = 0.01, period 9 fs: band from {4.5 * (1 - 0.01):.3f} "
      f"to 4.5 fs, width {0.5 * 0.01 * 9:.3f} fs")
for label, m_heavy, m_light in (("O-H", 15.999, 1.008),
                                ("O-H, H doubled", 15.999 - 2 * 1.008,
                                 2.016)):
    m_r = oscillators.reduced_mass(m_heavy, m_light)
    print(f"11.15 {label}: m_r {m_r:.4f}")
ratio = (oscillators.reduced_mass(15.999 - 2 * 1.008, 2.016)
         / oscillators.reduced_mass(15.999, 1.008))
print(f"   ratio {ratio:.4f}, slows by {math.sqrt(ratio):.4f}; O mass "
      f"{15.999 - 2 * 1.008:.3f}")
shape = water.molecule()
o_to_h = shape[1] - shape[0]
half = math.radians(water.ANGLE_HOH / 2)
for label, masses in (("ordinary", (15.999, 1.008)),
                      ("heavy H", (11.967, 3.024))):
    along = 2 * masses[1] * water.R_OH * math.cos(half) / 18.015
    print(f"11.16 {label}: centre of mass {along:.5f} Å from O along the "
          f"bisector")
p = np.array([0.2817, 0.2936, 0.4033])
mass = 64 * 18.015
speed = np.linalg.norm(p) / mass
print(f"11.18 |P| {np.linalg.norm(p):.4f}, M {mass:.2f} amu, speed "
      f"{1000 * speed:.4f} Å/ps, in 1 ns {1e6 * speed:.0f} Å, kinetic "
      f"energy {0.5 * units.MV2_TO_EV * (p @ p) / mass:.5f} eV")
ase_fs = 1 / math.sqrt(units.FORCE_TO_ACCEL)
print(f"11.19 ASE time unit {ase_fs:.4f} fs; a step of 2 is "
      f"{2 * ase_fs:.2f} fs; omega dt for the O-H stretch "
      f"{2 * math.pi / 8.88 * 2 * ase_fs:.2f}")
