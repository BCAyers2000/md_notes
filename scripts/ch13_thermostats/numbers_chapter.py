"""Numbers quoted in Chapter 13 that no figure script prints.

Run from scripts/ch13_thermostats/: python numbers_chapter.py
"""

import math

import numpy as np
from ch13 import (
    BARRIER,
    LI_MASS,
    N_FREE,
    SPACING,
    T_LI,
    T_LIQUID,
    cell_grid,
    tst_rate,
)

from mdlab import energy, thermostats, units
from mdlab.thermostats import Langevin, NoseHooverChain

REDUCED = dict(kb=1.0, mv2_to_energy=1.0)


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


heading("13.2 one Berendsen step")
lam2 = 1 + 10 / 100 * (135 / 150 - 1)
print(f"dt 10 fs, tau 100 fs, T 150 K, target 135 K: lambda^2 = {lam2:.4f}, "
      f"lambda = {math.sqrt(lam2):.5f}")
print(f"k_B T at 135 K = {units.KB * 135:.5f} eV; N_f k_B T / 2 = "
      f"{0.5 * N_FREE * units.KB * 135:.4f} eV")

heading("13.4 the Ornstein-Uhlenbeck step")
for gamma_ps in (0.1, 1.0, 10.0):
    c = math.exp(-gamma_ps / 1000 * 10.0)
    print(f"gamma {gamma_ps:g}/ps, dt 10 fs: c = e^(-gamma dt) = {c:.5f}, "
          f"sqrt(1 - c^2) = {math.sqrt(1 - c * c):.5f}")
sigma_ar = math.sqrt(units.KB * T_LIQUID / (39.948 * units.MV2_TO_EV))
print(f"argon at 135 K: sqrt(k_BT/m) = {sigma_ar:.6f} Å/fs")
rng = np.random.default_rng(0)
v = np.zeros(200000)
c = math.exp(-0.5 * 0.1)
for _ in range(400):
    v = c * v + math.sqrt(1 - c * c) * rng.standard_normal(v.size)
print(f"OU process (gamma 0.5, dt 0.1, k_BT/m 1) from rest after 40 units:"
      f" variance {v.var():.4f} (1 exactly in the long run)")


def oscillator(r):
    return 0.5 * np.einsum("...ix,...ix->...", r, r), -r


for dt in (0.5, 1.0, 1.5):
    r0 = np.random.default_rng(1).standard_normal((4000, 1, 1))
    out = thermostats.run(oscillator, np.ones(1), r0, np.zeros_like(r0), dt,
                          int(400 / dt), Langevin(1.0, 1.0, **REDUCED),
                          np.random.default_rng(2), every=1,
                          keep=("positions", "velocities"),
                          force_to_accel=1.0)
    x = out["positions"][int(100 / dt):].ravel()
    p = out["velocities"][int(100 / dt):].ravel()
    print(f"BAOAB on an oscillator (period 2 pi), dt {dt}: <x^2> "
          f"{np.mean(x * x):.4f} (exact 1), <v^2> {np.mean(p * p):.4f}, "
          f"1 - dt^2/4 = {1 - dt * dt / 4:.4f}")

heading("13.4 the transition-state rate")
kt = units.KB * T_LI
speed = math.sqrt(kt / (2 * math.pi * LI_MASS * units.MV2_TO_EV))
pts, area = cell_grid(600)
u, _ = energy.hexagonal_surface(pts)
inside = np.sum(np.exp(-u / kt)) * area
cell = math.sqrt(3) / 2 * SPACING**2
print(f"T {T_LI:g} K: k_BT {kt:.5f} eV; mean outward speed sqrt(k_BT/2pi m)"
      f" = {speed:.5f} Å/fs; cell area {cell:.4f} Å^2;"
      f" integral of e^(-U/k_BT) over it {inside:.4f} Å^2; edge length "
      f"{SPACING / math.sqrt(3):.4f} Å; rate {tst_rate(T_LI) * 1000:.3f}/ps")
t_edge = (np.arange(4000) + 0.5) / 4000 - 0.5
edge_line = np.array([SPACING / 2, 0.0]) + np.outer(
    t_edge * SPACING / math.sqrt(3), [0.0, 1.0])
u_edge, _ = energy.hexagonal_surface(edge_line)
edge = np.sum(np.exp(-u_edge / kt)) * SPACING / math.sqrt(3) / 4000
print(f"integral of e^(-U/k_BT) round the six edges: {6 * edge:.4f} Å; "
      f"rate = {speed:.6f} x {6 * edge:.4f}/{inside:.4f} per fs")
omega = 2 * math.pi / 170.3
print(f"BAOAB for lithium, steps of 5 fs: 1 - omega^2 dt^2/4 = "
      f"{1 - omega**2 * 25 / 4:.4f}")
print(f"Boltzmann factor of the barrier at {T_LI:g} K: "
      f"{math.exp(-BARRIER / kt):.4f}")
for t in (300.0, 600.0):
    rate = tst_rate(t) * 1e15
    print(f"transition-state rate at {t:g} K: {rate:.3e} per second, a hop "
          f"every {1e12 / rate:.3g} ps")

heading("13.5 Nosé-Hoover")
q = N_FREE * units.KB * T_LIQUID * 100.0**2
print(f"Q1 for 256 argon atoms, tau 100 fs: N_f k_BT tau^2 = {q:.4g} eV fs^2")
r0 = np.random.default_rng(4).standard_normal((1, 1, 1))
nh = NoseHooverChain(1.0, 1.0, 1, chain=1, **REDUCED)
out = thermostats.run(oscillator, np.ones(1), r0, np.zeros_like(r0), 0.05,
                      20000, nh, every=1, keep=(), force_to_accel=1.0)
extended = out["potential"][-1] + out["kinetic"][-1] + nh.energy()
start = out["potential"][0] + out["kinetic"][0]
effective = (out["potential"] + out["kinetic"] - out["heat"])[:, 0]
print(f"one oscillator under Nosé-Hoover, 1000 units: extended energy "
      f"{float(extended[0]):.6f} at the end against {float(start[0]):.6f} at "
      f"the start; at every step the effective energy stays within "
      f"{np.abs(effective - effective[0]).max():.1e} of its start")

heading("13.6 CSVR and TrajCast")
for system, stride, factor in (("default", 30.0, 100), ("water", 5.0, 10),
                               ("paracetamol", 7.0, 10),
                               ("quartz", 30.0, 2.5)):
    tau = factor * stride
    print(f"{system}: stride {stride:g} fs, tau = {factor:g} strides = "
          f"{tau:g} fs, c1 = e^(-Delta t/tau) = {math.exp(-1 / factor):.4f}")
print(f"stride 30 fs, c1 = 0.5: tau = 30/ln 2 = {30 / math.log(2):.1f} fs")
k = 0.5 * N_FREE * units.KB * 150.0
target = 0.5 * N_FREE * units.KB * T_LIQUID
c1 = math.exp(-10.0 / 100.0)
c2 = (1 - c1) * target / (k * N_FREE)
print(f"one CSVR step, K at 150 K, target 135 K, tau 100 fs, dt 10 fs: "
      f"c1 {c1:.5f}, c2 {c2:.3e}; with R1 = 0 and the sum of the other "
      f"764 squares at its mean 764, alpha^2 = {c1 + c2 * 764:.5f}")

heading("13.4 the one-dimensional estimate against the surface")


def curvature(point, direction, h=1e-4):
    d = np.array(direction, float)
    u = [energy.hexagonal_surface(np.array(point) + s * h * d)[0]
         for s in (-1, 0, 1)]
    return (u[0] - 2 * u[1] + u[2]) / h**2


hollow_k = curvature([0.0, 0.0], [0.0, 1.0])
bridge_k = curvature([SPACING / 2, 0.0], [0.0, 1.0])
one_d = math.exp(-0.3 / (units.KB * 300)) / 170.3e-15
surface_rate = tst_rate(300.0) * 1e15
print(f"curvature across the path: hollow {hollow_k:.4f}, bridge "
      f"{bridge_k:.4f} eV/Å^2, ratio {bridge_k / hollow_k:.4f}; one-"
      f"dimensional rate at 300 K {one_d:.3e}/s, surface {surface_rate:.3e}/s,"
      f" ratio {surface_rate / one_d:.2f}; 6 sqrt(3) = {6 * math.sqrt(3):.2f}")
