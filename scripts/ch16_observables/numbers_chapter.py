"""The numbers of Chapter 16 that no figure script prints.

Chapter 11's water as a diffusion coefficient; the answers to the checks
and the exercises, from the data each states.
"""

import math
import sys
from pathlib import Path

import numpy as np
from ch16 import ch12, ch15
from scipy.integrate import quad

from mdlab import units
from mdlab.analysis import structure, transport

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch11_constraints"))
import ch11  # noqa: E402

KB = units.KB

# Chapter 11's rigid water: 64 molecules, 10 ps, frames every 50 fs.
run = np.load(ch11.DATA / "runs" / "rigid_healthy.npz")
frames, m = run["frames"], run["masses"]
step = float(run["frame_times"][1] - run["frame_times"][0])
temperature = 2 * run["kinetic"].mean() / ((3 * 64 * 2 - 3) * KB)
oxygen = np.arange(0, len(m), 3)
lags = np.arange(101)
t = step * lags
each = []
for o in oxygen:
    msd = transport.msd(frames, 100, masses=m, select=[o]).sum(1)
    each.append(transport.diffusion_coefficient(t, msd, 1000.0, 5000.0))
each = np.array(each)
all_o = transport.msd(frames, 100, masses=m, select=oxygen).sum(1)
d_water = transport.diffusion_coefficient(t, all_o, 1000.0, 5000.0)
print(f"Chapter 11's water at {temperature:.0f} K (6 rigid degrees of freedom "
      f"per molecule, less 3): D of the oxygen atoms over 1-5 ps = "
      f"{d_water * 1e4:.3f}e-4 Å²/fs; mean of the 64 molecules "
      f"{each.mean() * 1e4:.3f}e-4 ± {each.std(ddof=1) / 8 * 1e4:.3f}e-4; "
      f"in cm²/s {d_water * 0.1:.2e}")

print("checks")
n = 500
print(f"16.1: g(r) far away with ASE's normalisation for N = {n}: "
      f"1 − 1/N = {1 - 1 / n:.4f}")
print("16.3: key 17 342 with N = 1000: i = 17, j = 342; 342 017 would be "
      "i = 342, j = 17, not a key, since every key has i < j")
nyquist = 1 / (2 * 7.0)  # per fs, for a stride of 7 fs
print(f"Nyquist frequency for Δt = 7 fs: {nyquist * 1000:.1f} THz, "
      f"{nyquist / units.C_CM_PER_FS:.0f} cm⁻¹")

# The chain's torsion, Σ (κ_n/2)[1 + (−1)^{n+1} cos nψ].
kappa = (0.06, -0.01, 0.12)
grid = np.radians(np.linspace(0, 180, 180001))
u = sum(0.5 * k * (1 + (-1) ** (n + 1) * np.cos(n * grid))
        for n, k in enumerate(kappa, start=1))
for lo, hi, kind in ((30, 90, "gauche well"), (90, 150, "barrier")):
    window = (grid >= math.radians(lo)) & (grid <= math.radians(hi))
    k = np.argmin(u[window]) if kind == "gauche well" else np.argmax(
        u[window])
    print(f"torsion {kind} at ±{np.degrees(grid[window][k]):.1f}°: "
          f"{u[window][k] * 1000:.1f} meV")
print(f"torsion at 0: {u[0] * 1000:.1f} meV, at 180°: {u[-1] * 1000:.1f}")
print(f"8.0 changes of well per ns: {1000 / 8.0:.0f} ps between changes")
print(f"ten crossings: the ratio of two wells' rates to 2/√10 = "
      f"{2 / math.sqrt(10):.2f}, so F to about two thirds of k_BT")

print("exercises")
# shell volume: first term misses by δr/r
for r, dr in ((1.0, 0.05), (5.0, 0.05)):
    exact = 4 / 3 * math.pi * ((r + dr) ** 3 - r**3)
    first = 4 * math.pi * r * r * dr
    print(f"shell at r = {r} Å, δr = {dr}: exact {exact:.5f}, first term "
          f"{first:.5f}, short by {1 - first / exact:.4f} (δr/r = "
          f"{dr / r:.3f})")
for n_atoms in (256, 2048):
    print(f"g far away for N = {n_atoms}: {1 - 1 / n_atoms:.5f}")
# mean-field energy with g = 0 inside σ and 1 beyond, full LJ
eps, sig, rho_s3 = 0.01034, 3.4, 0.8
u_mf = -16 / 9 * math.pi * rho_s3 * eps
print(f"g = step at σ: U/N = −(16/9)πρσ³ε = {u_mf * 1000:.2f} meV, "
      f"against the run's −50.09 meV")
# The same with the run's switched potential, and the run's own g(r) with
# the full potential, the part beyond L/2 taken with g = 1.
liquid = np.load(ch15.RUNS / "long_csvr.npz")
h_liq = liquid["cell"]
rho_liq = 256 / np.linalg.det(h_liq)
switched = ch12.model(h_liq).pair


def full_lj(r):
    return 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)


u_mf_switched = 0.5 * rho_liq * quad(
    lambda r: float(switched(np.array([r]))[0][0]) * 4 * np.pi * r * r,
    sig, ch12.R_CUT, limit=200)[0]
x_liq, g_liq = structure.rdf(liquid["positions"][1::4].astype(float), h_liq,
                             h_liq[0, 0] / 2 - 1e-6, 1163)
width = x_liq[1] - x_liq[0]
edges = np.append(x_liq - 0.5 * width, x_liq[-1] + 0.5 * width)
shells = structure.shell_volumes(edges)
tail = quad(lambda r: full_lj(r) * 4 * np.pi * r * r, edges[-1], np.inf)[0]
u_full = 0.5 * rho_liq * (np.sum(g_liq * full_lj(x_liq) * shells) + tail)
print(f"mean field with the switched potential {u_mf_switched * 1000:.2f} meV;"
      f" the run's g(r) with the full potential {u_full * 1000:.2f} meV, "
      f"{(u_mf - u_full) * 1000:.2f} meV below the full mean field")
# S(0) for a stated liquid
kappa, rho, temp = 0.5, 0.03, 300.0
print(f"S(0) for κ_T = {kappa} GPa⁻¹, ρ = {rho} Å⁻³, {temp:g} K: "
      f"{rho * KB * temp * kappa * units.EV_PER_A3_TO_GPA:.4f}")
# flicker of a single cutoff
period = 60.0
print(f"a vibration crossing one cutoff twice a period of {period:g} fs: "
      f"{2 * 1000 / period:.1f} events per ps")
# Poisson
for n_events in (25, 100, 400, 2500):
    print(f"{n_events} events: rate known to {1 / math.sqrt(n_events):.3f}")
print(f"2% needs {1 / 0.02 ** 2:.0f} events")
# random walks
print("1D: hops ±a at the rate k_h: D = k_h a²/2; 3D cubic: D = k_h a²/6")
# correlation factor of a walk whose steps repeat their direction, against
# the model surface's memory of direction at 1 ps⁻¹ (fig_layers.py)
memory = 3.46
c = (memory - 1) / (memory + 1)
print(f"(1 + c)/(1 − c) = {memory}: c = {c:.3f}")
# water ballistic crossover and VDOS at zero
d_w, m_w, t_w = 5.243e-4, 18.015, 304.0
kt_w = KB * t_w
cross = 2 * d_w * m_w * units.MV2_TO_EV / kt_w
print(f"water: 2Dm/k_BT = {cross:.0f} fs with m = 18.015")
m_o = 15.999
zero = 12 * m_o * units.MV2_TO_EV * d_w / kt_w  # fs
print(f"oxygen's 𝒟(0) = 12mD/k_BT = {zero:.1f} fs = {zero / 1000:.4f} per "
      f"THz")
# exponential VACF
print("C = C0 e^(−t/τ): MSD = 2C0τ[t − τ(1 − e^(−t/τ))], D = C0τ/3")
# Green-Kubo estimate of the viscosity from Table 15.1
volume = 256 / 0.02035
kt = KB * 135.0
sigma_xy = 0.0121 / units.EV_PER_A3_TO_GPA  # eV/Å³
tau = 180.0
pa_s = units.ELEMENTARY_CHARGE / units.ANGSTROM**3 * units.FEMTOSECOND
print(f"η ≈ (V/k_BT)σ²τ_int = {volume / kt * sigma_xy**2 * tau * pa_s:.3e} "
      f"Pa s (V = {volume:.0f} Å³)")
# Nernst-Einstein for the salt
v_salt = 13.4**3
t_salt = 1487.0
ne = units.ELEMENTARY_CHARGE * 32 * (5.19e-4 + 5.03e-4) * 1e-5 / (
    v_salt * 1e-30 * KB * t_salt)
print(f"salt σ_NE from the mean D's at {t_salt:g} K: {ne:.0f} S/m")
# trace of the guests' tensor
print(f"guests: a third of the trace {(1.16 + 1.02 + 0.0) / 3:.2f}e-4 Å²/fs")
# FFT count
n = 2**20
print(f"N = 2^20: N²/(N log2 N) = {n / 20:.0f}")
# Nyquist and folding
for nu, stride in ((18.25, 35.0), (21.0, 30.0)):
    nyq = 1000 / (2 * stride)
    print(f"{nu} THz with stride {stride:g} fs: Nyquist {nyq:.2f}, folds to "
          f"{2 * nyq - nu:.2f} THz; unaliased for strides below "
          f"{1000 / (2 * nu):.1f} fs")
print(f"30 THz at 20 fs: Nyquist 25, alias {50 - 30} THz")
# Lorentzian width
for gamma in (1.0, 2.0):
    print(f"friction {gamma} ps⁻¹: width γ/2π = "
          f"{gamma / (2 * math.pi):.3f} THz")
print(f"window of 10 ps resolves about {1 / 10:.2f} THz; of 1 ps, "
      f"{1 / 1:.1f} THz")
# F(d) minimum for the surface at 300 K
k_h = 0.9785
print(f"F(d) least at √(k_BT/k) = {math.sqrt(KB * 300 / k_h):.3f} Å at 300 K")
for crossings in (10, 100, 400):
    print(f"{crossings} crossings: F to {2 / math.sqrt(crossings):.2f} k_BT")
# ballistic check answer
m_chk, t_chk, d_chk = 40.0, 300.0, 3e-4
print(f"check 16.5: 2Dm/k_BT = "
      f"{2 * d_chk * m_chk * units.MV2_TO_EV / (KB * t_chk):.0f} fs")
print(f"check 16.10: correlations to 1 ps resolve about {1 / 1.0:.0f} THz")
print("check 16.4: 25 events: 1/√25 = 0.20; 100 events halve it")
events, exposure = 40, 100 * 1000.0  # molecule-ps
print(f"40 dissociations over {exposure:.0f} molecule-ps: "
      f"{events / exposure * 1e4:.1f} ± "
      f"{math.sqrt(events) / exposure * 1e4:.1f}"
      f"e-4 per ps")
print(f"mean field against the run: {(u_mf * 1000) - (-50.09):.2f} meV")
