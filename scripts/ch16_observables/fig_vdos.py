"""Figure fig:ob-vdos: the vibrational density of states.

(a) The fcc crystal at 21 K at fixed energy (runs.py, crystal_nve: 50 ps,
velocities every 10 fs, autocorrelation to 10 ps), against the histogram
of the normal-mode frequencies of the same 256-atom cell at its lattice
constant (bars), from the eigenvalues of its Hessian by finite
differences of the forces, and those modes' own C(t) through the same
window and transform (dashed). (b) The same crystal under Langevin friction
of 2 ps⁻¹ (crystal_langevin), against the fixed-energy density spread by
a Lorentzian of full width γ/2π (dashed). (c) Liquid argon at fixed
energy (argon_nve), with 12mD/k_BT at ν = 0 (point), D from the
Green-Kubo integral.

Prints the highest normal-mode frequency, the peaks, the overlap scores
of (a) and (b), and the liquid's 𝒟(0) against 12mD/k_BT.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import RUNS, argon_model, ch12, crystal_a0

from mdlab import units, viz
from mdlab.analysis import spectra, transport
from mdlab.viz import ACCENT, OCHRE, REFERENCE

DT = 10.0
THZ = 1000.0  # 1/fs to THz
LAG = 1000

# Normal modes of the cell at a₀.
a0 = crystal_a0()
r0, h = ch12.fcc(4, a0)
model = argon_model(h)
n = len(r0)
step = 1e-4
hessian = np.zeros((3 * n, 3 * n))
for k in range(3 * n):
    shift = np.zeros(3 * n)
    shift[k] = step
    f_plus = model(r0 + shift.reshape(n, 3))[1].ravel()
    f_minus = model(r0 - shift.reshape(n, 3))[1].ravel()
    hessian[:, k] = -(f_plus - f_minus) / (2 * step)
hessian = 0.5 * (hessian + hessian.T)
eig = np.linalg.eigvalsh(hessian / ch12.MASS * units.FORCE_TO_ACCEL)
modes = np.sqrt(np.clip(eig[3:], 0, None)) / (2 * np.pi) * THZ
print(f"a₀ = {a0:.5f} Å; three smallest eigenvalues "
      f"{', '.join(f'{e:.1e}' for e in eig[:3])} (the translations); "
      f"highest mode {modes.max():.3f} THz")

nve = np.load(RUNS / "crystal_nve.npz")
nu, dos = spectra.vdos(nve["velocities"], DT, LAG)
nu = nu * THZ
dos = dos / THZ  # per THz
width = nu[1] - nu[0]
edges = np.append(nu - 0.5 * width, nu[-1] + 0.5 * width)
counts, _ = np.histogram(modes, bins=edges)
hist = 3 * counts / counts.sum() / width
# The harmonic crystal's own C(t), Σ cos 2πν_k t over its modes, through
# the same window and transform: what the run would give with no
# anharmonicity.
lag_t = DT * np.arange(LAG + 1)
c_harm = np.cos(2 * np.pi * np.outer(lag_t, modes / THZ)).sum(1)
_, harm = spectra.density_of_states(c_harm, DT)
harm = harm / THZ
peaks = [nu[k] for k in range(1, len(nu) - 1)
         if dos[k] > dos[k - 1] and dos[k] > dos[k + 1] and dos[k] > 0.5]
print(f"resolution 1/T = {width:.3f} THz; VDOS peaks at "
      + ", ".join(f"{p:.2f}" for p in peaks) + " THz")
band = nu < 3.0
print(f"overlap score, VDOS against the modes through the same window: "
      f"{spectra.overlap_score(dos[band], harm[band]):.3f}; against the "
      f"bare histogram {spectra.overlap_score(dos[band], hist[band]):.3f}")

top = nu > modes.max()
print(f"area above the highest mode: run {np.trapezoid(dos[top], nu[top]):.3f}"
      f", modes through the window {np.trapezoid(harm[top], nu[top]):.3f}, "
      f"of 3; tallest peaks {dos.max():.2f} and {harm.max():.2f} per THz")
lang = np.load(RUNS / "crystal_langevin.npz")
_, dos_l = spectra.vdos(lang["velocities"], DT, LAG)
dos_l = dos_l / THZ
gamma = 0.002  # 1/fs
half = gamma / (2 * np.pi) * THZ / 2
kernel = half / np.pi / ((nu[:, None] - nu[None, :]) ** 2 + half**2)
spread = kernel @ dos * width
spread *= 3 / np.trapezoid(spread, nu)
whole = spectra.overlap_score(dos_l, spread)
print(f"the spread spectrum's tallest peak {spread.max():.2f} per THz; "
      f"overlap over the whole range {whole:.3f}")
print(f"Langevin γ = 2 ps⁻¹: full width γ/2π = {2 * half:.3f} THz; overlap "
      f"score with the spread fixed-energy density "
      f"{spectra.overlap_score(dos_l[band], spread[band]):.3f}, against "
      f"{spectra.overlap_score(dos_l[band], dos[band]):.3f} unspread")
for name, d in (("fixed energy", dos), ("Langevin", dos_l)):
    print(f"{name}: tallest peak {d.max():.2f} per THz at "
          f"{nu[np.argmax(d)]:.2f} THz")

liquid = np.load(RUNS / "argon_nve.npz")
v = liquid["velocities"]
nu_l, dos_q = spectra.vdos(v, DT, LAG)
nu_l, dos_q = nu_l * THZ, dos_q / THZ
temperature = 2 * liquid["kinetic"].mean() / ((3 * n - 3) * units.KB)
c = transport.velocity_autocorrelation(v, 1000)
d_gk = transport.running_integral(c, DT)[200:1001].mean() / 3
zero = 12 * ch12.MASS * units.MV2_TO_EV * d_gk / (units.KB * temperature)
print(f"liquid at {temperature:.2f} K: 𝒟(0) = {dos_q[0]:.3f} per THz "
      f"against 12mD/k_BT = {zero / THZ:.3f} per THz (D = "
      f"{d_gk * 1e4:.3f}e-4 Å²/fs); peak at {nu_l[np.argmax(dos_q)]:.2f} THz")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.42))
ax = axes[0]
ax.bar(nu, hist, width=width, color=viz.tint(REFERENCE, 0.35), lw=0)
ax.plot(nu, harm, color=REFERENCE, ls="--", lw=0.8)
ax.plot(nu, dos, color=OCHRE, lw=0.9)
ax.set_xlim(0, 2.6)
ax.set_xlabel(r"$\nu$ / THz")
ax.set_ylabel(r"$\mathcal{D}(\nu)$ / THz$^{-1}$")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.plot(nu, spread, color=REFERENCE, ls="--", lw=0.8)
ax.plot(nu, dos_l, color=ACCENT, lw=0.9)
ax.set_xlim(0, 2.6)
ax.set_ylim(0, axes[0].get_ylim()[1])
ax.set_xlabel(r"$\nu$ / THz")
viz.panel_tag(ax, "b")

ax = axes[2]
ax.plot(nu_l, dos_q, color=ACCENT, lw=0.9)
ax.plot([0], [zero / THZ], "o", color="black", ms=3)
ax.set_xlim(0, 2.6)
ax.set_xlabel(r"$\nu$ / THz")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "vdos.pdf")))
