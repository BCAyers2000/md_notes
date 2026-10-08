"""Figure fig:ob-salt: charge transport in the model molten salt.

runs.py, salt_s0 to salt_s7: 64 ions, 1 ns each at fixed energy after
melting and 40 ps at 1500 K. (a) The partial radial distribution
functions g_+−, g_++ and g_−−, from all eight runs. (b) The conductivity
of each run from the charge displacement Σqᵢrᵢ (circles) and from
Nernst-Einstein with the ions' own diffusion coefficients (squares),
against the run's temperature. (c) The vibrational density of states of
the cations and of the anions (salt_s0, velocities every 10 fs over
100 ps, correlations to 5 ps).

Prints the peaks and coordination numbers, each run's temperature,
conductivities by the Einstein and Green-Kubo routes, diffusion
coefficients and Haven ratio, their means with standard errors, and the
species' VDOS peaks and overlap.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import RUNS

from mdlab import units, viz
from mdlab.analysis import spectra, structure, transport
from mdlab.viz import ACCENT, OCHRE, OXBLOOD

E = units.ELEMENTARY_CHARGE
TO_SI = 1e-5 / 1e-30  # (Å²/fs)/Å³ in (m²/s)/m³


def conductivity(slope, volume, kt):
    """e² × slope/(6Vk_BT), the slope of ⟨|ΔΣqr|²⟩ in Å²/fs, in S/m."""
    return E * slope * TO_SI / (6 * volume * kt)


rows, frames_all = [], []
for seed in range(8):
    run = np.load(RUNS / f"salt_s{seed}.npz")
    q, m, h = run["charges"], run["masses"], run["cell"]
    volume = np.linalg.det(h)
    temperature = 2 * run["kinetic"].mean() / ((3 * len(q) - 3) * units.KB)
    kt = units.KB * temperature
    lags = np.arange(0, 2501, 25)
    t = 2.0 * lags
    fit = (t >= 1000) & (t <= 5000)
    charge = transport.msd(run["charge_displacement"][:, None, :],
                           remove_drift=False, origin_step=5,
                           lags=lags).sum(1)
    sigma = conductivity(np.polyfit(t[fit], charge[fit], 1)[0], volume, kt)
    current = transport.correlation(run["charge_current"], max_lag=2500)
    plateau = transport.running_integral(current, 2.0)[500:].mean()
    sigma_gk = E * plateau * TO_SI / (3 * volume * kt)
    pos = run["positions"].astype(float)
    frame = float(run["frame_dt"])
    tp = frame * np.arange(51)
    d = [transport.diffusion_coefficient(
        tp, transport.msd(pos, 50, masses=m, select=np.flatnonzero(q == s))
        .sum(1), 1000.0, 5000.0) for s in (1.0, -1.0)]
    sigma_ne = conductivity(6 * (32 * d[0] + 32 * d[1]), volume, kt)
    rows.append((temperature, sigma, sigma_gk, d[0], d[1], sigma_ne,
                 sigma_ne / sigma))
    frames_all.append(pos[::5])
    print(f"run {seed}: T = {temperature:.0f} K; σ = {sigma:.1f} S/m "
          f"(Green-Kubo {sigma_gk:.1f}); D₊ = {d[0] * 1e4:.2f}e-4, "
          f"D₋ = {d[1] * 1e4:.2f}e-4 Å²/fs; Nernst-Einstein "
          f"{sigma_ne:.1f} S/m; H_R = {sigma_ne / sigma:.3f}")
rows = np.array(rows)
mean = rows.mean(0)
err = rows.std(0, ddof=1) / np.sqrt(len(rows))
print(f"means over 8 runs: T = {mean[0]:.0f} ± {err[0]:.0f} K, σ = "
      f"{mean[1]:.0f} ± {err[1]:.0f} S/m, σ_NE = {mean[5]:.0f} ± {err[5]:.0f}"
      f" S/m, H_R = {mean[6]:.3f} ± {err[6]:.3f}; the Einstein and "
      f"Green-Kubo routes differ by at most "
      f"{np.max(np.abs(rows[:, 2] / rows[:, 1] - 1)) * 100:.1f}%")
ratio = rows[:, 3] / rows[:, 4]
print(f"run by run D₊/D₋ = {ratio.mean():.3f} ± "
      f"{ratio.std(ddof=1) / np.sqrt(len(ratio)):.3f}; cations faster in "
      f"{np.sum(ratio > 1)} runs of {len(ratio)}; H_R − 1 is "
      f"{(mean[6] - 1) / err[6]:.1f} errors against Student's 95% factor "
      f"for 7 degrees of freedom, 2.365")
print(f"temperatures range from {rows[:, 0].min():.0f} to "
      f"{rows[:, 0].max():.0f} K; H_R above 1.005 in "
      f"{np.sum(rows[:, 6] > 1.005)} runs, within 0.005 of 1 in "
      f"{np.sum(np.abs(rows[:, 6] - 1) <= 0.005)}, below in "
      f"{np.sum(rows[:, 6] < 0.995)}; D₊ = {mean[3] * 1e4:.2f} ± "
      f"{err[3] * 1e4:.2f}e-4, D₋ = {mean[4] * 1e4:.2f} ± {err[4] * 1e4:.2f}"
      f"e-4 Å²/fs")

frames = np.concatenate(frames_all)
q = np.load(RUNS / "salt_s0.npz")["charges"]
h = np.load(RUNS / "salt_s0.npz")["cell"]
plus, minus = np.flatnonzero(q > 0), np.flatnonzero(q < 0)
r_max = h[0, 0] / 2 - 1e-6
partial = {}
for key, (a, b) in {"+-": (plus, minus), "++": (plus, plus),
                    "--": (minus, minus)}.items():
    x, g = structure.rdf(frames, h, r_max, 134, centres=a, partners=b)
    partial[key] = g
    top = np.argmax(g)
    print(f"g_{key}: first peak {g[top]:.2f} at {x[top]:.2f} Å")
rho_minus = len(minus) / np.linalg.det(h)
n_c = structure.running_coordination(x, partial["+-"], rho_minus)
g = partial["+-"]
first_min = np.argmax((x > x[np.argmax(g)]) & (np.gradient(g) > 0))
print(f"anions around a cation out to the first minimum of g_+- "
      f"({x[first_min]:.2f} Å): {n_c[first_min]:.2f}")

run0 = np.load(RUNS / "salt_s0.npz")
v = run0["velocities"]
dt_v = float(run0["velocity_dt"])
nu, dos_p = spectra.vdos(v[:, plus], dt_v, 500)
_, dos_m = spectra.vdos(v[:, minus], dt_v, 500)
nu = nu * 1000
for name, d_ in (("cations", dos_p), ("anions", dos_m)):
    print(f"VDOS of the {name}: peak at {nu[np.argmax(d_)]:.2f} THz, "
          f"𝒟(0) = {d_[0] / 1000:.3f} per THz")
print(f"overlap of the two: {spectra.overlap_score(dos_p, dos_m):.3f}; "
      f"mass ratio {run0['masses'].max() / run0['masses'].min():.3f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
for key, colour, label in (("+-", ACCENT, r"$+-$"), ("++", OCHRE, r"$++$"),
                           ("--", OXBLOOD, r"$--$")):
    ax.plot(x, partial[key], color=colour, lw=0.9, label=label)
ax.axhline(1, color="black", lw=0.4)
ax.set_xlim(0, r_max)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$g_{\alpha\beta}(r)$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.plot(rows[:, 0], rows[:, 1], "o", color=ACCENT, ms=3,
        label=r"$\sigma_{\mathrm e}$")
ax.plot(rows[:, 0], rows[:, 5], "s", color=OCHRE, ms=3,
        label=r"$\sigma_{\mathrm{NE}}$")
ax.set_xlabel(r"$T$ / K")
ax.set_ylabel(r"conductivity / S\,m$^{-1}$")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "b")

ax = axes[2]
ax.plot(nu, dos_p / 1000, color=ACCENT, lw=0.9, label="cations")
ax.plot(nu, dos_m / 1000, color=OXBLOOD, lw=0.9, label="anions")
ax.set_xlim(0, 12)
ax.set_xlabel(r"$\nu$ / THz")
ax.set_ylabel(r"$\mathcal{D}(\nu)$ / THz$^{-1}$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "salt.pdf")))
