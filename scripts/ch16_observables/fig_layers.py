"""Figure fig:ob-layers: diffusion in layers, and hops.

(a) Chapter 14's layered solid with its 108 guests at 150 K (runs.py,
layered_150: 500 ps, every 100 fs): the guests' mean squared
displacement along x, y and z, measured from the centre of mass. (b)
Lithium on the model surface at 1000 K under Langevin friction of 1, 3,
10 and 30 ps⁻¹ (surface_1000 and surface_1000_g*): D from the MSD over
2-10 ps divided by k_h a²/4 from the counted hops (circles), and by
k_h⟨ℓ²⟩/4 with ⟨ℓ²⟩ the mean square length of a counted hop (squares).
(c) The
vibrational density of states of the lithium atoms at 600 K and 1000 K
(friction 1 ps⁻¹, 200 atoms, velocities every 5 fs over 20 ps), with the
harmonic frequency of the hollow (dotted).

Prints the guests' D along each axis with errors from 5 blocks, the hop
rates with their Poisson errors, D from the MSD, k_h a²/4 and f, the
share of counted hops longer than a and their mean square length, the
harmonic frequency, the VDOS peaks, and the estimate 6νe^{−βU_b}
against the hop rate and Chapter 13's transition-state rate.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch16 import RUNS, ch13

from mdlab import units, viz
from mdlab.analysis import spectra, transport
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE

layered = np.load(RUNS / "layered_150.npz")
guests = layered["guests"].astype(float) - layered["centre"][:, None, :]
frame = float(layered["frame_dt"])
lags = np.arange(501)
t = frame * lags
per_axis = transport.msd(guests, 500, remove_drift=False)
d_axes = []
for block in np.array_split(guests, 5):
    m = transport.msd(block, 500, remove_drift=False)
    d_axes.append([transport.diffusion_coefficient(t, m[:, k], 5000.0,
                                                   50000.0, dimensions=1)
                   for k in range(3)])
d_axes = np.array(d_axes)
mean, err = d_axes.mean(0), d_axes.std(0, ddof=1) / np.sqrt(5)
for axis, name in enumerate("xyz"):
    print(f"guests at 150 K, D_{name}{name} = {mean[axis] * 1e5:.3f} ± "
          f"{err[axis] * 1e5:.3f}e-5 Å²/fs")
spacing = layered["cell"][2, 2] / 9  # nine sheets
moves = np.abs(guests[:, :, 2] - guests[0, :, 2]).max()
print(f"z MSD at 1 ps {per_axis[10, 2]:.3f} Å²; no guest moves more than "
      f"{moves:.2f} Å along z in {len(guests) * frame / 1000:.0f} ps")
print(f"z MSD at 10 and 50 ps: {per_axis[100, 2]:.3f}, "
      f"{per_axis[500, 2]:.3f} Å²; the sheets are {spacing:.2f} Å apart, "
      f"its square {spacing**2:.1f} Å²")
span = len(guests) * frame
print(f"one crossing among {guests.shape[1]} guests in {span / 1000:.0f} ps: "
      f"D_zz ≈ {spacing**2 / (2 * guests.shape[1] * span):.1e} Å²/fs")

a = ch13.SPACING


def hop_lengths(path, core=0.7):
    """The squared length, in units of a², of each hop count_hops counts.

    A hop is the jump between the hollows an atom is assigned to.
    """
    site, dist = ch13.nearest_hollow(path)
    state = site[0].copy()
    squares = []
    for frame in range(1, len(path)):
        inside = dist[frame] < core
        moved = inside & np.any(site[frame] != state, axis=-1)
        jump = (site[frame][moved] - state[moved]) @ ch13.LATTICE
        squares.append(np.sum(jump**2, axis=-1) / a**2)
        state = np.where(inside[:, None], site[frame], state)
    return np.concatenate(squares)


rows = []
for gamma, name in ((1, "surface_1000"), (3, "surface_1000_g3"),
                    (10, "surface_1000_g10"), (30, "surface_1000_g30")):
    run = np.load(RUNS / f"{name}.npz")
    paths = run["positions"].astype(float)
    step = float(run["frame_dt"])
    hops = ch13.count_hops(paths).sum()
    exposure = paths.shape[1] * (len(paths) - 1) * step  # atom-fs
    rate = hops / exposure
    flat = np.concatenate([paths, np.zeros(paths.shape[:2] + (1,))], -1)
    m = transport.msd(flat, 200, remove_drift=False).sum(1)
    d = transport.diffusion_coefficient(step * np.arange(201), m, 2000.0,
                                        10000.0, dimensions=2)
    walk = rate * a * a / 4
    squares = hop_lengths(paths)
    rows.append((gamma, d / walk, d / walk / squares.mean()))
    print(f"γ = {gamma} ps⁻¹: {hops} hops, k_h = {rate * 1000:.4f} ± "
          f"{math.sqrt(hops) / exposure * 1000:.4f} per ps; D = "
          f"{d * 1e4:.3f}e-4, k_h a²/4 = {walk * 1e4:.3f}e-4 Å²/fs; "
          f"f = {d / walk:.2f}; counted hops longer than a: "
          f"{np.mean(squares > 1.01):.3f}; mean square hop "
          f"{squares.mean():.2f} a²; f over it "
          f"{d / walk / squares.mean():.2f}")

k_hollow = 3 * 0.3 * (4 * np.pi / (np.sqrt(3) * a)) ** 2 / 8
nu0 = math.sqrt(k_hollow * units.FORCE_TO_ACCEL / ch13.LI_MASS) / (2 * np.pi)
print(f"harmonic frequency of the hollow: k = 3U_b g²/8 = {k_hollow:.4f} "
      f"eV/Å², ν₀ = {nu0 * 1000:.2f} THz")
dens = {}
for temperature in (600, 1000):
    run = np.load(RUNS / f"surface_{temperature}.npz")
    v = run["velocities"].astype(float)
    v3 = np.concatenate([v, np.zeros(v.shape[:2] + (1,))], -1)
    nu, dos = spectra.vdos(v3, float(run["velocity_dt"]), 400)
    nu, dos = nu * 1000, dos / 1000 * 1.5  # two directions, area 2
    dens[temperature] = (nu, dos)
    band = nu > 1.0
    peak = nu[band][np.argmax(dos[band])]
    beta = 1 / (units.KB * temperature)
    crude = 6 * peak * math.exp(-0.3 * beta)
    hops = ch13.count_hops(run["positions"].astype(float)).sum()
    exposure = run["positions"].shape[1] * (len(run["positions"]) - 1) * \
        float(run["frame_dt"]) / 1000
    print(f"{temperature} K: VDOS peak {peak:.2f} THz; 6νe^(−βU_b) = "
          f"{crude:.3f} per ps; counted {hops / exposure:.3f} ± "
          f"{math.sqrt(hops) / exposure:.3f} per ps; transition-state rate "
          f"{ch13.tst_rate(temperature) * 1000:.3f} per ps")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
for axis, (name, colour) in enumerate((("x", ACCENT), ("y", OCHRE),
                                       ("z", OXBLOOD))):
    ax.plot(t / 1000, per_axis[:, axis], color=colour, lw=0.9,
            label=f"${name}$")
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"MSD / \AA$^2$")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "a")

ax = axes[1]
gam, ratio, memory = np.array(rows).T
ax.semilogx(gam, ratio, "o-", color=ACCENT, ms=3, lw=0.8,
            label=r"$4D/k_{\mathrm h}a^2$")
ax.semilogx(gam, memory, "s--", color=OCHRE, ms=3, lw=0.8,
            label=r"$4D/k_{\mathrm h}\langle\ell^2\rangle$")
ax.legend(fontsize=7, loc="upper right")
ax.axhline(1, **THRESHOLD_STYLE)
ax.set_xlabel(r"$\gamma$ / ps$^{-1}$")
ax.set_ylabel("correlation factor")
viz.panel_tag(ax, "b")

ax = axes[2]
for temperature, colour in ((600, ACCENT), (1000, OCHRE)):
    nu, dos = dens[temperature]
    ax.plot(nu, dos, color=colour, lw=0.9, label=f"{temperature} K")
ax.axvline(nu0 * 1000, **THRESHOLD_STYLE)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$\nu$ / THz")
ax.set_ylabel(r"$\mathcal{D}(\nu)$ / THz$^{-1}$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables",
                                             "layers.pdf")))
