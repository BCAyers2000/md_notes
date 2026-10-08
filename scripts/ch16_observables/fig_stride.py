"""Figure fig:ob-stride: what a stride does to a spectrum.

Twenty copies of the four-bead chain at fixed energy after 10 ps at
300 K (runs.py, chain_dense: velocities every 1 fs for 20 ps, each
copy's centre-of-mass velocity taken away). (a) The
vibrational density of states from every frame (correlations to 2 ps),
and from every 20th and every 30th frame, each to its own Nyquist
frequency (dotted). (b) The overlap score of the density from every
Δt-th frame against that from every frame, both cut at the Nyquist
frequency of Δt, against Δt; the dotted line marks 1/(2ν_b), with ν_b
the frequency of the highest peak, a stretch of the bonds.

Prints the bond stretch's harmonic frequency, the peaks from every
frame, where the bond peak lands at strides of 30 and 35 fs, and the
scores.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import CHAIN_BOND_K, CHAIN_MASS, RUNS

from mdlab import units, viz
from mdlab.analysis import spectra
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE

THZ = 1000.0
run = np.load(RUNS / "chain_dense.npz")
# Each copy's centre-of-mass velocity removed: the copies drift freely,
# which would put most of the weight at ν = 0.
copies = run["velocities"].astype(float)
copies = copies - copies.mean(axis=2, keepdims=True)
v = copies.reshape(len(copies), -1, 3)
harmonic = np.sqrt(2 * CHAIN_BOND_K * units.FORCE_TO_ACCEL
                   / (0.5 * CHAIN_MASS)) / (2 * np.pi) * THZ
print(f"{v.shape[1]} beads; bond stretch alone: √(2κ_r/m_r)/2π = "
      f"{harmonic:.2f} THz")

LAG_TIME = 2000.0  # fs
spectra_by_stride = {}
for stride in (1, 5, 10, 15, 20, 25, 30, 35, 40):
    nu, dos = spectra.vdos(v[::stride], float(stride),
                           int(LAG_TIME / stride))
    spectra_by_stride[stride] = (nu * THZ, dos / THZ)
nu1, dos1 = spectra_by_stride[1]
peaks = [nu1[k] for k in range(1, len(nu1) - 1)
         if dos1[k] > dos1[k - 1] and dos1[k] > dos1[k + 1]
         and dos1[k] > 0.02]
bond = max(peaks)
print("peaks from every frame: " + ", ".join(f"{p:.2f}" for p in peaks)
      + f" THz; the bond peak at {bond:.2f} THz, 1/(2ν_b) = "
      f"{1 / (2 * bond / THZ):.1f} fs")
for stride in (30, 35):
    nu_s, dos_s = spectra_by_stride[stride]
    nyquist = THZ / (2 * stride)
    for top in peaks[-2:]:
        folded = 2 * nyquist - top  # reflected at the Nyquist frequency
        near = np.abs(nu_s - folded) < 0.6
        found = nu_s[near][np.argmax(dos_s[near])]
        print(f"stride {stride} fs: Nyquist {nyquist:.2f} THz; the peak at "
              f"{top:.2f} THz should fold to {folded:.2f} THz; the largest "
              f"value near it is at {found:.2f} THz")

strides, scores = [], []
for stride, (nu_s, dos_s) in spectra_by_stride.items():
    nyquist = THZ / (2 * stride)
    ref = np.interp(nu_s, nu1, dos1)
    keep = nu_s <= nyquist
    scores.append(spectra.overlap_score(dos_s[keep], ref[keep]))
    strides.append(stride)
    print(f"stride {stride:2d} fs: overlap with every frame "
          f"{scores[-1]:.3f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.32))
ax = axes[0]
nu_s, dos_s = spectra_by_stride[1]
ax.plot(nu_s, dos_s, color=viz.REFERENCE, ls="--", lw=0.8,
        label=r"$\Delta t = 1$ fs")
for stride, colour in ((20, ACCENT), (30, OXBLOOD)):
    nu_s, dos_s = spectra_by_stride[stride]
    ax.plot(nu_s, dos_s, color=colour, lw=0.9,
            label=rf"$\Delta t = {stride}$ fs")
    ax.axvline(THZ / (2 * stride), color=colour, ls=":", lw=0.8)
ax.set_xlim(0, 30)
ax.set_xlabel(r"$\nu$ / THz")
ax.set_ylabel(r"$\mathcal{D}(\nu)$ / THz$^{-1}$")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.03, 1.0))
viz.panel_tag(ax, "a")

ax = axes[1]
ax.axvline(1 / (2 * bond / THZ), **THRESHOLD_STYLE)
ax.plot(strides, scores, "o-", color=OCHRE, ms=3, lw=0.8)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"$\Delta t$ / fs")
ax.set_ylabel("overlap score")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables",
                                             "stride.pdf")))
