"""Figure fig:sm-faults: faults that show in the temperature.

(a) Freedoms miscounted: the rigid water of Chapter 11 (10 ps at 2 fs),
its kinetic energy divided among 3N − n_c − 3 = 576 − 192 − 3 = 381
freedoms (grey dashed) and among 3N = 576 (teal). (b) A crystal started
on its lattice: 256 argon atoms on ASE's lattice, a = 5.26 Å, given
velocities at 40 K (dotted). (c) The drift counted: the liquid started
with a drift of 0.0005 Å/fs in every velocity, against the healthy
liquid (dashed). (d) Velocities drawn without the masses: argon atoms of
two masses, 39.948 and 83.798 amu, started with one spread of velocity
for every atom; the temperatures of the light (teal) and heavy (ochre)
atoms, with the run started correctly (dashed). (e) Velocities not
Gaussian: the liquid started with components spread evenly over an
interval; the excess kurtosis against the healthy run (dashed). (f) No
sharing: the chain of Fermi, Pasta, Ulam and Tsingou of Fig. 12.2, the
energy of each mode averaged over 20 000 units of time, against equal
sharing (dotted).

Prints the check values of Table tab:sm-faults.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ch12 import MASS, RUNS

from mdlab import chain, diagnostics, statmech, units, viz
from mdlab.viz import (
    ACCENT,
    OCHRE,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

BAD = dict(color=ACCENT, lw=0.6)
GOOD = dict(lw=0.6, **REFERENCE_STYLE)


def run(name):
    return np.load(RUNS / f"{name}.npz")


def temperature(r, n_free):
    return 2 * r["kinetic"] / (n_free * units.KB)


# (a) the rigid water of Chapter 11
water = np.load(Path(viz.THEORY, "data", "ch11_constraints", "runs",
                     "rigid_healthy.npz"))
right = temperature(water, statmech.degrees_of_freedom(192, 192))
wrong = temperature(water, 3 * 192)
print(f"(a) rigid water: T over 381 freedoms {right.mean():.1f} K, over 576 "
      f"{wrong.mean():.1f} K, ratio {wrong.mean() / right.mean():.3f}")

# (b) lattice start
lattice = run("lattice")
tl = temperature(lattice, 765)
print(f"(b) lattice start: T at the start {tl[0]:.1f} K, over the last 10 "
      f"ps {tl[1000:].mean():.1f} K, ratio {tl[1000:].mean() / tl[0]:.3f}")
low = int(np.argmin(tl[:100]))
print(f"    lowest {tl[low]:.1f} K at {lattice['times'][low]:.0f} fs; first "
      f"below 20 K at {lattice['times'][int(np.argmax(tl < 20))]:.0f} fs")

# (c) drift
liquid, drift = run("liquid"), run("drift")
tg, td = temperature(liquid, 765), temperature(drift, 765)
print(f"    healthy liquid: T at the start {tg[0]:.1f} K, over the last 10 ps "
      f"{tg[-1000:].mean():.1f} K, ratio {tg[-1000:].mean() / tg[0]:.3f}")
k_drift = 0.5 * units.MV2_TO_EV * 256 * MASS * 3 * 0.0005**2
print(f"    box mass {256 * MASS:.0f} amu; momentum sqrt(3) M v_d "
      f"{3**0.5 * 256 * MASS * 0.0005:.3f} amu Å/fs")
print(f"(c) drift: T {td.mean():.1f} K against {tg.mean():.1f} K; the drift's "
      f"kinetic energy {k_drift:.3f} eV, {2 * k_drift / (765 * units.KB):.1f} "
      f"K of apparent temperature")
mv = drift["velocities"]
p = np.einsum("i,tix->tx", drift["masses"], mv)
p_good = np.einsum("i,tix->tx", liquid["masses"], liquid["velocities"])
print(f"    total momentum |P| {np.linalg.norm(p, axis=1).mean():.3f} amu "
      f"Å/fs (healthy {np.linalg.norm(p_good, axis=1).max():.1e})")
relative = mv - (p / drift["masses"].sum())[:, None, :]
t_rel = statmech.kinetic_temperature(drift["masses"], relative, 765)
print(f"    the drift run with its drift taken out: {t_rel.mean():.1f} K")

# (d) masses ignored
light, heavy = np.arange(0, 256, 2), np.arange(1, 256, 2)
species = {}
for name in ("mixture_massless", "mixture_masses"):
    r = run(name)
    species[name] = (
        r["frame_times"],
        statmech.part_temperature(r["masses"], r["velocities"], atoms=light),
        statmech.part_temperature(r["masses"], r["velocities"], atoms=heavy),
    )
t0, l0, h0 = species["mixture_massless"]
print(f"(d) masses ignored: at the start light {l0[0]:.0f} K, heavy "
      f"{h0[0]:.0f} K; after 1 ps {l0[100]:.0f} and {h0[100]:.0f} K; last 10"
      f" ps {l0[1000:].mean():.0f} and {h0[1000:].mean():.0f} K")
t1, l1, h1 = species["mixture_masses"]
print(f"    scatter of 128 atoms' temperature: sqrt(2/384) = "
      f"{np.sqrt(2 / 384):.3f}")
print(f"    drawn with the masses: at the start light {l1[0]:.0f} K, heavy "
      f"{h1[0]:.0f} K; last 10 ps {l1[1000:].mean():.0f} and "
      f"{h1[1000:].mean():.0f} K")
gap = np.abs(l0 - h0)
settled = t0[int(np.argmax(np.convolve(gap, np.ones(20) / 20, "same")
                           < 15))]
print(f"    the gap first averages below 15 K at about {settled:.0f} fs")

# (e) uniform velocities
uni = run("uniform")
ku = np.array([diagnostics.excess_kurtosis(uni["masses"], f)
               for f in uni["velocities"]])
kh = np.array([diagnostics.excess_kurtosis(liquid["masses"], f)
               for f in liquid["velocities"]])
print(f"(e) uniform start: kurtosis {ku[0]:+.2f} at the start, "
      f"{ku[10]:+.2f} after 100 fs, {ku[20]:+.2f} after 200 fs; healthy "
      f"{kh[0]:+.2f} at the start")

# (f) no sharing
patterns, omega = chain.normal_modes(32)
fput = chain.run(4.0 * patterns[:, 0], np.zeros(32), 0.25, 0.05, 400000,
                 every=500)
mean_modes = fput["modes"].mean(0)
equal = fput["energy"][0] / 32
low_share = mean_modes[:4].sum() / mean_modes.sum()
print(f"(f) FPUT: modes 1 to 4 hold {low_share:.3f} of the energy on "
      f"average; equal sharing {4 / 32:.3f}; mode 1 "
      f"{mean_modes[0] / equal:.1f} times the equal share, modes 8 to 32 "
      f"at most {mean_modes[7:].max() / equal:.1e}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(3, 2, figsize=(viz.FULL, 6.3),
                         gridspec_kw=dict(hspace=0.75, wspace=0.4))
ax = axes.ravel()


def title(a, letter, text):
    a.set_title(f"({letter}) {text}", loc="left")


a = ax[0]
a.plot(water["times"] / 1000, wrong, **BAD)
a.plot(water["times"] / 1000, right, **GOOD)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ / K")
title(a, "a", "freedoms miscounted")
a = ax[1]
a.plot(lattice["times"] / 1000, tl, **BAD)
a.axhline(40.0, **THRESHOLD_STYLE)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ / K")
title(a, "b", "a crystal started on its lattice")
a = ax[2]
a.plot(drift["times"] / 1000, td, **BAD)
a.plot(liquid["times"][:2001] / 1000, tg[:2001], **GOOD)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ / K")
title(a, "c", "the drift counted")
a = ax[3]
t1, l1, h1 = species["mixture_masses"]
a.plot(t0 / 1000, l0, color=ACCENT, lw=0.6)
a.plot(t0 / 1000, h0, color=OCHRE, lw=0.6)
a.plot(t1 / 1000, l1, **GOOD)
a.set_xlim(0, 5)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ of a species / K")
title(a, "d", "velocities without the masses")
a = ax[4]
a.plot(uni["frame_times"] / 1000, ku, **BAD)
a.plot(liquid["frame_times"][:200] / 1000, kh[:200], **GOOD)
a.set_xlim(0, 2)
a.set_xlabel("time / ps")
a.set_ylabel("excess kurtosis")
title(a, "e", "velocities not Gaussian")
a = ax[5]
a.semilogy(np.arange(1, 33), mean_modes / equal, "o", color=ACCENT, ms=3)
a.axhline(1.0, **THRESHOLD_STYLE)
a.set_xlabel("mode")
a.set_ylabel("mean energy / equal share")
title(a, "f", "no sharing")

print("wrote", viz.save(fig, figure_path("ch12_ensembles", "faults.pdf")))
