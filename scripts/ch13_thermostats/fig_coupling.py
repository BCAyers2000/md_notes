"""Figure fig:th-coupling: how strongly a thermostat holds, and the dynamics.

(a) The diffusion coefficient of liquid argon, measured from the centre
of mass, against the collision rate of Andersen's thermostat and the
friction of Langevin's, each the mean of three runs of 100 ps with its
standard error, against the mean of three runs at fixed energy (band of
one standard error). (b) The rate at which lithium atoms on the model
surface hop between hollows at 1000 K, against the Langevin friction:
1000 atoms started from the canonical distribution (circles), and 1000
started at the bottom of a hollow with no potential energy, counted over
the first 100 ps (squares), against the transition-state rate (dashed).

Prints the numbers of Sections 13.3 and 13.4.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch13 import BARRIER, RUNS, T_LI, count_hops, diffusion, tst_rate

from mdlab import viz
from mdlab.units import KB
from mdlab.viz import ACCENT, OCHRE, REFERENCE, THERMOSTAT, figure_path, tint


def d_of(name):
    values = [diffusion(np.load(RUNS / f"{name}_s{s}.npz")["positions"],
                        500.0) * 1e4 for s in range(3)]
    return np.mean(values), np.std(values, ddof=1) / math.sqrt(3)


nve = d_of("nve")
print(f"fixed energy: D = {nve[0]:.3f} ± {nve[1]:.3f} × 1e-4 Å²/fs "
      f"= {nve[0] * 1e-5:.2e} cm²/s")
coupling = (0.1, 1, 10)
andersen = [d_of(f"andersen_{c:g}") for c in coupling]
langevin = [d_of(f"langevin_{c:g}") for c in coupling]
for c, (da, ea), (dl, el) in zip(coupling, andersen, langevin, strict=True):
    print(f"{c:g}/ps: Andersen D = {da:.3f} ± {ea:.3f} ({da / nve[0]:.2f} of "
          f"fixed energy), Langevin {dl:.3f} ± {el:.3f} "
          f"({dl / nve[0]:.2f})")
for name in ("berendsen_100", "berendsen_1000", "nh_100", "nh_1000",
             "nhc_100", "nhc_1000", "csvr_100", "csvr_1000"):
    d, e = d_of(name)
    print(f"{name}: D = {d:.3f} ± {e:.3f} ({d / nve[0]:.3f} of fixed "
          f"energy)")
raw = [diffusion(np.load(RUNS / f"langevin_0.1_s{s}.npz")["positions"],
                 500.0, relative=False) * 1e4 for s in range(3)]
print(f"Langevin 0.1/ps without removing the centre of mass: "
      f"{np.mean(raw):.3f} ± {np.std(raw, ddof=1) / math.sqrt(3):.3f}")

tst = tst_rate(T_LI) * 1000
print(f"transition-state rate at {T_LI:g} K: {tst:.3f}/ps")


def hop_rate(name):
    """Hops per atom per ps, with the standard error over the atoms."""
    path = np.load(RUNS / f"{name}.npz")["positions"]
    hops = count_hops(path)
    time = (len(path) - 1) * 10.0 / 1000  # frames every 10 fs, in ps
    return hops.mean() / time, hops.std(ddof=1) / math.sqrt(len(hops)) / time


frictions = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)
canonical = [hop_rate(f"li_langevin_{g:g}") for g in frictions]
hollow_f = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
hollow = [hop_rate(f"li_hollow_{g:g}") for g in hollow_f]
for g, (rate, err) in zip(frictions, canonical, strict=True):
    print(f"friction {g:g}/fs: {rate:.3f} ± {err:.3f} hops/ps "
          f"({rate / tst:.2f} of the transition-state rate)")
for g, (rate, _) in zip(hollow_f, hollow, strict=True):
    print(f"from the hollow, friction {g:g}/fs: {rate:.3f} hops/ps")
print(f"at fixed energy from the canonical start: "
      f"{hop_rate('li_langevin_0')[0]:.3f} hops/ps")
path = np.load(RUNS / "li_hollow_1e-05.npz")["positions"]
windows = []
for start in range(0, 10000, 2000):  # frames every 10 fs: 20 ps windows
    piece = path[start:start + 2001]
    windows.append(count_hops(piece).mean() / 20.0)
print("from the hollow, friction 1e-5/fs, hops/ps in successive 20 ps: "
      + ", ".join(f"{w:.3f}" for w in windows))
start = np.load(RUNS / "li_langevin_0.001.npz")
above = np.mean((start["potential"][0] + start["kinetic"][0]) > BARRIER)
print(f"canonical start: fraction of atoms with K + U above the barrier "
      f"{above:.3f}")
for g in (0.001, 0.003, 0.01):
    k = np.load(RUNS / f"li_langevin_{g:g}.npz")["kinetic"]
    print(f"Langevin {g:g}/fs: mean K / k_BT {k.mean() / (KB * T_LI):.4f}")
for name in ("li_csvr", "li_nhc", "li_nh", "li_andersen", "li_berendsen"):
    print(f"{name}: {hop_rate(name)[0]:.3f} hops/ps")

viz.use_style(notebook=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.4),
                           gridspec_kw=dict(wspace=0.35))
a.axhspan(nve[0] - nve[1], nve[0] + nve[1], color=tint(REFERENCE, 0.25),
          lw=0)
a.axhline(nve[0], color=REFERENCE, lw=0.8, ls="--")
a.text(0.03, 0.9, "fixed energy", color=REFERENCE, fontsize=6.5,
       transform=a.transAxes)
for values, colour, marker, label in ((andersen, THERMOSTAT["Andersen"], "s", "Andersen"),
                                      (langevin, THERMOSTAT["Langevin"], "o", "Langevin")):
    a.errorbar(coupling, [v[0] for v in values], [v[1] for v in values],
               color=colour, marker=marker, ms=4, lw=0.8, capsize=2,
               label=label)
a.set_xscale("log")
a.set_xlabel(r"collision rate or friction / ps$^{-1}$")
a.set_ylabel(r"$D$ / $10^{-4}$ Å$^2$ fs$^{-1}$")
a.set_ylim(0, 4.6)
a.legend(fontsize=6, loc="lower left")
viz.panel_tag(a, "a")

b.errorbar(np.array(frictions) * 1000, [c[0] for c in canonical],
           [c[1] for c in canonical], fmt="o", color=ACCENT, ms=4, lw=0.8,
           capsize=2, label="canonical start")
b.set_xscale("log")
b.set_ylim(0, 1.4)
b.plot(np.array(hollow_f) * 1000, [h[0] for h in hollow], "s",
       color=OCHRE, ms=3.5, label="from the hollow")
b.axhline(tst, color=REFERENCE, ls="--", lw=1.0)
b.text(0.97, 0.94, "transition-state rate", color=REFERENCE, fontsize=6.5,
       transform=b.transAxes, ha="right", va="top")
b.set_xlabel(r"friction / ps$^{-1}$")
b.set_ylabel(r"hops per atom per ps")
b.legend(fontsize=6, loc="lower left")
viz.panel_tag(b, "b")

print("wrote", viz.save(fig, figure_path("ch13_thermostats",
                                         "coupling.pdf")))
