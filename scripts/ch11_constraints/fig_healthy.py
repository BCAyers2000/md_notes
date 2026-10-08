"""Figure fig:cs-healthy: what a healthy run of rigid water looks like.

The 64 rigid molecules of runs.py followed by RATTLE for 10 ps with
δt = 2 fs. (a) The changes of the kinetic, potential and total energies
from their starting values, per molecule, over the first 2 ps. (b) The
largest error of a held distance and the distance the centre of mass
has moved, on a logarithmic axis, over the whole run. (c) The
root-mean-square distance the oxygen atoms have moved from their starts.

Prints the numbers of Section 11.8.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch11 import DATA, N_MOLECULES, load

from mdlab import diagnostics, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, figure_path

run = np.load(DATA / "runs" / "rigid_healthy.npz")
start = load()
m = start["masses"]
t, u, k = run["times"], run["potential"], run["kinetic"]
e = u + k
frames, frame_times = run["frames"], run["frame_times"]

print(f"run of {t[-1] / 1000:g} ps, {len(t) - 1} steps of {run['dt']} fs")
print(f"std(E)/std(K) {diagnostics.energy_fluctuation(u, k):.4f}")
drift = diagnostics.energy_drift(t, u, k)
print(f"drift {1000 * drift:+.2e} eV/ps, {1000 * drift / N_MOLECULES:+.2e} "
      f"eV/ps per molecule")
print(f"E spread {np.ptp(e):.4f} eV, {1000 * np.ptp(e) / N_MOLECULES:.3f} "
      f"meV per molecule")
print(f"K mean {k.mean():.3f} eV, standard deviation {k.std():.3f} eV; "
      f"per freedom {k.mean() / (6 * N_MOLECULES - 3):.5f} eV")
print(f"U mean {u.mean():.3f} eV, {u.mean() / N_MOLECULES:.4f} eV per "
      f"molecule")
print(f"largest bond error {run['bond_error'].max():.2e} Å")
first = t <= 2000
for name, series in (("K", k), ("U", u)):
    swing = 1000 * (series[first] - series[0]) / N_MOLECULES
    print(f"{name} - {name}(0) over the first 2 ps: {swing.min():+.1f} to "
          f"{swing.max():+.1f} meV per molecule")
half = len(t) // 2
print(f"mean E, second half less first: "
      f"{np.mean(e[len(t) - half:]) - np.mean(e[:half]):.2e} eV, centres "
      f"{(np.mean(t[len(t) - half:]) - np.mean(t[:half])) / 1000:.3f} ps "
      f"apart")
com = diagnostics.centre_of_mass_path(m, frames)
print(f"centre of mass moves at most {com.max():.1e} Å")
oxygen = frames[:, ::3]
rms = diagnostics.rms_displacement(oxygen)
for when in (1000, 5000, 10000):
    k_ = int(np.argmin(np.abs(frame_times - when)))
    print(f"RMS displacement of the oxygens at {when / 1000:g} ps: "
          f"{rms[k_]:.2f} Å")
print(f"largest move between saved frames (50 fs): "
      f"{diagnostics.largest_jump(frames, start['cell']):.3f} widths")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.4),
                         gridspec_kw=dict(wspace=0.6,
                                          width_ratios=[1.25, 1, 1]))
a, b, c = axes
early = t <= 2000
scale = 1000 / N_MOLECULES  # meV per molecule
a.plot(t[early] / 1000, scale * (k[early] - k[0]), color=OCHRE, lw=0.6,
       label=r"$K$")
a.plot(t[early] / 1000, scale * (u[early] - u[0]), color=OXBLOOD, lw=0.6,
       label=r"$U$")
a.plot(t[early] / 1000, scale * (e[early] - e[0]), color=ACCENT, lw=1.0,
       label=r"$E$")
a.set_xlabel("time / ps")
a.set_ylabel("change per molecule / meV")
a.set_ylim(-15, 22)
a.legend(fontsize=7, loc="upper center", ncol=3, columnspacing=0.8,
         handlelength=1.2)
viz.panel_tag(a, "a")

b.semilogy(t / 1000, run["bond_error"], color=ACCENT, lw=0.6,
           label="largest bond error")
b.semilogy(frame_times[1:] / 1000, com[1:], color=OCHRE, lw=0.8,
           label="centre of mass moved")
b.set_ylim(1e-12, 1e-8)
b.set_xlabel("time / ps")
b.set_ylabel("length / Å")
b.legend(fontsize=7, loc="upper left")
viz.panel_tag(b, "b")

c.plot(frame_times / 1000, rms, color=ACCENT)
c.set_xlabel("time / ps")
c.set_ylabel("RMS displacement of O / Å")
c.set_xlim(0, 10)
c.set_ylim(0, None)
viz.panel_tag(c, "c")

print("wrote", viz.save(fig, figure_path("ch11_constraints",
                                         "healthy.pdf")))
