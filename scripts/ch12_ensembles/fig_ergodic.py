"""Figure fig:sm-ergodic: time averages and ensemble averages.

(a) A chain of 32 atoms between walls (mdlab.chain), started with all its
energy in the first normal mode: the share of the energy in modes 1 to 4
against time, for the chain of Fermi, Pasta, Ulam and Tsingou
(α = 0.25), whose modes exchange energy and nearly return it, and for the
harmonic chain (α = 0, dashed), whose first mode keeps it all. (b) Liquid
argon (runs.py): the running time average of the kinetic energy of three
single atoms, divided by the average over all the atoms and all the
frames, the ensemble average (dotted at 1).

Prints the numbers of Section 12.2.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch12 import MASS, RUNS

from mdlab import chain, units, viz
from mdlab.viz import (
    CYCLE,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

N, ALPHA, AMPLITUDE, DT = 32, 0.25, 4.0, 0.05
patterns, omega = chain.normal_modes(N)
print(f"chain of {N}: slowest mode period {2 * np.pi / omega[0]:.1f}, "
      f"fastest {2 * np.pi / omega[-1]:.2f}")
fput = chain.run(AMPLITUDE * patterns[:, 0], np.zeros(N), ALPHA, DT, 400000,
                 every=500)
harm = chain.run(AMPLITUDE * patterns[:, 0], np.zeros(N), 0.0, DT, 400000,
                 every=500)
share = fput["modes"] / fput["modes"].sum(1)[:, None]
low = int(np.argmax(share[:, 0] < 0.1))  # first time below a tenth
back = low + int(np.argmax(share[low:, 0] > 0.9))  # first return above 0.9
peak = back + int(np.argmax(share[back:back + 40, 0]))
print(f"total energy {fput['energy'][0]:.4f}, kept to "
      f"{np.ptp(fput['energy']) / fput['energy'][0]:.1e} of itself")
print(f"FPUT: mode 1 share first below 0.1 at t = {fput['times'][low]:.0f} "
      f"(lowest {share[:, 0].min():.3f}), back to {share[peak, 0]:.3f} at "
      f"t = {fput['times'][peak]:.0f}")
print(f"FPUT: largest share ever in modes 5 to 32 "
      f"{share[:, 4:].sum(1).max():.3f}; equal sharing would give "
      f"{28 / 32:.3f}")
hs = harm["modes"] / harm["modes"].sum(1)[:, None]
print(f"harmonic: mode 1 share stays between {hs[:, 0].min():.12f} and 1")

run = np.load(RUNS / "liquid.npz")
m, v, t = run["masses"], run["velocities"], run["frame_times"]
per_atom = 0.5 * units.MV2_TO_EV * MASS * np.sum(v * v, axis=-1)
scale = per_atom.mean()  # the average over all atoms and frames
running = np.cumsum(per_atom, axis=0) / np.arange(1, len(t) + 1)[:, None]
print(f"liquid: {t[-1] / 1000:g} ps, {len(t)} frames; average kinetic "
      f"energy per atom {scale * 1000:.3f} meV")
for atom in (0, 100, 200):
    print(f"   atom {atom}: time average after 5, 50 ps "
          f"{running[100, atom] / scale:.3f}, {running[-1, atom] / scale:.3f}")
spread = running[-1] / scale
print(f"time averages over 50 ps of all 256 atoms: from {spread.min():.3f} "
      f"to {spread.max():.3f}, standard deviation {spread.std():.3f}")

viz.use_style(notebook=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.5),
                           gridspec_kw=dict(wspace=0.35))
for k in range(4):
    a.plot(fput["times"] / 1000, share[:, k], color=CYCLE[k], lw=0.9, ls=("-", "--", "-.", ":")[k],
           label=f"mode {k + 1}")
a.plot(harm["times"] / 1000, hs[:, 0], **REFERENCE_STYLE, lw=0.9)
a.set_xlabel("time / 1000 units")
a.set_ylabel("share of the energy")
a.set_ylim(0, 1.3)
a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
a.legend(fontsize=7, ncol=2, loc="upper right", columnspacing=1.0)
viz.panel_tag(a, "a")
for atom, colour in zip((0, 100, 200), CYCLE[:3], strict=True):
    b.plot(t / 1000, running[:, atom] / scale, color=colour, lw=0.9)
b.axhline(1.0, **THRESHOLD_STYLE)
b.set_xlabel("time / ps")
b.set_ylabel(r"time average of $K_i$ / $\langle K_i\rangle$")
b.set_ylim(0, 2.2)
viz.panel_tag(b, "b")

print("wrote", viz.save(fig, figure_path("ch12_ensembles", "ergodic.pdf")))
