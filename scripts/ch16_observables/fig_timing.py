"""Figure fig:ob-timing: what the bond analysis costs.

Liquid argon's model (switched Lennard-Jones, r_c = 8.5 Å, skin 1 Å) on
fcc lattices of 864 to 97 556 atoms at the liquid's density, each atom
moved at random by about 0.3 Å; bonds between atoms closer than 4.0 Å,
kept to 4.4 Å, about 4.4 per atom, a harder test than most molecules
give. (a) The time of one force evaluation with the neighbour list
already built, and of one frame's analysis by a ReactionLog fed with the
pairs and distances the force evaluation kept (bonds, molecules, graph
hashes, formulas and reactions), the least of three tries, against N.
(b) Their ratio, with the tenth that the analysis is meant to stay
within (dotted).

Prints each N's times, the ratio and the share of each part, as ratios
only, since the times themselves depend on the machine.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from ch16 import argon_model, ch12

from mdlab import bonds, viz
from mdlab.viz import ACCENT, OCHRE, THRESHOLD_STYLE


def least(function, tries=3):
    best = np.inf
    for _ in range(tries):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


sizes, forces, analyses = [], [], []
for cells in (6, 9, 14, 20, 29):
    r, h = ch12.fcc(cells, ch12.A_LIQUID)
    rng = np.random.default_rng(0)
    r = r + rng.normal(0, 0.3, r.shape)
    n = len(r)
    model = argon_model(h)
    model(r)  # builds the neighbour list
    moved = r + rng.normal(0, 0.02, r.shape)
    force = least(lambda model=model, moved=moved: model(moved))
    log = bonds.ReactionLog(bonds.BondTracker(
        ["Ar"] * n, pairs={("Ar", "Ar"): (4.0, 4.4)}))
    model(r)
    log.add(0.0, *model.last_pairs)  # first frame, and Numba compiles
    model(moved)
    pairs = model.last_pairs
    start = (log.tracker.keys.copy(), log._labels, log._species)

    def analyse(log=log, start=start, pairs=pairs):
        """One frame from the same starting state each time."""
        log.tracker.keys = start[0].copy()
        log._labels, log._species = start[1], start[2]
        log.add(1.0, *pairs)

    analysis = least(analyse)
    sizes.append(n)
    forces.append(force)
    analyses.append(analysis)
    print(f"N = {n}: analysis / force = {analysis / force:.3f}; "
          f"{2 * len(log.tracker.keys) / n:.1f} bonds per atom")
sizes, forces, analyses = map(np.array, (sizes, forces, analyses))
print(f"from N = {sizes[0]} to {sizes[-1]} ({sizes[-1] / sizes[0]:.0f} times "
      f"as many) the force evaluation takes {forces[-1] / forces[0]:.0f} "
      f"times as long and the analysis {analyses[-1] / analyses[0]:.0f} times")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.35))
ax = axes[0]
ax.loglog(sizes, forces / forces[0], "o-", color=OCHRE, ms=3, lw=0.8,
          label="force evaluation")
ax.loglog(sizes, analyses / forces[0], "o-", color=ACCENT, ms=3, lw=0.8,
          label="bond analysis")
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"time / force time at $N = 864$")
ax.legend(fontsize=6, loc="upper left")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.semilogx(sizes, analyses / forces, "o-", color=ACCENT, ms=3, lw=0.8)
ax.axhline(0.1, **THRESHOLD_STYLE)
ax.set_ylim(0, 0.3)
ax.set_xlabel(r"$N$")
ax.set_ylabel("analysis / force")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables",
                                             "timing.pdf")))
