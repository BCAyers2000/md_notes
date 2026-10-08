"""Figure fig:ob-carbon: carbon quenched from the melt.

216 carbon atoms at 2.0 g/cm³ with Tersoff's potential (runs.py,
carbon_fast and carbon_slow): 2 ps at 6000 K, the thermostat's target
lowered steadily to 300 K over 10 ps or 40 ps, then 2 ps at 300 K. (a)
g(r) of the melt (its last 1.5 ps at 6000 K) and of the quenched solid
(the last 2 ps of the slow quench), with r_on and r_off of the default
band (dotted). (b) The fractions of atoms with two, three and four bonds
against the thermostat's target temperature, for the fast quench (thin)
and the slow (thick), from 0.5 ps on, once the starting lattice has
melted.

Prints the band, g(r)'s first peak and minimum, the fractions over the
last 2 ps of each quench with their spread, the bond events and the
number of molecules at the end.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import RUNS

from mdlab import bonds, viz
from mdlab.analysis import structure
from mdlab.neighbours import cell_list_pairs
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE

runs = {name: np.load(RUNS / f"carbon_{name}.npz")
        for name in ("fast", "slow") if (RUNS / f"carbon_{name}.npz").exists()}
tracker0 = bonds.BondTracker(["C"] * 216)
r_on, r_off = tracker0.r_on[0, 0], tracker0.r_off[0, 0]
print(f"carbon: covalent radius sum {2 * bonds.covalent_radii(['C'])[0]:.2f}"
      f" Å; r_on = {r_on:.3f} Å, r_off = {r_off:.3f} Å")

def largest(counts):
    """The number of atoms in the largest species, from formulas C<n>."""
    return max(int("".join(c for c in s if c.isdigit()) or 1) for s in counts)


fractions = {}
for name, run in runs.items():
    pos = run["positions"].astype(float)
    h = run["cell"]
    tracker = bonds.BondTracker(["C"] * len(pos[0]))
    log = bonds.ReactionLog(tracker)
    share = []
    for t, frame in zip(run["times"], pos, strict=True):
        i, j, _, dist = cell_list_pairs(frame, h, tracker.reach)
        log.add(float(t), i, j, dist)
        a, b = np.divmod(tracker.keys, 216)
        z = np.bincount(np.concatenate([a, b]), minlength=216)
        share.append(np.bincount(z, minlength=6)[:6] / 216)
    share = np.array(share)
    melted = run["times"] > 500  # past the starting lattice
    fractions[name] = (run["target"][melted], share[melted])
    last = run["times"] >= run["times"][-1] - 2000
    mean, spread = share[last].mean(0), share[last].std(0)
    print(f"{name} quench, last 2 ps: two bonds {mean[2]:.3f} ± "
          f"{spread[2]:.3f}, three {mean[3]:.3f} ± {spread[3]:.3f}, four "
          f"{mean[4]:.3f} ± {spread[4]:.3f}; {len(log.events)} bond events; "
          f"{len(log.counts[-1])} species at the end, the largest molecule "
          f"{largest(log.counts[-1])} atoms")
    last = max(e[0] for e in log.events)
    at = run["target"][np.searchsorted(run["times"], last)]
    late_events = [e for e in log.events if e[0] > run["times"][-1] - 4000]
    early = sum(1 for e in log.events if e[0] <= 2000)
    print(f"{name}: {len(log.events)} bond events after the first frame, "
          f"{early} of them in the first 2 ps")
    print(f"{name}: last bond event at {last / 1000:.2f} ps, target "
          f"{at:.0f} K; events in the last 4 ps: {len(late_events)}")
    melt = (run["times"] > 500) & (run["times"] <= 2000)
    print(f"{name}, melt at 6000 K: two {share[melt, 2].mean():.3f}, three "
          f"{share[melt, 3].mean():.3f}, four {share[melt, 4].mean():.3f}")

source = runs["slow"] if "slow" in runs else runs["fast"]
h = source["cell"]
times = source["times"]
x, g_melt = structure.rdf(source["positions"][(times > 500) & (times <= 2000)]
                          .astype(float), h, 6.4, 256)
x, g_cold = structure.rdf(source["positions"][times >= times[-1] - 2000]
                          .astype(float), h, 6.4, 256)
for label, g in (("melt", g_melt), ("quenched", g_cold)):
    peak = np.argmax(g)
    window = (x > 1.6) & (x < 2.6)
    low = np.flatnonzero(window)[np.argmin(g[window])]
    print(f"{label}: first peak {g[peak]:.2f} at {x[peak]:.3f} Å; first "
          f"minimum {g[low]:.3f} at {x[low]:.3f} Å")
    empty = window & (g == 0)
    if empty.any():
        width = x[1] - x[0]
        print(f"{label}: g is zero from {x[empty].min() - width / 2:.2f} to "
              f"{x[empty].max() + width / 2:.2f} Å")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.32))
ax = axes[0]
ax.plot(x, g_melt, color=OCHRE, lw=0.9, label="6000 K")
ax.plot(x, g_cold, color=ACCENT, lw=0.9, label="300 K")
for edge in (r_on, r_off):
    ax.axvline(edge, **THRESHOLD_STYLE)
ax.set_xlim(0, 6.4)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$g(r)$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
for name, width in (("fast", 0.6), ("slow", 1.2)):
    if name not in fractions:
        continue
    target, share = fractions[name]
    order = slice(None, None, -1)
    for z, colour in ((2, OCHRE), (3, ACCENT), (4, OXBLOOD)):
        ax.plot(target[order], share[order, z], color=colour, lw=width,
                label=f"{z} bonds" if name == "slow" else None)
ax.set_xlim(6000, 300)
ax.set_ylim(0, 1)
ax.set_xlabel("target temperature / K")
ax.set_ylabel("fraction of atoms")
ax.legend(fontsize=7, loc="center left")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables",
                                             "carbon.pdf")))
