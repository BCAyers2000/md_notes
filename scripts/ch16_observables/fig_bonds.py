"""Figure fig:ob-bonds: bonds, molecules and their reactions in the A-B gas.

runs.py, gas_<T>: 250 AB molecules in 63 Å, 100 ps under Langevin
friction of 1 ps⁻¹. (a) Bond events per picosecond at 2000 K, counted
with a single cutoff at r_on and with the band from r_on to r_off,
against the least lifetime an event must have to count. (b) The numbers
of AB, of free A and of the larger complexes against time at 2000 K.
(c) The equilibrium constant n_AB V/(n_A n_B) from the band's counts over
the last 90 ps (points, with the standard error of 9 blocks), and from
the single cutoff's, against ∫ 4πr² e^{−βφ} dr out to r_on (line).

Prints r_on and r_off, the period of small vibrations, the event counts,
the mean numbers of each species and the reaction network at 2000 K, the
complexes' lifetimes, and the equilibrium constants with their ratio to
the integral.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import (
    GAS_MASS,
    GAS_SIDE,
    LIKE_R,
    MORSE_DEPTH,
    MORSE_OFF,
    MORSE_ON,
    MORSE_WIDTH,
    RUNS,
    morse_pair,
)
from scipy.integrate import quad

from mdlab import bonds, units, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE

TEMPERATURES = (1500, 1750, 2000, 2250, 2500)
FRAME = 10.0  # fs
SETTLE = 10_000.0  # fs left out at the start
V = GAS_SIDE**3
print(f"r_on = {MORSE_ON:.4f} Å (φ = −D/2), r_off = {MORSE_OFF:.4f} Å "
      f"(φ = −D/8)")
mu = 0.5 * GAS_MASS
period = 2 * np.pi * np.sqrt(mu / (2 * MORSE_DEPTH * MORSE_WIDTH**2
                                   * units.FORCE_TO_ACCEL))
print(f"small vibrations of AB: period 2π√(m_r/2D_M a_M²) = {period:.1f} fs, "
      f"reduced mass {mu:.4f} amu")
# A straight B-A-B with both bonds at the length that minimises its energy.
lengths = np.linspace(1.1, 2.0, 9001)
line = (2 * morse_pair(lengths)[0]
        + MORSE_DEPTH * np.exp(-2 * MORSE_WIDTH * (2 * lengths - LIKE_R)))
best = np.argmin(line)
print(f"B-A-B in a line: least energy {line[best]:.3f} eV with bonds of "
      f"{lengths[best]:.2f} Å, {-MORSE_DEPTH - line[best]:.2f} eV below AB "
      f"and a free B")


def events_after(run, key, start=0.0):
    ev = run[f"{key}_events"]
    return [(float(t), int(i), int(j), int(s)) for t, i, j, s in ev
            if t > start]


run = np.load(RUNS / "gas_2000.npz")
lifetimes = np.array([0, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000.0])
span = (run["times"][-1]) / 1000  # ps
per_ps = {}
for key in ("single", "band"):
    ev = events_after(run, key)
    per_ps[key] = [len(bonds.persistent(ev, tau)) / span for tau in lifetimes]
    print(f"2000 K, {key}: {len(ev)} events in {span:.0f} ps; kept with a "
          f"least lifetime of 0.5 ps: {per_ps[key][5] * span:.0f} "
          f"({per_ps[key][0] / per_ps[key][5]:.1f} times fewer); of 2 ps: "
          f"{per_ps[key][-1] * span:.0f}, "
          f"{100 * (1 - per_ps[key][-1] / per_ps[key][5]):.0f}% fewer")
ratio = per_ps["single"][0] / per_ps["band"][0]
print(f"the band cuts the events by {ratio:.1f} times; with 0.5 ps of "
      f"lifetime the two criteria keep "
      f"{per_ps['single'][5] * span:.0f} and {per_ps['band'][5] * span:.0f}")

names = list(run["band_species"])
counts = run["band_counts"]
settled = run["times"] > SETTLE
means = counts[settled].mean(0)
print("2000 K, mean numbers after 10 ps: " + ", ".join(
    f"{s} {m:.1f}" for s, m in zip(names, means, strict=True) if m > 0.05))
reactions, number = np.unique(run["band_reactions"], return_counts=True)
order = np.argsort(-number)
print("reaction network at 2000 K (band, every event):")
for k in order[:8]:
    print(f"   {reactions[k]}: {number[k]}")
print(f"   {len(reactions)} kinds of reaction in all")
span_settled = (run["times"][-1] - SETTLE) / 1000
for complex_, forming in (("AB2", "AB + B -> AB2"), ("A2B", "A + AB -> A2B")):
    made = np.sum((run["band_reactions"] == forming)
                  & (run["band_reaction_times"] > SETTLE))
    present = means[names.index(complex_)]
    print(f"{complex_}: formed {made / span_settled:.1f} times per ps after "
          f"10 ps, {present:.2f} present on average: each lasts "
          f"{present / (made / span_settled):.3f} ps")
print(f"AB fell from 250 to {means[names.index('AB')]:.1f} on average, "
      f"{250 - means[names.index('AB')]:.0f} net breaks")

def bonded_integral(beta):
    """∫ 4πr² e^{−βφ} dr from 0.5 Å (where φ is huge) out to r_on."""
    return quad(lambda x: 4 * np.pi * x * x
                * np.exp(-beta * float(morse_pair(np.array(x))[0])),
                0.5, MORSE_ON, limit=200)[0]


constants = {}
for temperature in TEMPERATURES:
    r = np.load(RUNS / f"gas_{temperature}.npz")
    beta = 1 / (units.KB * temperature)
    k_int = bonded_integral(beta)
    late = r["times"] > SETTLE
    for key in ("band", "single"):
        sp = list(r[f"{key}_species"])
        c = r[f"{key}_counts"][late].astype(float)
        ratio = (c[:, sp.index("AB")] * V
                 / (c[:, sp.index("A")] * c[:, sp.index("B")]))
        blocks = [b.mean() for b in np.array_split(ratio, 9)]
        constants[temperature, key] = (np.mean(blocks),
                                       np.std(blocks, ddof=1) / 3, k_int)
    mean, err, _ = constants[temperature, "single"]
    band, band_err, _ = constants[temperature, "band"]
    print(f"{temperature} K: ∫ = {k_int:.0f} Å³; single cutoff "
          f"{mean:.0f} ± {err:.0f} ({mean / k_int:.2f}); band {band:.0f} "
          f"± {band_err:.0f} ({band / k_int:.2f})")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
ax.plot(lifetimes / 1000, per_ps["single"], "o-", color=OXBLOOD, ms=2.5,
        lw=0.8, label="one cutoff")
ax.plot(lifetimes / 1000, per_ps["band"], "o-", color=ACCENT, ms=2.5, lw=0.8,
        label="band")
ax.set_yscale("log")
ax.set_yticks([30, 100, 300])
ax.set_yticklabels(["30", "100", "300"])
ax.minorticks_off()
ax.set_xlabel("least lifetime / ps")
ax.set_ylabel("events per ps")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
t = run["times"] / 1000
ax.plot(t, counts[:, names.index("AB")], color=ACCENT, lw=0.7, label="AB")
ax.plot(t, counts[:, names.index("A")], color=OCHRE, lw=0.7, label="A")
larger = counts.sum(1) - counts[:, [names.index(s) for s in
                                    ("AB", "A", "B")]].sum(1)
ax.plot(t, larger, color=OXBLOOD, lw=0.7, label="larger")
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel("number")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "b")

ax = axes[2]
inverse = np.array([1 / (units.KB * t) for t in TEMPERATURES])
grid = np.linspace(inverse.min() - 0.2, inverse.max() + 0.2, 50)
ints = [bonded_integral(b) for b in grid]
ax.semilogy(grid, ints, color=REFERENCE, ls="--", lw=0.8)
for key, colour, marker in (("band", ACCENT, "o"), ("single", OXBLOOD, "s")):
    m = [constants[t, key][0] for t in TEMPERATURES]
    e = [constants[t, key][1] for t in TEMPERATURES]
    ax.errorbar(inverse, m, e, fmt=marker, color=colour, ms=2.5, lw=0.7,
                label="band" if key == "band" else "one cutoff")
ax.set_xlabel(r"$1/k_{\mathrm B}T$ / eV$^{-1}$")
ax.set_ylabel(r"$n_{\mathrm{AB}}V/n_{\mathrm A}n_{\mathrm B}$ / \AA$^3$")
ax.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "bonds.pdf")))
