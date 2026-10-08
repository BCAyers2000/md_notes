"""Figure fig:os-period: the period grows with the energy.

The period of the motion in each well of fig:os-wells divided by the
period of small swings, T0 = 2 pi / omega0, against the energy above the
bottom as a fraction of the depth, computed by oscillators.period. The
cosine well's period grows without bound as the energy approaches the
barrier; so do the others as the energy approaches the depth, where the
pair comes apart. The harmonic well (dotted) keeps T = T0.

Prints the periods quoted in Section 4.5 (sec:os-anharmonic).
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import oscillators, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

FRACTIONS = np.concatenate(
    [np.linspace(0.01, 0.95, 48), 1 - np.logspace(np.log10(0.05), -4, 24)]
)


def cosine(x):
    return float(oscillators.cosine_well(x, 1.0, 1.0)[0])


def morse(y):
    return float(oscillators.morse(y, 1.0, 1.0, 0.0)[0])


def lj(r):
    return float(oscillators.lennard_jones(r, 1.0, 1.0)[0]) + 1.0


WELLS = (
    (
        "cosine",
        cosine,
        0.0,
        0.002,
        2 * np.pi / (2 * np.pi / np.sqrt(2)),
        ACCENT,
    ),
    ("Morse", morse, 0.0, 0.01, 2 * np.pi / np.sqrt(2), OCHRE),
    (
        "Lennard-Jones",
        lj,
        2 ** (1 / 6),
        0.002,
        2 * np.pi / np.sqrt(72 / 2 ** (1 / 3)),
        OXBLOOD,
    ),
)
# T0 = 2π/√(U''/m) with m = 1: U'' = 2π²/Λ² · D for the cosine well
# (Λ = D = 1), 2Da² = 2 for Morse, 72ε/(2^(1/3)σ²) for Lennard-Jones

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.FULL * 0.62, 2.5))
for (name, well, bottom, step, t0, colour), linestyle in zip(
    WELLS, ("-", "--", "-."), strict=True
):
    ratios = [
        oscillators.period(
            well, f, 1.0, bottom, step, force_to_accel=1.0, nodes=400
        )
        / t0
        for f in FRACTIONS
    ]
    ax.plot(FRACTIONS, ratios, color=colour, label=name, ls=linestyle)
    for f in (0.1, 0.5, 0.9):
        r = (
            oscillators.period(
                well, f, 1.0, bottom, step, force_to_accel=1.0, nodes=400
            )
            / t0
        )
        print(f"{name:14s} E = {f} depth: T/T0 = {r:.3f}")
ax.axhline(1.0, **THRESHOLD_STYLE)
ax.set_xlim(0, 1)
ax.set_ylim(0.9, 6)
ax.set_xlabel(r"energy above the bottom / depth")
ax.set_ylabel(r"period / small-swing period")
ax.legend(loc="upper left")

print("wrote", viz.save(fig, figure_path("ch04_oscillations", "period.pdf")))
