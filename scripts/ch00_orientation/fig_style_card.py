"""Style card: the colour roles and the element colours.

Drawn through the same path as every book figure (house stylesheet,
final width, PGF save).

Not placed in the book; it is the check that the figure toolchain works
and the reference sheet for the colour law.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import (
    CYCLE,
    ELEMENT,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

ROLES = ("method", "second", "third", "fourth", "neutral")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(width_ratios=(1.25, 1))
)

t = np.linspace(0.0, 1.0, 200)
for k, (colour, role) in enumerate(zip(CYCLE[:4], ROLES[:4], strict=True)):
    left.plot(t, np.exp(-(k + 1) * t) + 0.05 * k, color=colour, label=role)
left.plot(t, np.exp(-0.5 * t), label="reference", **REFERENCE_STYLE)
left.axhline(0.2, label="threshold", **THRESHOLD_STYLE)
left.set_xlabel(r"time $t$ / fs")
left.set_ylabel(r"signal $A(t)$")
left.legend(ncols=2, loc="upper right")
viz.panel_tag(left, "a")

symbols = list(ELEMENT)
for k, symbol in enumerate(symbols):
    spec = ELEMENT[symbol]
    right.scatter(
        k,
        0,
        s=900 * spec["radius"] ** 2,
        color=spec["colour"],
        edgecolor="#202A33",
        linewidth=0.4,
    )
    right.text(k, -0.9, symbol, ha="center", va="top")
right.set_xlim(-0.7, len(symbols) - 0.3)
right.set_ylim(-1.6, 1.0)
right.axis("off")
viz.panel_tag(right, "b")

print(
    "wrote", viz.save(fig, figure_path("ch00_orientation", "style_card.pdf"))
)
