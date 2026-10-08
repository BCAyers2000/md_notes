"""Figure fig:mo-sinusoid: the sinusoid x(t) = A sin(omega t + phi).

A = 1.5 m, omega = pi rad/s (period 2 s) and phi = pi/3, against the
unshifted A sin(omega t) (dashed). Arrows mark the amplitude, the period
and the shift phi/omega by which every crest arrives earlier.

Prints the numbers that Section 1.5 (sec:mo-trig) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, figure_path

A, OMEGA, PHI = 1.5, np.pi, np.pi / 3
PERIOD = 2 * np.pi / OMEGA
SHIFT = PHI / OMEGA
print(
    f"period 2 pi / omega = {PERIOD:.3f} s, frequency = {1 / PERIOD:.3f} "
    f"Hz, shift phi / omega = {SHIFT:.4f} s"
)
crest = (np.pi / 2 - PHI) / OMEGA  # first crest of the shifted wave
crest_unshifted = (np.pi / 2) / OMEGA  # first crest of A sin(omega t)
print(
    f"first crests: shifted at t = {crest:.4f} s, unshifted at "
    f"{crest_unshifted:.4f} s"
)

# a double-headed arrow that marks a length on the plot
DIMENSION = dict(
    arrowstyle="<|-|>",
    color="black",
    lw=0.7,
    mutation_scale=7,
    shrinkA=0,
    shrinkB=0,
)

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.FULL, 2.3))
t = np.linspace(0.0, 4.2, 600)
ax.axhline(0.0, color="black", lw=0.4)
ax.plot(
    t,
    A * np.sin(OMEGA * t),
    lw=1.0,
    label=r"$A\sin\omega t$",
    **REFERENCE_STYLE,
)
ax.plot(
    t,
    A * np.sin(OMEGA * t + PHI),
    color=ACCENT,
    label=r"$A\sin(\omega t + \phi)$",
)
# amplitude, at the second crest of the shifted wave
tc2 = crest + PERIOD
ax.annotate("", (tc2, 0.0), (tc2, A), arrowprops=DIMENSION)
ax.text(tc2 + 0.05, A / 2, r"$A$", va="center")
# period, between two crests, drawn above them
ax.annotate(
    "", (crest, A + 0.3), (crest + PERIOD, A + 0.3), arrowprops=DIMENSION
)
ax.text(
    crest + PERIOD / 2,
    A + 0.38,
    r"period $2\pi/\omega$",
    ha="center",
    va="bottom",
    fontsize=9,
)
for tc in (crest, crest + PERIOD):
    ax.plot([tc, tc], [A, A + 0.3], color="black", lw=0.4)
# shift between the crests of the two waves
ax.annotate(
    "", (crest, -A - 0.3), (crest_unshifted, -A - 0.3), arrowprops=DIMENSION
)
ax.text(
    (crest + crest_unshifted) / 2,
    -A - 0.38,
    r"$\phi/\omega$",
    ha="center",
    va="top",
    fontsize=9,
)
for tc, y0 in ((crest, A), (crest_unshifted, A)):
    ax.plot([tc, tc], [y0, -A - 0.3], color="black", lw=0.4, ls=":")
ax.set_xlim(0.0, 4.2)
ax.set_ylim(-2.4, 2.4)
ax.set_xlabel(r"time $t$ / s")
ax.set_ylabel(r"$x$ / m")
ax.legend(
    loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, borderaxespad=0.1
)

print("wrote", viz.save(fig, figure_path("ch01_motion", "sinusoid.pdf")))
