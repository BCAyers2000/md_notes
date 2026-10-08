"""Figure fig:pe-pairs: three pair potentials with the same well.

The Lennard-Jones, Morse and Buckingham (exp-6) energies, each scaled so
that its minimum lies at r_m with depth ε: (a) φ/ε against r/r_m near
the well, (b) the wall at short range on a scale that is logarithmic
beyond ±1. The Morse width is a r_m = 6, which gives
the Lennard-Jones curvature at the minimum, 72 ε/r_m²; the exp-6 steepness
is α = 14. Near the minimum the three agree; they differ in the wall and
in the tail. The exp-6 energy turns over at short range.

Prints the numbers of Section 8.3.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from mdlab import potentials, units, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

ALPHA = 14.0
SIGMA = 2 ** (-1 / 6)  # LJ with its minimum at r = 1


def lj(r):
    return potentials.lennard_jones(r, 1.0, SIGMA)


def morse(r):
    return potentials.morse(r, 1.0, 6.0, 1.0)


def exp6(r):
    a = 6 / (ALPHA - 6) * math.exp(ALPHA)
    c = ALPHA / (ALPHA - 6)
    return potentials.buckingham(r, a, 1 / ALPHA, c)


h = 1e-4
for name, f in (("LJ", lj), ("Morse", morse), ("exp-6", exp6)):
    phi, slope = f(1.0)
    k = (f(1 + h)[0] - 2 * phi + f(1 - h)[0]) / h**2
    print(
        f"{name}: phi(r_m) = {float(phi):.6f}, slope {float(slope):.1e}, "
        f"curvature {float(k):.2f} eps/r_m², phi(2 r_m) = "
        f"{float(f(2.0)[0]):.4f}"
    )
r_top = brentq(lambda r: float(exp6(r)[1]), 0.1, 0.9)
print(
    f"exp-6 turns over at r = {r_top:.4f} r_m, where phi = "
    f"{float(exp6(r_top)[0]):.2f} eps; phi(0.1 r_m) = "
    f"{float(exp6(0.1)[0]):.3e} eps"
)
print(f"Morse at r = 0: {float(morse(0.0)[0]):.4e} eps")
print(f"LJ minimum at 2^(1/6) sigma = {2 ** (1 / 6):.6f} sigma")
eps_argon = 120.0 * units.KB  # eV
sigma_argon, mass_argon = 3.4, 39.948  # Å, amu
tau = sigma_argon * math.sqrt(mass_argon * units.MV2_TO_EV / eps_argon)
print(
    f"argon: eps = {eps_argon:.5f} eV, r_m = {2 ** (1 / 6) * 3.4:.3f} Å, "
    f"tau = sigma sqrt(m/eps) = {tau:.1f} fs"
)
print(f"LJ stiffness 36 2^(2/3) = {36 * 2 ** (2 / 3):.2f} eps/sigma²")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.36)
)
CURVES = (
    (lj, ACCENT, "Lennard-Jones"),
    (morse, OCHRE, "Morse"),
    (exp6, OXBLOOD, "exp-6"),
)
r = np.linspace(0.85, 2.6, 600)
for f, colour, label in CURVES:
    left.plot(r, f(r)[0], color=colour, label=label)
left.axhline(0, **THRESHOLD_STYLE)
left.set_xlim(0.85, 2.6)
left.set_ylim(-1.2, 1.5)
left.set_xlabel(r"$r / r_{\mathrm{m}}$")
left.set_ylabel(r"$\varphi / \varepsilon$")
left.legend(loc="upper right")
viz.panel_tag(left, "a")

r = np.linspace(0.02, 1.0, 2000)
for f, colour, _ in CURVES:
    right.plot(r, f(r)[0], color=colour)
right.set_yscale("symlog", linthresh=1.0)
right.axhline(0, **THRESHOLD_STYLE)
right.set_xlim(0.0, 1.0)
right.set_ylim(-1e8, 1e12)
ticks = [-1e8, -1e4, 0, 1e4, 1e8, 1e12]
right.set_yticks(
    ticks,
    [r"$-10^{8}$", r"$-10^{4}$", "0", r"$10^{4}$", r"$10^{8}$", r"$10^{12}$"],
)
right.yaxis.set_minor_locator(plt.NullLocator())
right.set_xlabel(r"$r / r_{\mathrm{m}}$")
right.set_ylabel(r"$\varphi / \varepsilon$")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch08_potentials", "pairs.pdf")))
