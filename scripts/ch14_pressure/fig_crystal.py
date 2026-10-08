"""Figure fig:pr-crystal: the pressure of a crystal at rest.

The face-centred cubic crystal of the switched Lennard-Jones argon, 4 × 4
× 4 cubic cells (256 atoms), every atom on its site, so K = 0 and the
pressure is the virial's alone. (a) The energy per atom (ochre) and the
pressure (teal) against the cubic spacing a, the zero of pressure marked
(dotted). (b) The crystal at a₀ stretched along x by the strain ε_xx: the
components P_xx, P_yy (equal to P_zz) and P_xy (dashed, zero) of the
pressure tensor.

Prints a₀ from P = 0 and from the least energy, the bulk modulus from
the slope of P, the elastic constants C₁₁ and C₁₂ from (b), and
(C₁₁ + 2C₁₂)/3 against the bulk modulus.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch14 import GPA, SIG, argon_model, fcc
from scipy.optimize import brentq, minimize_scalar

from mdlab import viz
from mdlab.cell import cell_volume
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path


def state(a, strain=None):
    """Energy per atom and the pressure tensor of the strained crystal."""
    r, h = fcc(4, a)
    s = np.eye(3) + (np.zeros((3, 3)) if strain is None else strain)
    r, h = r @ s.T, s @ h
    model = argon_model(h)
    u, _ = model(r)
    return u / len(r), model.virial / cell_volume(h)


a0 = brentq(lambda a: np.trace(state(a)[1]) / 3, 5.0, 5.6, xtol=1e-12)
a_min = minimize_scalar(lambda a: state(a)[0], bounds=(5.1, 5.4),
                        method="bounded", options=dict(xatol=1e-9)).x
print(f"a0 from P = 0: {a0:.5f} Å = {a0 / SIG:.5f} σ; from the least "
      f"energy: {a_min:.5f} Å")
print(f"energy per atom at a0: {state(a0)[0]:.6f} eV")
step = 1e-4


def pressure(a):
    return np.trace(state(a)[1]) / 3


# B = −V dP/dV, with V ∝ a³, so dV/V = 3 da/a
bulk = -a0 * (pressure(a0 + step) - pressure(a0 - step)) / (3 * 2 * step)
print(f"bulk modulus B = −V dP/dV at a0: {bulk * GPA:.3f} GPa")
for a in (5.20, 5.40):
    print(f"  P at a = {a} Å: {pressure(a) * GPA:+.4f} GPa")


def tensor(exx):
    e = np.zeros((3, 3))
    e[0, 0] = exx
    return state(a0, e)[1]


eps = 1e-5
dp = (tensor(eps) - tensor(-eps)) / (2 * eps)
c11, c12 = -dp[0, 0], -dp[1, 1]
print(f"C11 = {c11 * GPA:.3f} GPa, C12 = {c12 * GPA:.3f} GPa; "
      f"(C11 + 2 C12)/3 = {(c11 + 2 * c12) / 3 * GPA:.3f} GPa; "
      f"P_xy changes by {dp[0, 1] * GPA:.1e} GPa per unit strain")
for exx in (-0.02, 0.02):
    t = tensor(exx) * GPA
    print(f"  ε_xx = {exx:+.2f}: P_xx {t[0, 0]:+.4f}, P_yy {t[1, 1]:+.4f}, "
          f"P_zz {t[2, 2]:+.4f}, P_xy {t[0, 1]:+.1e} GPa")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                                 gridspec_kw=dict(wspace=0.95))
spacings = np.linspace(5.0, 5.8, 81)
states = [state(a) for a in spacings]
ax_a.plot(spacings, [np.trace(p) / 3 * GPA for _, p in states],
          color=ACCENT, lw=1.2)
ax_a.axhline(0.0, **THRESHOLD_STYLE)
ax_a.axvline(a0, **THRESHOLD_STYLE)
ax_a.set_xlabel(r"$a$ / Å")
ax_a.set_ylabel(r"$P$ / GPa", color=ACCENT)
twin = ax_a.twinx()
twin.plot(spacings, [u * 1000 for u, _ in states], color=OCHRE, lw=1.0)
twin.set_ylabel(r"$U/N$ / meV", color=OCHRE)
viz.panel_tag(ax_a, "a")

strains = np.linspace(-0.02, 0.02, 21)
tensors = np.array([tensor(e) for e in strains]) * GPA
ax_b.plot(strains * 100, tensors[:, 0, 0], color=ACCENT, lw=1.2,
          label=r"$P_{xx}$")
ax_b.plot(strains * 100, tensors[:, 1, 1], color=OCHRE, lw=1.0,
          label=r"$P_{yy} = P_{zz}$")
ax_b.plot(strains * 100, tensors[:, 0, 1], color=OXBLOOD, lw=1.0,
          ls="--", label=r"$P_{xy}$")
ax_b.set_xlabel(r"$\epsilon_{xx}$ / %")
ax_b.set_ylabel("pressure / GPa")
ax_b.legend(fontsize=6, loc="upper right", handlelength=2.5)
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, figure_path("ch14_pressure", "crystal.pdf")))
