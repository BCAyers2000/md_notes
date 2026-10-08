"""Figure fig:md-ewald: the Ewald sum of rock salt.

Rock salt in its cubic cell of side 2d with k_e = 1 and d = 1, so that
the energy per ion pair is −𝓜, the Madelung constant 1.747565 (Evjen
1932). (a) The real-space, reciprocal-space and self parts of the energy
per ion pair against the splitting parameter α, and their sum, which does
not move. (b) The error of the sum against the size ε of the terms left
out, at α = 2/d.

Prints the numbers of Section 10.3.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import ewald, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE, figure_path

FCC = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
POS = np.vstack([FCC, FCC + [1.0, 0.0, 0.0]])
Q = np.array([1.0] * 4 + [-1.0] * 4)
H = 2 * np.eye(3)


def per_pair(alpha, accuracy=1e-12):
    e, _, parts = ewald.ewald_energy_forces(
        Q, POS, H, alpha=alpha, accuracy=accuracy, coulomb=1.0
    )
    return e / 4, {k: v / 4 for k, v in parts.items()}


alphas = np.linspace(0.6, 5.0, 45)
rows = [per_pair(a) for a in alphas]
total = np.array([e for e, _ in rows])
parts = {
    k: np.array([p[k] for _, p in rows])
    for k in ("real", "reciprocal", "self")
}
print(
    f"Madelung constant {-total.mean():.7f}, spread over alpha "
    f"{np.ptp(total):.1e}"
)
for a in (1.0, 2.0, 4.0):
    e, p = per_pair(a)
    print(
        f"alpha = {a}: real {p['real']:.4f}, reciprocal "
        f"{p['reciprocal']:.4f}, self {p['self']:.4f}, sum {e:.6f}"
    )

reference = per_pair(2.0, 1e-15)[0]
eps = np.logspace(-1, -12, 23)
errors = np.array([abs(per_pair(2.0, x)[0] - reference) for x in eps])
for x in (1e-2, 1e-4, 1e-8):
    print(
        f"accuracy {x:g}: error {abs(per_pair(2.0, x)[0] - reference):.1e}"
        f", r_cut = {np.sqrt(-np.log(x)) / 2.0:.2f} d, G_cut = "
        f"{2 * 2.0 * np.sqrt(-np.log(x)):.1f}/d"
    )

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.4)
)
for key, colour, label in (
    ("real", ACCENT, "real space"),
    ("reciprocal", OCHRE, "reciprocal space"),
    ("self", OXBLOOD, "self"),
):
    left.plot(alphas, parts[key], color=colour, lw=1.0, label=label)
left.plot(alphas, total, color="black", lw=1.4, label="sum")
left.set_xlabel(r"$\alpha d$")
left.set_ylabel(r"energy per ion pair / $(k_{\mathrm e}/d)$")
left.legend(loc="lower left", fontsize=7)
viz.panel_tag(left, "a")

right.loglog(eps, errors + 1e-17, "o", color=ACCENT, ms=3)
right.loglog(eps, eps, **REFERENCE_STYLE)
right.set_xlabel("size of the terms left out")
right.set_ylabel(r"error per ion pair / $(k_{\mathrm e}/d)$")
right.invert_xaxis()
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch10_code", "ewald.pdf")))
