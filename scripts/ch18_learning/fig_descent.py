"""Figure fig:lo-descent: gradient descent and its stability limit.

The loss L(w) = ½(w − w*)ᵀA(w − w*) of two weights, A with eigenvalues 1
and 10 along axes turned by 30°, from w = (−2, 1.5) relative to w*. (a)
Gradient descent's paths for rates η of 0.05, 0.18 and 0.205 (2/λ_max =
0.2) over its contours. (b) The loss against the step for η of 0.05,
0.1, 0.18 and 0.205, with the term of the slowest direction,
½λc²(1 − ηλ)^(2t) for c the start's component along it (dotted).

Prints each rate's factors 1 − ηλ along the two eigenvectors, the best
fixed rate 2/(λ_min + λ_max) and its factor (κ − 1)/(κ + 1), and the steps
each rate needs to bring the loss down by 10⁶; then the condition number
of XᵀX for the polynomials of fig_basis.py, of degree 3, 8 and 14 in the
scaled distance and of degree 8 in r itself, and the steps the best fixed
rate would need for each.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import basis, readings

from mdlab import learn, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

LAMBDAS = np.array([1.0, 10.0])
TURN = np.radians(30)
R = np.array([[np.cos(TURN), -np.sin(TURN)], [np.sin(TURN), np.cos(TURN)]])
A = R @ np.diag(LAMBDAS) @ R.T
START = np.array([-2.0, 1.5])
RATES = (0.05, 0.1, 0.18, 0.205)


def loss_and_gradient(w):
    return 0.5 * w @ A @ w, A @ w


def steps_for(factor, drop=1e6):
    """Steps for the loss, which falls as factor², to fall by ``drop``."""
    return np.inf if factor >= 1 else np.log(drop) / (-2 * np.log(factor))


def steps_at_best(kappa, drop=1e6):
    """Steps at the best fixed rate, whose factor is (κ − 1)/(κ + 1)."""
    return np.log(drop) / (2 * np.log1p(2 / (kappa - 1)))


paths = {}
for rate in RATES:
    paths[rate] = learn.minimise(loss_and_gradient, START,
                                 learn.GradientDescent(rate), 60)
    factors = 1 - rate * LAMBDAS
    print(f"η = {rate}: factors {factors[0]:+.3f} and {factors[1]:+.3f}; the "
          f"loss falls 10⁶ times in {steps_for(np.abs(factors).max()):.0f} "
          f"steps")
best = 2 / LAMBDAS.sum()
kappa = LAMBDAS[1] / LAMBDAS[0]
print(f"best fixed rate 2/(λ_min + λ_max) = {best:.4f}, factor (κ − 1)/(κ + 1)"
      f" = {(kappa - 1) / (kappa + 1):.4f}, {steps_at_best(kappa):.0f} steps "
      "for 10⁶")

r, _ = readings(15, np.random.default_rng(182))
for label, X in (("scaled, degree 3", basis(r, 3)),
                 ("scaled, degree 8", basis(r, 8)),
                 ("scaled, degree 14", basis(r, 14)),
                 ("r itself, degree 8", learn.polynomial_basis(r, 8))):
    s = np.linalg.svd(X, compute_uv=False)  # XᵀX has the eigenvalues s²
    k = (s.max() / s.min()) ** 2
    print(f"XᵀX, {label}: condition number {k:.1e}; the best fixed rate "
          f"needs {steps_at_best(k):.1e} steps for 10⁶")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.4),
                         gridspec_kw=dict(wspace=0.35))
ax = axes[0]
g = np.linspace(-2.6, 2.6, 200)
X, Y = np.meshgrid(g, g)
W = np.stack([X, Y], axis=-1)
Z = 0.5 * np.einsum("...i,ij,...j->...", W, A, W)
ax.contour(X, Y, Z, levels=np.geomspace(0.02, 30, 10), colors="0.75",
           linewidths=0.5)
for rate, colour, shown in ((0.205, OXBLOOD, 9), (0.05, ACCENT, 25),
                            (0.18, OCHRE, 25)):
    path = paths[rate][0][:shown]
    ax.plot(path[:, 0], path[:, 1], "o-", ms=1.8, lw=0.7, color=colour,
            label=rf"$\eta = {rate}$")
ax.plot(0, 0, "+", color="k", ms=6)
ax.set_aspect("equal")
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.6, 2.6)
ax.set_xlabel(r"$w_1 - w_1^\ast$")
ax.set_ylabel(r"$w_2 - w_2^\ast$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
for rate, colour in zip(RATES, (ACCENT, "0.5", OCHRE, OXBLOOD), strict=True):
    loss = paths[rate][1]
    ax.semilogy(loss, color=colour, lw=0.9, label=rf"$\eta = {rate}$")
    # the term of the slowest direction, ½λc²(1 − ηλ)^(2t), c the start's
    # component along its eigenvector
    k = int(np.argmax(np.abs(1 - rate * LAMBDAS)))
    c = (R.T @ START)[k]
    t = np.arange(61)
    ax.semilogy(t, 0.5 * LAMBDAS[k] * c * c
                * (1 - rate * LAMBDAS[k]) ** (2 * t), **REFERENCE_STYLE,
                lw=0.6)
ax.set_ylim(1e-12, 1e3)
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.legend(fontsize=7, loc="lower left")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "descent.pdf")))
