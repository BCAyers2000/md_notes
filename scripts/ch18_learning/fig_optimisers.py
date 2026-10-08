"""Figure fig:lo-optimisers: momentum and Adam in a narrow valley.

The loss L = ½(λ₁u₁² + λ₂u₂²) of two weights, λ₁ = 1 and λ₂ = 100, with u
the weights measured along the valley's axes, from w = (−2.5, 0.5).
Gradient descent at its best fixed rate 2/(λ₁ + λ₂); momentum at the rate
and momentum of Polyak, 4/(√λ₁ + √λ₂)² and ((√κ − 1)/(√κ + 1))²; Adam with
the rate 0.05. (a) The steep weight w₂ against the step in the valley with
its axes along the weights. (b) The loss against the step there, with the
contractions per step that the theory gives gradient descent and momentum
(dotted). (c) The loss against the step in the same valley turned by 45°,
so that its axes lie between the weights.

Prints the rates, the factors per step (κ − 1)/(κ + 1) and (√κ − 1)/
(√κ + 1), and the steps each optimiser needs to bring the loss below
10⁻⁸ in each valley.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import learn, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

L1, L2 = 1.0, 100.0
KAPPA = L2 / L1
START = np.array([-2.5, 0.5])
STEPS = 300


def valley(turn):
    c, s = np.cos(turn), np.sin(turn)
    R = np.array([[c, -s], [s, c]])
    A = R @ np.diag([L1, L2]) @ R.T
    return A, lambda w: (0.5 * w @ A @ w, A @ w)


gd_rate = 2 / (L1 + L2)
heavy_rate = 4 / (np.sqrt(L1) + np.sqrt(L2)) ** 2
heavy_mu = ((np.sqrt(KAPPA) - 1) / (np.sqrt(KAPPA) + 1)) ** 2
gd_factor = (KAPPA - 1) / (KAPPA + 1)
heavy_factor = (np.sqrt(KAPPA) - 1) / (np.sqrt(KAPPA) + 1)
print(f"κ = {KAPPA:.0f}; gradient descent at η = {gd_rate:.4f}, factor "
      f"{gd_factor:.4f}; momentum at η = {heavy_rate:.4f}, μ = "
      f"{heavy_mu:.4f}, factor {heavy_factor:.4f}")


def optimisers():
    return {"gradient descent": learn.GradientDescent(gd_rate),
            "momentum": learn.Momentum(heavy_rate, momentum=heavy_mu),
            "Adam": learn.Adam(0.05)}


results = {}
for label, turn in (("along the weights", 0.0), ("turned by 45°",
                                                 np.pi / 4)):
    _, fn = valley(turn)
    results[label] = {name: learn.minimise(fn, START, opt, STEPS)
                      for name, opt in optimisers().items()}
    for name, (_, loss) in results[label].items():
        below = np.flatnonzero(loss < 1e-8)
        print(f"{label}, {name}: loss below 10⁻⁸ after "
              f"{below[0] if len(below) else 'more than ' + str(STEPS)} "
              f"steps; {loss[-1]:.2g} after {STEPS}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.2),
                         gridspec_kw=dict(wspace=0.55))
colours = (ACCENT, OCHRE, OXBLOOD)
ax = axes[0]
for (name, (path, _)), colour in zip(results["along the weights"].items(),
                                     colours, strict=True):
    ax.plot(path[:40, 1], "o-", ms=1.5, lw=0.7, color=colour, label=name)
ax.axhline(0, color="0.75", lw=0.5)
ax.set_xlabel("step")
ax.set_ylabel(r"steep weight $w_2$")
fig.legend(*ax.get_legend_handles_labels(), fontsize=7, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, -0.14))
viz.panel_tag(ax, "a")

for ax, label, tag in ((axes[1], "along the weights", "b"),
                       (axes[2], "turned by 45°", "c")):
    for (name, (_, loss)), colour in zip(results[label].items(), colours,
                                         strict=True):
        ax.semilogy(loss, lw=0.9, color=colour, label=name)
    if tag == "b":
        t = np.arange(STEPS + 1)
        loss0 = results[label]["gradient descent"][1][0]
        for factor in (gd_factor, heavy_factor):
            ax.semilogy(t, loss0 * factor ** (2 * t), **REFERENCE_STYLE,
                        lw=0.6)
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"({tag}) {label}", fontsize=7, loc="left", pad=9)

print("wrote", viz.save(fig, viz.figure_path("ch18_learning",
                                             "optimisers.pdf")))
