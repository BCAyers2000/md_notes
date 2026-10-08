"""Figure fig:lo-basis: basis functions and overfitting.

Liquid argon's pair energy between 3.3 Å and 2σ = 6.8 Å, where it is the
plain Lennard-Jones energy, read at 15 distances drawn at random with
noise of 0.2 meV (the training readings), and at 200 equally spaced
distances without noise (the test). The model is a polynomial in the
scaled distance s = (r − 5.05 Å)/1.75 Å, which runs from −1 to 1, of
degree 0 to 14, fitted by least squares; and, for comparison, the two
functions (σ/r)⁶ and (σ/r)¹², the form the energy has. (a) The readings
and the fits of degree 3, 8 and 14. (b) The rms miss on the training
readings and on the test against the degree, with the noise (dotted) and
the two-function fit's test miss (cross); the fit of degree 14 passes
through every reading, and its training miss is drawn on the axis.

Prints the misses for every degree, the degree with the least test miss,
and the two-function fit's weights against 4ε and −4ε.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import NOISE, R_HIGH, R_LOW, argon_pair, basis, ch12, readings

from mdlab import learn, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE, THRESHOLD_STYLE

DEGREES = np.arange(15)


def physics(r):
    u = ch12.SIG / np.asarray(r)
    return np.c_[u**12, u**6]


rng = np.random.default_rng(182)
r_train, e_train = readings(15, rng)
r_test = np.linspace(R_LOW, R_HIGH, 200)
e_test = argon_pair(r_test)
train_miss, test_miss = [], []
for degree in DEGREES:
    w = learn.least_squares(basis(r_train, degree), e_train)
    train_miss.append(np.sqrt(np.mean((basis(r_train, degree) @ w
                                       - e_train) ** 2)))
    test_miss.append(np.sqrt(np.mean((basis(r_test, degree) @ w
                                      - e_test) ** 2)))
train_miss, test_miss = np.array(train_miss), np.array(test_miss)
for degree, a, b in zip(DEGREES, train_miss, test_miss, strict=True):
    print(f"degree {degree:2d}: rms miss {a * 1000:8.4f} meV on the "
          f"training readings, {b * 1000:10.4f} meV on the test")
best = int(DEGREES[np.argmin(test_miss)])
print(f"least test miss at degree {best}: {test_miss.min() * 1000:.3f} meV;"
      f" the noise {NOISE * 1000:.1f} meV")
w2 = learn.least_squares(physics(r_train), e_train)
miss2 = np.sqrt(np.mean((physics(r_test) @ w2 - e_test) ** 2))
print(f"(σ/r)¹² and (σ/r)⁶: weights {w2[0]:.5f} and {w2[1]:.5f} eV against "
      f"4ε = {4 * ch12.EPS:.5f} and −4ε; test miss {miss2 * 1000:.4f} meV")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                         gridspec_kw=dict(wspace=0.35))
ax = axes[0]
ax.plot(r_test, e_test * 1000, **REFERENCE_STYLE, lw=0.8)
ax.plot(r_train, e_train * 1000, "o", ms=3, color="k", zorder=5)
for degree, colour in ((3, ACCENT), (8, OCHRE), (14, OXBLOOD)):
    w = learn.least_squares(basis(r_train, degree), e_train)
    ax.plot(r_test, basis(r_test, degree) @ w * 1000, color=colour, lw=0.9,
            label=f"degree {degree}")
ax.set_ylim(-14, 8)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$\varphi(r)$ / meV")
ax.legend(fontsize=7, loc="upper right", frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.95)
viz.panel_tag(ax, "a")

ax = axes[1]
FLOOR = 1e-2  # meV: the exact fit of degree 14 is drawn on the axis
ax.semilogy(DEGREES, np.maximum(train_miss * 1000, FLOOR), "o-", ms=3,
            lw=0.8, color=ACCENT, label="training readings", clip_on=False)
ax.semilogy(DEGREES, test_miss * 1000, "s-", ms=3, lw=0.8, color=OXBLOOD,
            label="test")
ax.axhline(NOISE * 1000, **THRESHOLD_STYLE)
ax.semilogy([1], [miss2 * 1000], "x", ms=6, color=OCHRE,
            label=r"$(\sigma/r)^{12}$ and $(\sigma/r)^6$")
ax.set_ylim(FLOOR, 1e5)
ax.set_xlabel("degree of the polynomial")
ax.set_ylabel("rms miss / meV")
ax.legend(fontsize=7, loc="upper center")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "basis.pdf")))
