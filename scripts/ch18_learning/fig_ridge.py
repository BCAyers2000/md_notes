"""Figure fig:lo-ridge: a penalty on large weights.

The readings of fig_basis.py (15 training readings of argon's pair energy
with noise of 0.2 meV, the same seed), 15 more drawn the same way for
validation, and the noise-free test. The polynomial of degree 14 fitted
by least squares with the ridge penalty λ|w|², λ from 10⁻¹² to 10. (a)
The rms miss on the training, validation and test readings against λ.
(b) The fit at the λ of least validation miss against the fit without
the penalty, and the readings.

Prints the misses at each λ, the λ chosen, its test miss against the best
of fig_basis.py, the weights' size with and without the penalty, and the
spread of the weights σ_w = σ/√λ that the Bayesian reading assigns to it.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import NOISE, R_HIGH, R_LOW, argon_pair, basis, readings

from mdlab import learn, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

DEGREE = 14
LAMBDAS = 10.0 ** np.arange(-12, 1.01, 0.5)

rng = np.random.default_rng(182)
r_train, e_train = readings(15, rng)
r_valid, e_valid = readings(15, rng)
r_test = np.linspace(R_LOW, R_HIGH, 200)
e_test = argon_pair(r_test)
misses = {"training": [], "validation": [], "test": []}
for lam in LAMBDAS:
    w = learn.least_squares(basis(r_train, DEGREE), e_train, ridge=lam)
    for name, r, e in (("training", r_train, e_train),
                       ("validation", r_valid, e_valid),
                       ("test", r_test, e_test)):
        misses[name].append(np.sqrt(np.mean((basis(r, DEGREE) @ w - e)**2)))
misses = {k: np.array(v) for k, v in misses.items()}
for lam, a, b, c in zip(LAMBDAS, *misses.values(), strict=True):
    print(f"λ = {lam:.1e}: rms miss {a * 1000:.4f} (training), "
          f"{b * 1000:.4f} (validation), {c * 1000:.4f} meV (test)")
k = int(np.argmin(misses["validation"]))
chosen = LAMBDAS[k]
w_free = learn.least_squares(basis(r_train, DEGREE), e_train)
w_ridge = learn.least_squares(basis(r_train, DEGREE), e_train, ridge=chosen)
print(f"chosen λ = {chosen:.1e} by validation: test miss "
      f"{misses['test'][k] * 1000:.3f} meV; least test miss over λ "
      f"{misses['test'].min() * 1000:.3f} meV")
print(f"|w| without the penalty {np.linalg.norm(w_free):.3e} eV, with it "
      f"{np.linalg.norm(w_ridge):.3e} eV; σ_w = σ/√λ = "
      f"{NOISE / np.sqrt(chosen) * 1000:.2f} meV")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                         gridspec_kw=dict(wspace=0.35))
ax = axes[0]
for (name, values), colour, mark in zip(misses.items(),
                                        (ACCENT, OCHRE, OXBLOOD),
                                        ("o-", "^-", "s-"), strict=True):
    ax.loglog(LAMBDAS, values * 1000, mark, ms=2.5, lw=0.8, color=colour,
              label=name)
ax.axvline(chosen, **REFERENCE_STYLE, lw=0.8)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("rms miss / meV")
ax.legend(fontsize=7, loc="upper right", frameon=True,
          framealpha=1.0, facecolor="white", edgecolor="none")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.plot(r_test, e_test * 1000, **REFERENCE_STYLE, lw=0.8)
ax.plot(r_train, e_train * 1000, "o", ms=3, color="k", zorder=5)
ax.plot(r_test, basis(r_test, DEGREE) @ w_free * 1000, color=OXBLOOD,
        lw=0.8, label=r"$\lambda = 0$")
ax.plot(r_test, basis(r_test, DEGREE) @ w_ridge * 1000, color=ACCENT,
        lw=1.0, label=rf"$\lambda = 10^{{{np.log10(chosen):.0f}}}$")
ax.set_ylim(-14, 8)
ax.set_xlabel(r"$r$ / \AA")
ax.set_ylabel(r"$\varphi(r)$ / meV")
ax.legend(fontsize=7, loc="upper right", frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.95)
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "ridge.pdf")))
