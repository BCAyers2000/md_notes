"""Figure fig:lo-training: what the training looks like.

The readings of fig_ridge.py (15 for training and 15 for validation of
argon's pair energy, noise 0.2 meV), fitted by lowering the mean squared
miss with an optimiser from zero weights. (a) Healthy: degree 8 in the
scaled distance, Adam at the rate 10⁻⁴: training and validation misses
fall together to near the noise. (b) The rate: gradient descent on the
same fit at 0.9 and 1.1 times 2/λ_max of the loss's Hessian, and at
10⁻³ of it. (c) Overfitting: degree 14, Adam at 10⁻⁴, no penalty, for
10⁵ steps: the validation miss turns up while the training miss falls;
the step of least validation miss is where training would stop early.
(d) Unscaled features: degree 8 in r itself, gradient descent at its
stability limit, against the scaled features.

Prints each run's misses at the end, the largest rise of Adam's training
miss in one step at the rates 10⁻³ and 10⁻⁴, the step of least validation
miss in (c) and its misses there and at the end, and the condition numbers
of the two designs of (d).
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import NOISE, basis, readings
from matplotlib.ticker import FuncFormatter, NullFormatter

from mdlab import learn, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE

rng = np.random.default_rng(182)
r_train, e_train = readings(15, rng)
r_valid, e_valid = readings(15, rng)


def run(X, Xv, optimiser, steps, every=1):
    """Train from zero; rms misses (meV) on training and validation."""
    n = len(e_train)

    def loss_and_gradient(w):
        miss = X @ w - e_train
        return np.mean(miss**2), 2 * X.T @ miss / n

    path, _ = learn.minimise(loss_and_gradient, np.zeros(X.shape[1]),
                             optimiser, steps)
    path = path[::every]
    train = np.sqrt(np.mean((path @ X.T - e_train) ** 2, axis=1)) * 1000
    valid = np.sqrt(np.mean((path @ Xv.T - e_valid) ** 2, axis=1)) * 1000
    return train, valid


def stability(X):
    return 2 / np.linalg.eigvalsh(2 * X.T @ X / len(X)).max()


X8, V8 = basis(r_train, 8), basis(r_valid, 8)
healthy = run(X8, V8, learn.Adam(1e-4), 40000, every=10)
print(f"(a) degree 8, Adam 1e-4, 40000 steps: training {healthy[0][-1]:.3f}, "
      f"validation {healthy[1][-1]:.3f} meV (least {healthy[1].min():.3f});"
      f" noise {NOISE * 1000:.1f}")
for rate in (1e-3, 1e-4):
    tr, _ = run(X8, V8, learn.Adam(rate), 40000)
    print(f"Adam at {rate:g}: the largest rise of the training miss in one "
          f"step after the first 1000, {np.max(np.diff(tr[1000:])):.3f} meV")
limit = stability(X8)
rates = {0.9: run(X8, V8, learn.GradientDescent(0.9 * limit), 400),
         1.1: run(X8, V8, learn.GradientDescent(1.1 * limit), 400),
         1e-3: run(X8, V8, learn.GradientDescent(1e-3 * limit), 400)}
for k, (tr, _) in rates.items():
    print(f"(b) gradient descent at {k} × 2/λ_max = {k * limit:.3e}: "
          f"training miss after 400 steps {tr[-1]:.3g} meV")
X14, V14 = basis(r_train, 14), basis(r_valid, 14)
over = run(X14, V14, learn.Adam(1e-4), 100000, every=10)
stop = int(np.argmin(over[1]))
print(f"(c) degree 14, Adam 1e-4: least validation miss {over[1][stop]:.3f} "
      f"meV at step {stop * 10} (training {over[0][stop]:.3f}); at the end "
      f"training {over[0][-1]:.3f}, validation {over[1][-1]:.3f} meV")
R8, RV8 = learn.polynomial_basis(r_train, 8), learn.polynomial_basis(r_valid,
                                                                     8)
raw = run(R8, RV8, learn.GradientDescent(0.9 * stability(R8)), 2000)
scaled = run(X8, V8, learn.GradientDescent(0.9 * limit), 2000)
for label, X in (("r itself", R8), ("scaled", X8)):
    s = np.linalg.svd(X, compute_uv=False)
    print(f"(d) degree 8, {label}: condition number of XᵀX "
          f"{(s.max() / s.min()) ** 2:.1e}")
print(f"(d) after 2000 steps at 0.9 × the limit: training miss "
      f"{raw[0][-1]:.3f} meV (r itself), {scaled[0][-1]:.3f} meV (scaled)")

viz.use_style(notebook=False)
fig, axes = plt.subplots(2, 2, figsize=(viz.FULL, 3.6),
                         gridspec_kw=dict(wspace=0.4, hspace=0.6))
plain = FuncFormatter(lambda v, _: f"{v:g}")
ax = axes[0, 0]
steps = np.arange(len(healthy[0])) * 10
ax.loglog(steps[1:], healthy[0][1:], color=ACCENT, lw=0.9, label="training")
ax.loglog(steps[1:], healthy[1][1:], color=OXBLOOD, lw=0.9,
          label="validation")
ax.axhline(NOISE * 1000, **THRESHOLD_STYLE)
ax.set_xlabel("step")
ax.set_ylabel("rms miss / meV")
ax.set_title("healthy", fontsize=7)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(plain)
    axis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.2, 0.5, 1, 2, 5])
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[0, 1]
for (k, (tr, _)), colour in zip(rates.items(), (ACCENT, OXBLOOD, OCHRE),
                                strict=True):
    ax.semilogy(tr, color=colour, lw=0.9,
                label=rf"${k:g}\times2/\lambda_{{\max}}$")
ax.axhline(NOISE * 1000, **THRESHOLD_STYLE)
ax.set_ylim(1e-1, 1e4)
ax.set_xlabel("step")
ax.set_ylabel("training miss / meV")
ax.set_title("the rate", fontsize=7)
ax.yaxis.set_major_formatter(plain)
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "b")

ax = axes[1, 0]
steps = np.arange(len(over[0])) * 10
ax.loglog(steps[1:], over[0][1:], color=ACCENT, lw=0.9, label="training")
ax.loglog(steps[1:], over[1][1:], color=OXBLOOD, lw=0.9, label="validation")
ax.axvline(stop * 10, color="0.6", lw=0.8)
ax.axhline(NOISE * 1000, **THRESHOLD_STYLE)
ax.set_xlabel("step")
ax.set_ylabel("rms miss / meV")
ax.set_title("overfitting", fontsize=7)
ax.set_ylim(top=20)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(plain)
    axis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "c")

ax = axes[1, 1]
ax.semilogy(raw[0], color=OXBLOOD, lw=0.9, label=r"powers of $r$")
ax.semilogy(scaled[0], color=ACCENT, lw=0.9, label="scaled distance")
ax.axhline(NOISE * 1000, **THRESHOLD_STYLE)
ax.set_xlabel("step")
ax.set_ylabel("training miss / meV")
ax.set_title("unscaled features", fontsize=7)
ax.yaxis.set_major_formatter(plain)
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.2, 0.5, 1, 2, 5])
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "d")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "training.pdf")))
