"""Figure fig:lo-sgd: gradients from mini-batches.

The A-B molecule at 300 K (ch18.bond_run, 2001 readings of stretch and
force): the force fitted by a straight line in the scaled stretch s =
x/σ_x, with the mean squared miss as the loss. (a) The loss over all the
readings against the passes through the data (epochs) for gradient descent
on all of them at a time, for mini-batches of 10 at a fixed rate, and for
mini-batches of 10 with a rate that falls as 1/(1 + epoch/5); the least
loss, by least squares, dotted. (b) The first weight's gradient from 1000
mini-batches of 10 at the least-squares weights, where the full gradient
is zero, against the Gaussian with the spread σ_g/√10 that a mean of 10
independent readings has (dashed).

Prints the least loss, each run's loss after 50 epochs above it, the
spread of the mini-batch gradient against σ_g/√b, and the cost in
gradient evaluations.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import bond_run
from matplotlib.ticker import NullFormatter

from mdlab import learn, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

DEGREE, BATCH, EPOCHS = 1, 10, 50

x, f = bond_run(300, 20.0, 0)
X = learn.polynomial_basis(x / x.std(), DEGREE)
n = len(f)
w_best = learn.least_squares(X, f)
floor = np.mean((X @ w_best - f) ** 2)


def loss(w, rows=slice(None)):
    return np.mean((X[rows] @ w - f[rows]) ** 2)


def gradient(w, rows=slice(None)):
    m = len(f[rows])
    return 2 * X[rows].T @ (X[rows] @ w - f[rows]) / m


hessian = 2 * X.T @ X / n
rate = 1.0 / np.linalg.eigvalsh(hessian).max()  # half the stability limit
print(f"least loss {floor:.3e} (eV/Å)²; the rate {rate:.4f}, half of "
      f"2/λ_max of the loss's Hessian")


def train(batch, falling, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(DEGREE + 1)
    history = [loss(w)]
    for epoch in range(EPOCHS):
        eta = rate / (1 + epoch / 5) if falling else rate
        if batch is None:
            w = w - eta * gradient(w)
        else:
            for rows in learn.batches(n, batch, rng):
                w = w - eta * gradient(w, rows)
        history.append(loss(w))
    return np.array(history)


runs = {"all readings": train(None, False),
        f"batches of {BATCH}": train(BATCH, False),
        f"batches of {BATCH}, falling rate": train(BATCH, True)}
for name, history in runs.items():
    close = np.flatnonzero(history - floor < 1e-6)
    print(f"{name}: loss after {EPOCHS} epochs {history[-1] - floor:.2e} "
          f"above the least; within 1e-6 of it from epoch "
          f"{close[0] if len(close) else 'never'}")
print(f"gradient evaluations of one reading: {EPOCHS * n} for each run; "
      f"steps: {EPOCHS} for all readings, {EPOCHS * int(np.ceil(n / BATCH))}"
      f" for batches of {BATCH}")

rng = np.random.default_rng(1)
samples = np.array([gradient(w_best, rows)[1] for rows in
                    (rng.choice(n, BATCH, replace=False)
                     for _ in range(1000))])
each = 2 * X[:, 1] * (X @ w_best - f)  # one reading's gradient, weight 1
print(f"mini-batch gradient of the slope's weight: mean {samples.mean():.2e},"
      f" spread {samples.std():.4f}; σ_g/√{BATCH} = "
      f"{each.std() / np.sqrt(BATCH):.4f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                         gridspec_kw=dict(wspace=0.35))
ax = axes[0]
for (name, history), colour in zip(runs.items(), (ACCENT, OXBLOOD, OCHRE),
                                   strict=True):
    ax.semilogy(np.arange(EPOCHS + 1), history, color=colour, lw=0.9,
                label=name)
ax.axhline(floor, **REFERENCE_STYLE, lw=0.8)
ax.set_ylim(0.012, 0.7)
ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5],
             ["0.02", "0.05", "0.1", "0.2", "0.5"])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("epoch")
ax.set_ylabel(r"loss / (eV\,\AA$^{-1}$)$^2$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.hist(samples, bins=40, density=True, color=ACCENT, alpha=0.6)
g = np.linspace(samples.min(), samples.max(), 200)
sd = each.std() / np.sqrt(BATCH)
ax.plot(g, np.exp(-0.5 * (g / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
        **REFERENCE_STYLE, lw=0.8)
ax.set_xlabel(r"slope gradient / eV\,\AA$^{-1}$")
ax.set_ylabel(r"density / (eV\,\AA$^{-1}$)$^{-1}$")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "sgd.pdf")))
