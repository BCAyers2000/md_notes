"""Figure fig:lo-leak: consecutive frames leak between splits.

runs.py's two runs of liquid argon, a frame every 10 fs. From each frame,
the histogram of its 32 640 pair distances in 110 bins of 0.05 Å from 3.0
to 8.5 Å; the potential energy is linear in it, through the pair energy at
each bin, to 21 meV rms. Ridge regression (λ = 10⁻², the features scaled
by the training frames' mean and spread) learns U from the first n frames
of run a after its first 10 ps, for n from 200 to 2000 (2 to 20 ps), with
a fifth held out for validation either at random or in whole blocks of
50 frames (0.5 ps), ten splits each; the test is run b's frames every
50 fs after its first 10 ps, a run the model never saw. (a) U over 4 ps
of run a, with the frames a random split and a blocked split hold out.
(b) The rms miss on validation (each split) and on the test against the
length of the training run, with the spread over the ten splits.

Prints the statistical inefficiency of U at 10 fs and the effective
number of independent frames, the binning's own miss, and the misses at
each length.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import DATA, argon_pair
from matplotlib.ticker import NullFormatter

from mdlab import learn, viz
from mdlab.analysis import stats
from mdlab.cell import minimum_image
from mdlab.viz import ACCENT, OCHRE, OXBLOOD

EDGES = np.linspace(3.0, 8.5, 111)
LENGTHS = np.array([200, 400, 1000, 2000])  # frames of 10 fs
BLOCK = 50
RIDGE = 1e-2
SPLITS = 10


def histograms(run, frames):
    """Each frame's counts of pair distances in the bins of EDGES."""
    h = run["cell"]
    i, j = np.triu_indices(run["positions"].shape[1], 1)
    out = []
    for f in run["positions"][frames]:
        d = np.linalg.norm(minimum_image(f[i] - f[j], h), axis=1)
        out.append(np.histogram(d, EDGES)[0])
    return np.array(out, dtype=float)


def fit_and_miss(X, y, train, tests):
    """Ridge on the scaled features of ``train``; rms misses, meV."""
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-12
    offset = y[train].mean()

    def design(Z):
        return np.c_[np.ones(len(Z)), (Z - mu) / sd]

    w = learn.least_squares(design(X[train]), y[train] - offset, ridge=RIDGE)
    return [1000 * np.sqrt(np.mean((design(Z) @ w + offset - t) ** 2))
            for Z, t in tests]


a = np.load(DATA / "argon_dense_a.npz")
b = np.load(DATA / "argon_dense_b.npz")
kept = np.arange(1000, 3001)
Xa, ya = histograms(a, kept), a["potential"][kept]
test_frames = np.arange(1000, 3001, 5)
Xb, yb = histograms(b, test_frames), b["potential"][test_frames]
centres = 0.5 * (EDGES[1:] + EDGES[:-1])
binned = Xa @ argon_pair(centres)
g = stats.statistical_inefficiency(ya)
print(f"U at 10 fs: statistical inefficiency {g:.1f} frames; the binned pair "
      f"energy misses U by {1000 * np.sqrt(np.mean((binned - ya)**2)):.1f} "
      f"meV against a spread of {1000 * ya.std():.1f} meV")

rows = []
for n in LENGTHS:
    found = {"random": [], "blocked": [], "test": []}
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        for split in ("random", "blocked"):
            train, valid = (learn.random_split(n, 0.2, rng)
                            if split == "random"
                            else learn.blocked_split(n, 0.2, BLOCK, rng))
            v, t = fit_and_miss(Xa[:n], ya[:n], train,
                                [(Xa[:n][valid], ya[:n][valid]), (Xb, yb)])
            found[split].append(v)
            found["test"].append(t)
    rows.append({k: (np.mean(x), np.std(x)) for k, x in found.items()})
    r = rows[-1]
    print(f"{n} frames ({n / 100:.0f} ps, about {n / g:.0f} independent): "
          f"validation {r['random'][0]:.1f} ± {r['random'][1]:.1f} (random),"
          f" {r['blocked'][0]:.1f} ± {r['blocked'][1]:.1f} (blocked); test "
          f"{r['test'][0]:.1f} ± {r['test'][1]:.1f} meV")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                         gridspec_kw=dict(wspace=0.35))
ax = axes[0]
span = np.arange(400)
t = span * 0.01  # ps
ax.plot(t, (ya[span] - ya[span].mean()) * 1000, color="0.6", lw=0.6)
rng = np.random.default_rng(0)
_, valid_r = learn.random_split(400, 0.2, rng)
_, valid_b = learn.blocked_split(400, 0.2, BLOCK, rng)
ax.plot(t[valid_r], (ya[valid_r] - ya[span].mean()) * 1000, "o", ms=1.6,
        color=OXBLOOD, label="held out at random")
for k, start in enumerate(valid_b[::BLOCK]):
    ax.axvspan(t[start], t[min(start + BLOCK - 1, 399)], color=ACCENT,
               alpha=0.15, lw=0, label="held out in blocks" if k == 0
               else None)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$U - \langle U\rangle$ / meV")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
ps = LENGTHS / 100
for key, colour, mark, label in (
        ("random", OXBLOOD, "o-", "validation, random"),
        ("blocked", ACCENT, "s-", "validation, blocked"),
        ("test", OCHRE, "^-", "an independent run")):
    mean = np.array([r[key][0] for r in rows])
    sd = np.array([r[key][1] for r in rows])
    ax.errorbar(ps, mean, sd, fmt=mark, ms=3, lw=0.8, color=colour,
                capsize=0, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(ps, [f"{x:g}" for x in ps])
ax.set_yticks([20, 50, 100], ["20", "50", "100"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("length of the training run / ps")
ax.set_ylabel("rms miss of $U$ / meV")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "leak.pdf")))
