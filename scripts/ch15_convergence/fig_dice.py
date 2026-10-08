"""Figure fig:cv-dice: the error of an average of independent samples.

(a) The mean of n rolls of a fair die, 200 000 times over, for n = 4 and
64, against the Gaussian of mean 3.5 and variance σ²/n (dashed). (b) The
spread of the mean against n, against σ/√n (dashed). (c) How often the
interval x̄ ± 1.96 s/√n, with s the spread of the n rolls themselves,
holds 3.5, against n (teal), and the interval with Student's factor in
place of 1.96 (ochre), against 0.95 (dotted).

Prints the numbers of Section 15.1.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, THRESHOLD_STYLE

rng = np.random.default_rng(0)
TRIALS = 200_000
SIGMA = math.sqrt(35 / 12)
print(f"one roll: mean 3.5, variance 35/12 = {35 / 12:.4f}, σ = {SIGMA:.4f}")

sizes = 2 ** np.arange(0, 9)
spreads, cover_z, cover_t = [], [], []
means = {}
for n in sizes:
    rolls = rng.integers(1, 7, (TRIALS, n))
    m = rolls.mean(axis=1)
    means[n] = m
    spreads.append(m.std())
    if n > 1:
        s = rolls.std(axis=1, ddof=1) / math.sqrt(n)
        t_factor = st.t.ppf(0.975, n - 1)
        cover_z.append(np.mean(np.abs(m - 3.5) <= 1.96 * s))
        cover_t.append(np.mean(np.abs(m - 3.5) <= t_factor * s))
        print(f"n = {n:3d}: spread of the mean {m.std():.4f} (σ/√n "
              f"{SIGMA / math.sqrt(n):.4f}); ±1.96 s/√n holds 3.5 in "
              f"{cover_z[-1]:.3f}, ±{t_factor:.3f} s/√n in {cover_t[-1]:.3f}")
three = rng.integers(1, 7, (TRIALS, 3))
s3 = three.std(axis=1, ddof=1) / math.sqrt(3)
m3 = three.mean(axis=1)
print(f"n = 3: ±1.96 s/√n holds 3.5 in "
      f"{np.mean(np.abs(m3 - 3.5) <= 1.96 * s3):.3f}; Student's factor for "
      f"two degrees of freedom {st.t.ppf(0.975, 2):.3f} holds it in "
      f"{np.mean(np.abs(m3 - 3.5) <= st.t.ppf(0.975, 2) * s3):.3f}")
gauss = rng.standard_normal((TRIALS, 3))
sg = gauss.std(axis=1, ddof=1) / math.sqrt(3)
print(f"three Gaussian samples: ±1.96 s/√n holds the mean in "
      f"{np.mean(np.abs(gauss.mean(axis=1)) <= 1.96 * sg):.3f} (exact "
      f"{2 * st.t.cdf(1.96, 2) - 1:.3f}); ±4.303 s/√n in "
      f"{np.mean(np.abs(gauss.mean(axis=1)) <= st.t.ppf(0.975, 2) * sg):.3f}")
print(f"Student's 95% factor for four degrees of freedom (a difference of "
      f"two means of three): {st.t.ppf(0.975, 4):.3f}")
for k, p in ((1, 0.6827), (2, 0.9545)):
    print(f"a Gaussian within {k} standard deviation(s): "
          f"{math.erf(k / math.sqrt(2)):.4f} ({p})")
print(f"factor for 95%: {st.norm.ppf(0.975):.3f}")

viz.use_style(notebook=False)
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(viz.FULL, 2.0), gridspec_kw=dict(wspace=0.5))
grid = np.linspace(1, 6, 400)
for n, strength in ((4, 0.5), (64, 1.0)):
    edges = (np.arange(n * 1, n * 6 + 2) - 0.5) / n
    ax_a.hist(means[n], bins=edges, density=True, histtype="step",
              color=viz.tint(ACCENT, strength), lw=1.0, label=f"$n = {n}$")
    sd = SIGMA / math.sqrt(n)
    ax_a.plot(grid, np.exp(-0.5 * ((grid - 3.5) / sd) ** 2)
              / (sd * math.sqrt(2 * math.pi)), **REFERENCE_STYLE, lw=0.8)
ax_a.set_xlim(1, 6)
ax_a.set_xlabel(r"mean of $n$ rolls")
ax_a.set_ylabel("density")
ax_a.legend(fontsize=7, loc="upper right", handlelength=1.0,
            handletextpad=0.4)
viz.panel_tag(ax_a, "a")

ax_b.loglog(sizes, spreads, "o", color=ACCENT, ms=3)
ax_b.loglog(sizes, SIGMA / np.sqrt(sizes), **REFERENCE_STYLE, lw=0.8)
ax_b.set_xlabel(r"$n$")
ax_b.set_ylabel("spread of the mean")
viz.panel_tag(ax_b, "b")

ax_c.semilogx(sizes[1:], cover_z, "o-", color=ACCENT, ms=3, lw=0.8,
              label=r"$1.96$")
ax_c.semilogx(sizes[1:], cover_t, "s-", color=OCHRE, ms=3, lw=0.8,
              label="Student")
ax_c.axhline(0.95, **THRESHOLD_STYLE)
ax_c.set_ylim(0.6, 1.0)
ax_c.set_xlabel(r"$n$")
ax_c.set_ylabel("fraction holding 3.5")
ax_c.legend(fontsize=7, loc="lower right")
viz.panel_tag(ax_c, "c")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence", "dice.pdf")))
