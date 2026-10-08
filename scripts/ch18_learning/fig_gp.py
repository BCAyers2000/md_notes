"""Figure fig:lo-gp: a Gaussian process through a few readings.

Eight readings of argon's pair energy between 3.3 and 6.8 Å, with noise
of 0.2 meV (ch18.readings, seed 184), and the noise-free test. A Gaussian
process with the squared-exponential kernel, height 10 meV (about the
depth of the well) and the noise known, conditioned on them. (a) The
posterior mean and twice its standard deviation for the length scale of
greatest marginal likelihood. (b) The same for a length a quarter of
that and four times it. (c) The log marginal likelihood of the readings
against the length.

Prints the length of greatest marginal likelihood, the test miss and the
fraction of test points within twice the standard deviation for each of
the three lengths, and the standard deviation at the readings and midway
between the two farthest apart.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import NOISE, R_HIGH, R_LOW, argon_pair, readings

from mdlab import learn, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

HEIGHT = 0.010  # eV
LENGTHS = np.exp(np.linspace(np.log(0.05), np.log(10.0), 200))  # Å

rng = np.random.default_rng(184)
r, e = readings(8, rng)
grid = np.linspace(R_LOW - 0.3, R_HIGH + 0.3, 300)
test = np.linspace(R_LOW, R_HIGH, 200)
truth = argon_pair(test)


def process(length):
    return learn.GaussianProcess(length, HEIGHT, NOISE).fit(r, e)


evidence = np.array([process(ell).log_marginal_likelihood()
                     for ell in LENGTHS])
best = float(LENGTHS[np.argmax(evidence)])
print("readings at " + ", ".join(f"{x:.2f}" for x in r) + " Å")
print(f"greatest marginal likelihood at the length {best:.3f} Å")
for factor in (0.25, 1.0, 4.0):
    gp = process(best * factor)
    mean, var = gp.predict(test)
    sd = np.sqrt(var)
    inside = np.mean(np.abs(mean - truth) <= 2 * sd)
    print(f"length {best * factor:.3f} Å: test miss "
          f"{np.sqrt(np.mean((mean - truth) ** 2)) * 1000:.3f} meV, "
          f"{inside:.2f} of the test within two standard deviations")
gp = process(best)
gap = np.argmax(np.diff(r))
middle = 0.5 * (r[gap] + r[gap + 1])
print(f"standard deviation at the readings "
      f"{np.sqrt(gp.predict(r)[1]).max() * 1000:.3f} meV at most; midway "
      f"in the widest gap, {r[gap]:.2f} to {r[gap + 1]:.2f} Å, "
      f"{np.sqrt(gp.predict([middle])[1][0]) * 1000:.3f} meV")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.45))
for ax, factors, tag in ((axes[0], (1.0,), "a"), (axes[1], (0.25, 4.0), "b")):
    ax.plot(test, truth * 1000, **REFERENCE_STYLE, lw=0.8)
    for factor, colour in zip(factors, (ACCENT if factors == (1.0,)
                                        else OCHRE, OXBLOOD), strict=False):
        mean, var = process(best * factor).predict(grid)
        sd = np.sqrt(var)
        ax.fill_between(grid, (mean - 2 * sd) * 1000, (mean + 2 * sd) * 1000,
                        color=colour, alpha=0.25, lw=0)
        ax.plot(grid, mean * 1000, color=colour, lw=0.9,
                label=rf"$\ell = {best * factor:.2f}$\,\AA")
    ax.plot(r, e * 1000, "o", ms=3, color="k", zorder=5)
    ax.set_ylim(-16, 10)
    ax.set_xlabel(r"$r$ / \AA")
    ax.set_ylabel(r"$\varphi(r)$ / meV")
    ax.legend(fontsize=7, loc="upper right")
    viz.panel_tag(ax, tag)

ax = axes[2]
ax.semilogx(LENGTHS, evidence, color=ACCENT, lw=1.0)
ax.axvline(best, **REFERENCE_STYLE, lw=0.8)
ax.set_ylim(evidence.max() - 40, evidence.max() + 3)
ax.set_xlabel(r"length scale $\ell$ / \AA")
ax.set_ylabel("log marginal likelihood")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "gp.pdf")))
