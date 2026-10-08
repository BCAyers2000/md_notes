"""Figure fig:lo-bond: one parameter learned by least squares.

(a) A spring stretched by 1 to 10 cm and its force read on a dial with an
error of 0.05 N (readings drawn by computer, the true stiffness 25 N/m),
with the line through the origin of least squares. (b) The sum of squared
misses against the stiffness tried, a parabola whose lowest point is the
fit. (c) Chapter 16's A-B molecule under Langevin dynamics at 100 and
1000 K (ch18.bond_run, 20 ps, a reading every 10 fs; every fifth shown):
the force along the bond against its stretch, the Morse force (grey) and
the straight lines of least squares.

Prints the spring's fitted stiffness with its error; the bond's spread of
stretches against the harmonic √(k_BT/k); the stiffness of the straight
line at 100, 300 and 1000 K, and the slope at the bottom of polynomials
of degree 2 to 4 with their rms misses, against 2Da² = 8 eV/Å².
"""

import matplotlib.pyplot as plt
import numpy as np
from ch18 import DEPTH, R0, STIFFNESS, WIDTH, bond_run
from scipy.stats import t as t_dist

from mdlab import learn, potentials, units, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

rng = np.random.default_rng(18)
stretch = np.arange(1, 11) / 100  # m
force = 25.0 * stretch + rng.normal(0, 0.05, stretch.size)  # N
k_spring = np.sum(force * stretch) / np.sum(stretch * stretch)
misses = force - k_spring * stretch
sigma = np.sqrt(np.sum(misses**2) / (stretch.size - 1))
error = sigma / np.sqrt(np.sum(stretch**2))
print(f"spring: k = {k_spring:.2f} ± {error:.2f} N/m from {stretch.size} "
      f"readings, misses of rms {sigma:.3f} N; {(k_spring - 25) / error:.1f}"
      f" errors from the 25 N/m drawn from, a miss that Student's t with "
      f"{stretch.size - 1} degrees of freedom exceeds in "
      f"{2 * t_dist.sf(abs(k_spring - 25) / error, stretch.size - 1):.3f} "
      "of fits")

fits = {}
for temperature in (100, 300, 1000):
    x, f = bond_run(temperature, 20.0, 0)
    fits[temperature] = (x, f)
    rms = x.std()
    harmonic = np.sqrt(units.KB * temperature / STIFFNESS)
    line = learn.least_squares(learn.polynomial_basis(x, 1), f)
    print(f"{temperature} K: stretches spread by {rms:.4f} Å (harmonic "
          f"{harmonic:.4f}), from {x.min():.3f} to {x.max():.3f} Å; "
          f"straight line: stiffness {-line[1]:.3f} eV/Å²")
    for degree in (2, 3, 4):
        X = learn.polynomial_basis(x, degree)
        w = learn.least_squares(X, f)
        miss = np.sqrt(np.mean((X @ w - f) ** 2))
        print(f"   degree {degree}: slope at the bottom {-w[1]:.3f} eV/Å², "
              f"rms miss {miss:.5f} eV/Å")
print(f"2Da² = {STIFFNESS:.1f} eV/Å²")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
ax.plot(stretch * 100, force, "o", ms=3, color=ACCENT)
grid = np.linspace(0, 0.105, 50)
ax.plot(grid * 100, k_spring * grid, color=OCHRE, lw=1.0)
ax.set_xlabel("stretch / cm")
ax.set_ylabel("force / N")
viz.panel_tag(ax, "a")

ax = axes[1]
trial = np.linspace(22, 28, 100)
loss = [np.sum((force - k * stretch) ** 2) for k in trial]
ax.plot(trial, loss, color=ACCENT, lw=1.0)
ax.axvline(k_spring, **REFERENCE_STYLE)
ax.set_xlabel(r"$k$ tried / N\,m$^{-1}$")
ax.set_ylabel(r"loss / N$^2$")
viz.panel_tag(ax, "b")

ax = axes[2]
grid = np.linspace(-0.25, 0.75, 200)
ax.plot(grid, -potentials.morse(grid + R0, DEPTH, WIDTH, R0)[1],
        **REFERENCE_STYLE, lw=0.8)
for temperature, colour in ((1000, OXBLOOD), (100, ACCENT)):
    x, f = fits[temperature]
    ax.plot(x[::5], f[::5], ".", ms=1.5, color=colour, alpha=0.5)
    w = learn.least_squares(learn.polynomial_basis(x, 1), f)
    span = np.linspace(x.min(), x.max(), 2)
    ax.plot(span, w[0] + w[1] * span, color=colour, lw=1.0,
            label=f"{temperature} K")
ax.set_xlim(-0.25, 0.75)
ax.set_ylim(-1.5, 2.0)
ax.set_xlabel(r"stretch $x$ / \AA")
ax.set_ylabel(r"bond force / eV\,\AA$^{-1}$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch18_learning", "bond.pdf")))
