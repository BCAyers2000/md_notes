"""Figure fig:cs-shake: how SHAKE converges.

(a) One bond of length 1 between equal masses, started stretched by 1%,
10% and 30% along a direction 18° off the old bond: the relative error of
its length after each sweep, on a logarithmic axis. Each sweep is a
Newton step and roughly squares the error. (b) The 64 rigid molecules of
the prepared water box after one unconstrained step of 2 fs: the sweeps
SHAKE needs against the tolerance, and the largest bond error left. The
three bonds of each molecule share atoms, so correcting one disturbs the
others and the error falls by a steady factor per sweep.

Prints the numbers of Section 11.1.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch11 import N_MOLECULES, load, model

from mdlab import constraints, units, viz, water
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, figure_path

# (a) one bond, sweep by sweep
old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
angle = math.radians(18.0)
histories = {}
for stretch in (0.01, 0.1, 0.3):
    length = 1 + stretch
    r = np.array([[0.0, 0.0, 0.0],
                  [length * math.cos(angle), length * math.sin(angle), 0.0]])
    errors = [stretch]
    for _ in range(7):
        r, _ = constraints.shake(r, old, [1.0, 1.0], [[0, 1]], [1.0],
                                 tolerance=0.0, max_sweeps=1, strict=False)
        errors.append(abs(np.linalg.norm(r[1] - r[0]) - 1.0))
    histories[stretch] = np.array(errors)
    shown = ", ".join(f"{e:.2e}" for e in errors[:5])
    print(f"stretch {stretch:.0%}: errors after 0-4 sweeps {shown}")
    ratios = [errors[k + 1] / errors[k] ** 2 for k in range(3)]
    print("   error / (previous error)^2:",
          ", ".join(f"{x:.3f}" for x in ratios))
    bond = r[1] - r[0]
    turn = math.degrees(math.acos(bond[0] / np.linalg.norm(bond)))
    print(f"   the corrected bond lies {turn:.1f} degrees off the old one; "
          f"1/(2 cos^2) = {1 / (2 * math.cos(math.radians(turn)) ** 2):.3f}")

# (b) the water box after one unconstrained step
start = load()
h, m = start["cell"], start["masses"]
r0, v0 = start["positions"], start["velocities"]
bonds, lengths = water.constraint_bonds(N_MOLECULES)
_, f0 = model(h)(r0)
dt = 2.0
drifted = r0 + dt * (v0 + 0.5 * dt * units.FORCE_TO_ACCEL * f0 / m[:, None])
before = np.abs(constraints.bond_errors(drifted, bonds, lengths, h)).max()
print(f"water box, one step of {dt} fs unconstrained: largest bond error "
      f"{before:.3e} Å ({before / water.R_OH:.2%} of O-H)")
tolerances = 10.0 ** -np.arange(2, 15)
sweeps, left = [], []
for tol in tolerances:
    r, n = constraints.shake(drifted, r0, m, bonds, lengths, h, tol)
    sweeps.append(n)
    left.append(np.abs(constraints.bond_errors(r, bonds, lengths, h)).max())
    print(f"tolerance {tol:.0e}: {n} sweeps, largest error {left[-1]:.2e} Å")
slope = np.polyfit(np.log10(tolerances[2:]), np.array(sweeps[2:]), 1)[0]
print(f"sweeps per factor of 10 in the tolerance: {-slope:.2f}, so each "
      f"sweep cuts the error by about {10 ** (-1 / slope):.2f}")

viz.use_style(notebook=False)
fig, (left_ax, right_ax) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.45)
)
for stretch, colour in ((0.01, ACCENT), (0.1, OCHRE), (0.3, OXBLOOD)):
    e = histories[stretch]
    end = int(np.argmax(e < 1e-15))  # the first sweep at round-off
    e = e[:end] if end else e
    left_ax.semilogy(np.arange(len(e)), e, "o-", color=colour,
                     ms=3, lw=0.9, label=f"{stretch:.0%}")
left_ax.set_xlabel("sweeps")
left_ax.set_ylabel("relative error of the length")
left_ax.set_ylim(1e-15, 1)
left_ax.set_xticks(range(5))
left_ax.set_yticks(10.0 ** np.arange(-15, 1, 3))
left_ax.legend(title="stretch", fontsize=8, title_fontsize=8)
viz.panel_tag(left_ax, "a")

right_ax.semilogx(tolerances, sweeps, "o-", color=ACCENT, ms=3, lw=0.9)
right_ax.set_xlabel("tolerance")
right_ax.set_ylabel("sweeps", color=ACCENT)
right_ax.invert_xaxis()
twin = right_ax.twinx()
twin.loglog(tolerances, left, "s", color=OCHRE, ms=3)
twin.set_ylabel("largest bond error / Å", color=OCHRE)
twin.spines["right"].set_visible(True)
viz.panel_tag(right_ax, "b")

print("wrote", viz.save(fig, figure_path("ch11_constraints", "shake.pdf")))
