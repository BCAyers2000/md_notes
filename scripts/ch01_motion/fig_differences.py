"""Figure fig:mo-differences: velocities from sampled positions.

A coordinate oscillating as x(t) = A sin(w t), with A = 0.1 Å and period
2 pi / w = 20 fs, is sampled every tau. (a) Largest error of the forward and
central difference velocities against tau, with their leading terms
A w² tau / 2 and A w³ tau² / 6.
(b) Relative error of the central difference, exactly 1 - sin(w tau)/(w tau),
against its leading term (w tau)²/6.

Prints the numbers that Section 1.9 (sec:mo-atoms) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import kinematics, viz
from mdlab.viz import (
    ACCENT,
    OCHRE,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

AMPLITUDE = 0.1  # Å
PERIOD = 20.0  # fs
OMEGA = 2 * np.pi / PERIOD

steps = np.geomspace(0.01, 5.0, 25)
errors = {"forward": [], "central": []}
for h in steps:
    t = np.arange(0.0, 2 * PERIOD + 2 * h, h)
    x = AMPLITUDE * np.sin(OMEGA * t)
    v = AMPLITUDE * OMEGA * np.cos(OMEGA * t)
    errors["forward"].append(
        np.max(np.abs(kinematics.forward_difference(x, h) - v[:-1]))
    )
    errors["central"].append(
        np.max(np.abs(kinematics.central_difference(x, h) - v[1:-1]))
    )
errors = {k: np.array(e) for k, e in errors.items()}
vmax = AMPLITUDE * OMEGA
print(f"largest speed A w = {vmax:.5f} Å/fs")
for h in (0.1, 1.0, 2.0):
    wh = OMEGA * h
    print(
        f"tau = {h} fs: central relative error = "
        f"{1 - np.sin(wh) / wh:.5f}, (w tau)^2/6 = {wh**2 / 6:.5f}; "
        f"forward relative bound w tau / 2 = {wh / 2:.4f}"
    )

# Two sinusoids that cancel: the criterion bounds the error relative to
# the sum of the component speeds, not relative to v at every instant.
TAU2, T2 = 0.1, 0.01  # s


def two_sines(t):
    return np.sin(t) - 0.5 * np.sin(2 * t)  # m, with t in s


v_true = np.cos(T2) - np.cos(2 * T2)
v_cd = (two_sines(T2 + TAU2) - two_sines(T2 - TAU2)) / (2 * TAU2)
print(
    f"sin t - sin(2t)/2, tau = {TAU2} s: (w_max tau)^2/6 = "
    f"{100 * (2 * TAU2) ** 2 / 6:.2f}%; at t = {T2} s v = {v_true:.2e} m/s, "
    f"central difference {v_cd:.2e} m/s, {v_cd / v_true:.0f} times too large; "
    f"error {abs(v_cd - v_true):.1e} m/s against the bound "
    f"{(2 * TAU2) ** 2 / 6 * (1 + 1):.1e} m/s"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.38)
)
left.loglog(
    steps,
    0.5 * AMPLITUDE * OMEGA**2 * steps,
    lw=0.9,
    label="leading term",
    **REFERENCE_STYLE,
)
left.loglog(
    steps, AMPLITUDE * OMEGA**3 * steps**2 / 6, lw=0.9, **REFERENCE_STYLE
)
left.loglog(
    steps,
    errors["forward"],
    "o",
    color=OCHRE,
    ms=3,
    label=r"forward, $\propto \tau$",
)
left.loglog(
    steps,
    errors["central"],
    "o",
    color=ACCENT,
    ms=3,
    label=r"central, $\propto \tau^2$",
)
left.set_xlabel(r"sampling interval $\tau$ / fs")
left.set_ylabel(r"largest error / \AA\,fs$^{-1}$")
left.legend(loc="lower right", handlelength=1.4)
viz.panel_tag(left, "a")

ratio = np.linspace(0.0, 0.5, 200)
wh = 2 * np.pi * ratio
exact = 1 - np.sinc(wh / np.pi)  # numpy's sinc is sin(pi x)/(pi x)
right.plot(ratio, 100 * exact, color=ACCENT, label="exact")
right.plot(
    ratio,
    100 * wh**2 / 6,
    lw=1.0,
    label=r"$(\omega\tau)^2/6$",
    **REFERENCE_STYLE,
)
right.axvline(1.0 / PERIOD, **THRESHOLD_STYLE)
right.set_xlabel(r"$\tau$ / period")
right.set_ylabel(r"relative error / \%")
right.set_xlim(0, 0.5)
right.set_ylim(0, 100)
right.legend(loc="upper left")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch01_motion", "differences.pdf")))
