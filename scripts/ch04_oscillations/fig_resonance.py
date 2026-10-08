"""Figure fig:os-resonance: the response to a sinusoidal push.

The block of Section 2.8 (omega0 = 10 rad/s) driven by a force of
amplitude 0.5 N, so f = 1 m/s^2, at angular frequency omega, for three
drag rates. (a) The steady amplitude; the dotted line is the stretch the
same force would give if held still, f/omega0^2 = 1 cm. (b) The phase lag
delta of the motion behind the push.

Prints the numbers that Section 4.7 (sec:os-driven) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from mdlab import oscillators, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

OMEGA0, F = 10.0, 1.0
GAMMAS = ((1.0, ACCENT), (2.0, OCHRE), (4.0, OXBLOOD))
omega = np.linspace(0.0, 20.0, 2001)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.32)
)
for gamma, colour in GAMMAS:
    amp, lag = oscillators.driven_steady_state(omega, OMEGA0, gamma, F)
    left.plot(
        omega,
        100 * amp,
        color=colour,
        label=rf"$\gamma = {gamma:g}$ s$^{{-1}}$",
    )
    right.plot(omega, lag / np.pi, color=colour)
    at_res, _ = oscillators.driven_steady_state(OMEGA0, OMEGA0, gamma, F)
    # the half-power width, from the two roots of A = A_max / √2 either
    # side of the peak at √(ω₀² − γ²/2)
    peak = np.sqrt(OMEGA0**2 - gamma**2 / 2)
    a_max, _ = oscillators.driven_steady_state(peak, OMEGA0, gamma, F)
    level = float(a_max) / np.sqrt(2)

    def excess(w, g=gamma, level=level):
        return (
            float(oscillators.driven_steady_state(w, OMEGA0, g, F)[0]) - level
        )

    width = brentq(excess, peak, 2 * OMEGA0) - brentq(excess, 1e-9, peak)

    print(
        f"gamma {gamma}: A(omega0) = {100 * float(at_res):.2f} cm, "
        f"ratio to the static stretch {float(at_res) / (F / OMEGA0**2):.1f};"
        f" width where A > A_max/sqrt2: {width:.4f} rad/s"
    )
left.axhline(100 * F / OMEGA0**2, **THRESHOLD_STYLE)
left.set_xlim(0, 20)
left.set_ylim(0, 10.5)
left.set_xlabel(r"driving frequency $\omega$ / rad\,s$^{-1}$")
left.set_ylabel(r"amplitude $A$ / cm")
right.legend(*left.get_legend_handles_labels(), loc="lower right")
viz.panel_tag(left, "a")
right.axhline(0.5, **THRESHOLD_STYLE)
right.set_xlim(0, 20)
right.set_ylim(0, 1)
right.set_yticks([0, 0.5, 1], ["0", r"$\pi/2$", r"$\pi$"])
right.set_xlabel(r"driving frequency $\omega$ / rad\,s$^{-1}$")
right.set_ylabel(r"phase lag $\delta$")
viz.panel_tag(right, "b")

print(
    "wrote", viz.save(fig, figure_path("ch04_oscillations", "resonance.pdf"))
)
