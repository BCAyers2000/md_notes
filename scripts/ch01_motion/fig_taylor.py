"""Figure fig:mo-taylor: Taylor polynomials of sin t about t1 = pi/4.

(a) The function and its Taylor polynomials of degrees 1, 2 and 3, which
agree with it ever more closely near t1. (b) The error of the degree-n
polynomial at t1 + tau against tau, falling as tau^(n+1).

Prints the numbers that Section 1.8 (sec:mo-taylor) quotes.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import (
    ACCENT,
    GREEN,
    OCHRE,
    OXBLOOD,
    REFERENCE_STYLE,
    figure_path,
)

T1 = np.pi / 4
# derivatives of sin at t1, in order: sin, cos, -sin, -cos, sin
DERIVATIVES = [np.sin(T1), np.cos(T1), -np.sin(T1), -np.cos(T1), np.sin(T1)]
DEGREES = (1, 2, 3)
COLOURS = (OCHRE, OXBLOOD, GREEN)


def taylor(tau, degree):
    """Taylor polynomial of sin about t1, evaluated at t1 + tau."""
    return sum(
        DERIVATIVES[k] * tau**k / math.factorial(k) for k in range(degree + 1)
    )


for degree in (0, *DEGREES):
    errors = [
        abs(np.sin(T1 + tau) - taylor(tau, degree)) for tau in (0.1, 0.05)
    ]
    print(
        f"degree {degree}: error {errors[0]:.3e} at tau = 0.1, "
        f"{errors[1]:.3e} at tau = 0.05; halving tau divides it by "
        f"{errors[0] / errors[1]:.2f} (2^{degree + 1} = {2 ** (degree + 1)})"
    )

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1,
    2,
    figsize=(viz.FULL, 2.6),
    gridspec_kw=dict(width_ratios=(1.35, 1), wspace=0.3),
)
t = np.linspace(-np.pi, 2 * np.pi, 400)
left.axhline(0.0, color="black", lw=0.4)
left.plot(t, np.sin(t), color="black", lw=1.6)
for degree, colour in zip(DEGREES, COLOURS, strict=True):
    left.plot(t, taylor(t - T1, degree), color=colour, lw=1.0)
left.plot(T1, np.sin(T1), "o", color="black", ms=3.5, zorder=4)
left.annotate(
    r"$t_1 = \pi/4$",
    (T1, np.sin(T1)),
    xytext=(-6, 8),
    textcoords="offset points",
    ha="right",
    fontsize=9,
)
viz.pi_ticks(left, -2, 4)
left.set_xlim(-np.pi, 2 * np.pi)
left.set_ylim(-2.0, 2.0)
left.set_xlabel(r"$t$ / rad")
left.set_ylabel(r"$x(t)$")
# the degrees share their colours with the key of panel (b)
left.text(4.15, -0.55, r"$\sin t$", fontsize=9)
viz.panel_tag(left, "a")

tau = np.geomspace(1e-3, 1.0, 40)
for degree, colour in zip((0, *DEGREES), (ACCENT, *COLOURS), strict=True):
    err = np.abs(np.sin(T1 + tau) - taylor(tau, degree))
    sparse = slice(None, None, 3)
    right.loglog(
        tau[sparse],
        err[sparse],
        "o",
        mfc="white",
        mec=colour,
        mew=0.9,
        ms=3.2,
        label=f"degree {degree}",
        zorder=2,
    )
    # the first term the polynomial leaves out
    n = degree + 1
    lead = abs(DERIVATIVES[n]) * tau**n / math.factorial(n)
    right.loglog(
        tau,
        lead,
        lw=0.8,
        zorder=3,
        label="leading term" if degree == 0 else None,
        **REFERENCE_STYLE,
    )
right.set_xlabel(r"interval $\tau$")
right.set_ylabel(r"error at $t_1 + \tau$")
right.set_xlim(1e-3, 1.0)
right.set_ylim(1e-14, 2.0)
# list the degrees first and the leading term last
handles, labels = right.get_legend_handles_labels()
first = labels.index("leading term")
listing = [i for i in range(len(labels)) if i != first] + [first]
right.legend(
    [handles[i] for i in listing],
    [labels[i] for i in listing],
    loc="lower right",
    handlelength=1.4,
    fontsize=8,
)
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch01_motion", "taylor.pdf")))
