"""Figure fig:cs-resonance: where a multiple-time-step method resonates.

(a) One coordinate pulled by a fast spring (ω_f) and a slow one
(ω_s = 0.3 ω_f), stepped by RESPA with the fast motion followed by 50
inner steps: half the trace of the outer-step matrix, from
2 cos θ − (ω_s² δt/ω_f) sin θ, against δt in units of the fast period
𝒯_f; beyond ±1 (dotted) the motion grows. (b) The same model run for 300
outer steps from q = 1 at rest: the largest size of the state,
sqrt(q² + (v/ω_f)²), reached, on a logarithmic axis.

Prints the numbers of Section 11.5.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from mdlab import respa, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, THRESHOLD_STYLE, figure_path

OMEGA_F, RATIO = 1.0, 0.3
OMEGA_S = RATIO * OMEGA_F
PERIOD = 2 * math.pi / OMEGA_F


def half_trace(dt, n_inner=None):
    return np.trace(respa.outer_step_matrix(dt, OMEGA_F, OMEGA_S,
                                            n_inner)) / 2


def fast(q):
    return 0.5 * OMEGA_F**2 * float(np.sum(q * q)), -OMEGA_F**2 * q


def slow(q):
    return 0.5 * OMEGA_S**2 * float(np.sum(q * q)), -OMEGA_S**2 * q


def largest(dt, steps=300):
    """The largest size sqrt(q² + (v/ω_f)²) of the state in ``steps``."""
    out = respa.run(fast, slow, [1.0], [[1.0]], [[0.0]], dt, 50, steps,
                    force_to_accel=1)
    q, v = out["positions"][:, 0, 0], out["velocities"][:, 0, 0]
    return float(np.sqrt(q * q + (v / OMEGA_F) ** 2).max())


x = np.linspace(0.02, 1.15, 1131)
exact = np.array([half_trace(t * PERIOD) for t in x])
inner = np.array([half_trace(t * PERIOD, 50) for t in x])
print(f"largest difference of the half trace, exact against 50 inner "
      f"steps: {np.abs(exact - inner).max():.2e}")


def edge(lo, hi, level):
    """Where half the trace crosses ``level`` between lo and hi."""
    return brentq(lambda t: half_trace(t * PERIOD) - level, lo, hi)


# The bands where |half trace| > 1: below 1/2 it falls below -1 and comes
# back to -1 at 1/2 exactly; below 1 it rises above 1 and comes back at 1.
BANDS = [(edge(0.40, 0.475, -1.0), 0.5), (edge(0.85, 0.95, 1.0), 1.0)]
for lo, hi in BANDS:
    print(f"unstable for dt/T_f from {lo:.4f} to {hi:.4f}, width "
          f"{hi - lo:.4f}")
print(f"at dt/T_f = 0.5 exactly, half the trace is "
      f"{half_trace(0.5 * PERIOD):+.6f}")
print(f"predicted width of the band below 1/2 for small (w_s/w_f)^2: "
      f"(w_s/w_f)^2 / 2 = {RATIO**2 / 2:.4f} of T_f; keeping the slow term "
      f"at theta = pi - eps: b / (2 (1 + b)) = "
      f"{RATIO**2 / (2 * (1 + RATIO**2)):.4f}")
at_half = respa.outer_step_matrix(0.5 * PERIOD, OMEGA_F, OMEGA_S)
print(f"at dt = T_f/2 the outer step has rows {at_half.round(6).tolist()}; "
      f"v gains {at_half[1, 0]:.4f} q per step; 2 c = "
      f"{0.5 * PERIOD * OMEGA_S**2:.4f}")
print(f"lower edges with the slow term kept at pi - eps: 1/(2 (1 + b)) = "
      f"{0.5 / (1 + RATIO**2):.4f}, below the period 1/(1 + b) = "
      f"{1 / (1 + RATIO**2):.4f}")
print(f"centre of the first band {(BANDS[0][0] + 0.5) / 2:.4f}; half the "
      f"period of both springs together 0.5 / sqrt(1 + b) = "
      f"{0.5 / math.sqrt(1 + RATIO**2):.4f}")
grid = np.linspace(0.05, 1.15, 221)
peaks = np.array([largest(t * PERIOD) for t in grid])
for t in (0.30, 0.45, 0.48, 0.49, 0.5, 0.52, 0.9, 0.97, 1.03):
    print(f"dt/T_f = {t:.2f}: half trace {half_trace(t * PERIOD):+.4f}, "
          f"largest size of the state in 300 steps "
          f"{largest(t * PERIOD):.3e}")

viz.use_style(notebook=False)
fig, (top, bottom) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.4)
)
top.plot(x, exact, color=ACCENT)
for level in (1, -1):
    top.axhline(level, **THRESHOLD_STYLE)
for lo, hi in BANDS:
    top.axvspan(lo, hi, color=ACCENT, alpha=0.15, lw=0)
top.plot(x, np.cos(OMEGA_F * x * PERIOD), **REFERENCE_STYLE, lw=0.9)
top.set_xlabel(r"outer step $\delta t/\mathcal{T}_\mathrm{f}$")
top.set_ylabel("half the trace")
top.set_xlim(0, 1.15)
viz.panel_tag(top, "a")
bottom.semilogy(grid, peaks, color=ACCENT)
for lo, hi in BANDS:
    bottom.axvspan(lo, hi, color=ACCENT, alpha=0.15, lw=0)
bottom.set_xlabel(r"outer step $\delta t/\mathcal{T}_\mathrm{f}$")
bottom.set_ylabel("largest size in 300 steps")
bottom.set_xlim(0, 1.15)
viz.panel_tag(bottom, "b")

print("wrote", viz.save(fig, figure_path("ch11_constraints",
                                         "resonance.pdf")))
