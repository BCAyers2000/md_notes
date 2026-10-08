"""Figure fig:ha-maps: a step map that keeps area, and one that does not.

The block of Chapter 4, 0.5 kg on a spring of 50 N/m (ω = 10 rad/s),
released from rest at 4 cm, stepped with δt = 0.02 s (ωδt = 0.2) by two
step maps of Hamilton's equations: the forward Euler method, which
multiplies the energy and every area by 1 + ω²δt² at each step, and the
symplectic Euler method, which keeps area. (a) The states after each
step for 1 s, with the true ellipse (dashed). (b) The energy over 10 s,
as a multiple of its starting value, on a logarithmic axis.

Prints the numbers of Section 7.7.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import hamiltonian, viz
from mdlab.viz import METHOD, REFERENCE_STYLE, figure_path

MASS, K, Q0, DT = 0.5, 50.0, 0.04, 0.02
OMEGA = math.sqrt(K / MASS)


def dh_dq(q, p):
    return K * q


def dh_dp(q, p):
    return p / MASS


def energy(q, p):
    return p**2 / (2 * MASS) + 0.5 * K * q**2


def march(step, n):
    q, p = np.empty(n + 1), np.empty(n + 1)
    q[0], p[0] = Q0, 0.0
    for i in range(n):
        q[i + 1], p[i + 1] = step(dh_dq, dh_dp, q[i], p[i], DT)
    return q, p


factor = 1 + (OMEGA * DT) ** 2
print(f"omega dt = {OMEGA * DT:.2f}; Euler factor per step {factor:.4f}")
print(
    f"after 50 steps (1 s) {factor**50:.4f}; after one period "
    f"({2 * math.pi / OMEGA:.4f} s) {factor ** (2 * math.pi / OMEGA / DT):.4f}"
)
print(
    "Euler energy reaches 1000 x after "
    f"{math.log(1000) / math.log(factor) * DT:.2f} s"
)
print(
    f"steps to double the energy: ln2/ln(1 + w²dt²) = "
    f"{math.log(2) / math.log(factor):.2f}"
)
for wdt in (0.1, 0.01):
    print(f"  at omega dt = {wdt}: {math.log(2) / math.log(1 + wdt**2):.2f}")
E0 = energy(Q0, 0.0)
qe, pe = march(hamiltonian.euler_step, 500)
qs, ps = march(hamiltonian.symplectic_euler_step, 500)
ratio_e, ratio_s = energy(qe, pe) / E0, energy(qs, ps) / E0
print(
    f"E0 = {E0:.4f} J; Euler after 1 s {ratio_e[50]:.4f}, after 10 s "
    f"{ratio_e[500]:.3e}"
)
print(
    f"symplectic Euler over 10 s: E/E0 between {ratio_s.min():.4f} and "
    f"{ratio_s.max():.4f}"
)
long_q, long_p = march(hamiltonian.symplectic_euler_step, 100000)
long = energy(long_q, long_p) / E0
print(f"  over 2000 s: between {long.min():.4f} and {long.max():.4f}")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.38)
)
s = np.linspace(0, 2 * np.pi, 300)
left.plot(
    100 * Q0 * np.cos(s), MASS * OMEGA * Q0 * np.sin(s), **REFERENCE_STYLE
)
left.plot(
    100 * qe[:51],
    pe[:51],
    "o-",
    color=METHOD["forward Euler"],
    ms=2.2,
    lw=0.6,
    label="forward Euler",
)
left.plot(
    100 * qs[:51],
    ps[:51],
    "o-",
    color=METHOD["symplectic Euler"],
    ms=2.2,
    lw=0.6,
    label="symplectic Euler",
)
left.set_xlim(-12, 12)
left.set_ylim(-0.62, 0.62)
left.set_xlabel(r"position $q$ / cm")
left.set_ylabel(r"momentum $p$ / kg m s$^{-1}$")
viz.panel_tag(left, "a")

t = DT * np.arange(501)
right.semilogy(
    t, ratio_e, color=METHOD["forward Euler"], label="forward Euler"
)
right.semilogy(
    t, ratio_s, color=METHOD["symplectic Euler"], label="symplectic Euler"
)
right.legend(loc="center right", fontsize=8, handlelength=1.5)
right.axhline(1.0, **REFERENCE_STYLE)
right.set_xlim(0, 10)
right.set_ylim(0.5, 1e3)
exponent = math.floor(math.log10(ratio_e[-1]))
mantissa = ratio_e[-1] / 10**exponent
right.annotate(rf"continues to ${mantissa:.1f}\times10^{{{exponent}}}$ at 10 s",
               xy=(math.log(1000) / math.log(factor) * DT, 1000),
               xytext=(0.98, 0.97), textcoords="axes fraction",
               ha="right", va="top", fontsize=7.5,
               arrowprops=dict(arrowstyle="->", color=METHOD["forward Euler"],
                               linewidth=0.7))
right.set_xlabel(r"time $t$ / s")
right.set_ylabel(r"energy / starting energy")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch07_hamilton", "maps.pdf")))
