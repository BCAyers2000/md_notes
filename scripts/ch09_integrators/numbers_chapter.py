"""Numbers quoted in Chapter 9 that no figure script prints.

Run from theory/: python scripts/ch09_integrators/numbers_chapter.py
"""

import math

import numpy as np

from mdlab import integrators


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


heading("9.3 reversal: the run of the mdlab test")
rng = np.random.default_rng(1)
r0, v0 = rng.normal(size=(3, 2)), rng.normal(size=(3, 2))


def quartic(r):
    return -(r**3)


r, v, f = r0, v0, None
for sign in (1, -1):
    v, f = sign * v, None
    for _ in range(200):
        r, v, f = integrators.velocity_verlet_step(
            r, v, [1, 2, 3], quartic, 0.05, f, 1.0
        )
print(
    f"200 steps forwards and back: positions within "
    f"{np.abs(r - r0).max():.1e}, velocities within "
    f"{np.abs(-v - v0).max():.1e}"
)

heading("9.5 and checks: the shadow energy")
for x in (0.1, 0.2, 0.5):
    w = math.acos(1 - x**2 / 2) / x
    print(
        f"omega dt = {x}: band {x**2 / 4:.4f}, computed frequency "
        f"x {w:.5f} ({100 * (w - 1):.3f}% fast)"
    )

for x in (0.2, 0.1):
    print(
        f"symplectic Euler at omega dt = {x}: band {1 / (1 + x / 2):.4f} "
        f"to {1 / (1 - x / 2):.4f}"
    )

heading("9.6 the step for water and lithium")
w_oh = 2 * math.pi / 8.88
print(
    f"O-H: omega = {w_oh:.4f} rad/fs, 2/omega = {2 / w_oh:.3f} fs, "
    f"1/omega = {1 / w_oh:.3f} fs"
)
for dt in (0.5, 1.0):
    print(
        f"  dt = {dt} fs: omega dt = {w_oh * dt:.3f}, band "
        f"{100 * (w_oh * dt) ** 2 / 4:.1f}%"
    )
print(f"  ex: 3% band: dt = {math.sqrt(0.12) / w_oh:.3f} fs")
w_li = 2 * math.pi / 170.3
print(
    f"lithium hollow: omega = {w_li:.4f} rad/fs, limit {2 / w_li:.1f} fs,"
    f" 1% band at {0.2 / w_li:.2f} fs"
)
w_model = math.sqrt(0.7338)
print(
    f"model water: omega = {w_model:.4f}, limit {2 / w_model:.3f} fs; at "
    f"2.3 fs 1 - (omega dt)^2/4 = {1 - (w_model * 2.3) ** 2 / 4:.3f}, "
    f"swing x {1 / math.sqrt(1 - (w_model * 2.3) ** 2 / 4):.1f}"
)
print(f"ex: steps per period for a 1% band: {2 * math.pi / 0.2:.1f}")

heading("exercises")
t = 2 - 2.05**2
root = math.sqrt(t * t / 4 - 1)
print(
    f"growth at 2.05: trace {t:.4f}, roots {t / 2 + root:.4f} and "
    f"{t / 2 - root:.4f}"
)
x = 0.2
factor = 1 - x**6 / 72 + x**8 / 576
print(
    f"RK4 spring at x = 0.2: x^6/72 = {x**6 / 72:.3e}, x^8/576 = "
    f"{x**8 / 576:.2e}, factor 1 - {1 - factor:.3e}; after 10000 steps "
    f"{factor**10000:.4f}, lost {100 * (1 - factor**10000):.2f}%"
)
print(f"learned 5 fs step: omega dt = {w_oh * 5:.3f}")
x = w_oh * 0.7
ratio = math.acos(1 - x**2 / 2) / x
print(
    f"dt = 0.7 fs: omega dt = {x:.4f}, cos = {1 - x**2 / 2:.4f}, "
    f"omega~ dt = {x * ratio:.4f}, ratio {ratio:.4f}, 3756 -> "
    f"{3756 * ratio:.0f} cm-1"
)
print(
    f"check: 1 ps at 0.5 fs = {1000 / 0.5:.0f} forces; RK4 at 2 fs = "
    f"{4 * 1000 / 2:.0f}"
)
