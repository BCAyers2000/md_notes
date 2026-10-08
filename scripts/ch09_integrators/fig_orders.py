"""Figure fig:in-orders: how the error of each method shrinks with δt.

The pendulum of Chapter 7 in reduced units, H = p²/2 + 1 − cos q, started
at q = 1 with p = 0.5. (a) The error of the angle after a time of 4, against
the step δt, for forward Euler, velocity Verlet and RK4, with lines of
slope 1, 2 and 4. (b) The error of one step from the same start, against
δt, with lines of slope 2, 3 and 5. The reference motion is
`hamiltonian.flow`, accurate to about 1e-11.

Prints the fitted slopes quoted in Sections 9.1, 9.2 and 9.7.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import hamiltonian, integrators, viz
from mdlab.viz import METHOD, REFERENCE_STYLE, figure_path

Q0, P0, T_END = 1.0, 0.5, 4.0
METHODS = {
    "forward Euler": "euler",
    "velocity Verlet": "velocity_verlet",
    "RK4": "rk4",
}
COLOURS = {name: METHOD[name] for name in METHODS}


def force(r):
    return -np.sin(r)


def exact(t):
    q, _ = hamiltonian.flow(
        lambda q, p: np.sin(q), lambda q, p: p, [Q0], [P0], [0.0, t]
    )
    return q[-1, 0]


def run(method, dt, n):
    r, v, f = np.array([[Q0]]), np.array([[P0]]), None
    for _ in range(n):
        if method == "velocity_verlet":
            r, v, f = integrators.velocity_verlet_step(
                r, v, [1.0], force, dt, f, 1.0
            )
        elif method == "euler":
            r, v = integrators.euler_step(r, v, [1.0], force, dt, 1.0)
        else:
            r, v = integrators.rk4_step(r, v, [1.0], force, dt, 1.0)
    return r.item()


steps = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
q_end = exact(T_END)
global_err, local_err = {}, {}
for name, method in METHODS.items():
    global_err[name] = np.array(
        [abs(run(method, dt, int(round(T_END / dt))) - q_end) for dt in steps]
    )
    local_err[name] = np.array(
        [abs(run(method, dt, 1) - exact(dt)) for dt in steps]
    )
for name in METHODS:
    g = np.polyfit(np.log(steps), np.log(global_err[name]), 1)[0]
    loc = np.polyfit(np.log(steps), np.log(local_err[name]), 1)[0]
    print(
        f"{name}: global slope {g:.2f}, one-step slope {loc:.2f}; "
        f"global error at dt = 0.05: {global_err[name][2]:.2e}"
    )
signed = [run("rk4", dt, int(round(T_END / dt))) - q_end for dt in steps]
print("RK4 signed global errors:", " ".join(f"{e:+.2e}" for e in signed))
pairs = np.log2(global_err["RK4"][:-1] / global_err["RK4"][1:])
print("RK4 slopes between successive steps:", pairs.round(1))
vv = global_err["velocity Verlet"]
print(
    f"velocity Verlet: halving dt divides the global error by "
    f"{vv[2] / vv[3]:.2f}; one-step error by "
    f"{local_err['velocity Verlet'][2] / local_err['velocity Verlet'][3]:.2f}"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.35)
)
for ax, errs, orders in (
    (left, global_err, (1, 2, 4)),
    (right, local_err, (2, 3, 5)),
):
    for (name, err), order in zip(errs.items(), orders, strict=True):
        ax.loglog(steps, err, "o", color=COLOURS[name], ms=3.5, label=name)
        ax.loglog(
            steps,
            err[-1] * (steps / steps[-1]) ** order,
            lw=0.8,
            **REFERENCE_STYLE,
        )
        ax.text(
            steps[0] * 1.15,
            err[0],
            f"{order}",
            color="0.4",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel(r"step $\delta t$")
    ax.set_xlim(0.009, 0.32)
left.set_ylabel("error after $t = 4$")
right.set_ylabel("error of one step")
left.legend(loc="lower right", fontsize=7)
viz.panel_tag(left, "a")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch09_integrators", "orders.pdf")))
