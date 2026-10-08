"""Figure fig:ne-frames: one throw seen from two frames of reference.

A passenger in a train moving at u = (5, 0) m/s throws a ball straight
up at 6 m/s. (a) In the frame of reference of the train the ball rises
and falls along a vertical line. (b) From the platform the same ball
follows a parabola; velocity arrows (teal) differ between the two frames
of reference by u, while acceleration arrows (dark red) are the same in
both. (c) The velocities at the second instant, v = v' + u.

Prints the numbers that Section 2.2 (sec:ne-frames) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import dynamics, kinematics, viz
from mdlab.viz import ACCENT, OXBLOOD, REFERENCE, figure_path

U_TRAIN = np.array([5.0, 0.0])  # the train, seen from the platform, m/s
V_THROW = np.array([0.0, 6.0])  # the throw, seen in the train, m/s
GRAVITY = np.array([0.0, -G])  # the same acceleration in both frames
T_FLIGHT = 2 * V_THROW[1] / G
N_MARKS = 5  # equally spaced instants at which the ball is drawn
ARROW_SCALE_V, ARROW_SCALE_A = 1 / 0.16, 1 / 0.06  # drawn length per unit

# The throw from the platform, then the same throw seen in the train -------
t = np.linspace(0.0, T_FLIGHT, 200)
r_platform, v_platform = kinematics.constant_acceleration(
    t, np.zeros(2), V_THROW + U_TRAIN, GRAVITY
)
r_train, _ = dynamics.change_frame(t, r_platform, v_platform, U_TRAIN)

marks = np.linspace(0.0, T_FLIGHT, N_MARKS)
rm_platform, vm_platform = kinematics.constant_acceleration(
    marks, np.zeros(2), V_THROW + U_TRAIN, GRAVITY
)
rm_train, vm_train = dynamics.change_frame(
    marks, rm_platform, vm_platform, U_TRAIN
)

print(
    f"rise time {V_THROW[1] / G:.4f} s; flight {T_FLIGHT:.4f} s; "
    f"greatest height {V_THROW[1] ** 2 / (2 * G):.3f} m; the platform "
    f"sees it land {U_TRAIN[0] * T_FLIGHT:.3f} m along the track"
)
print(
    f"launch speed: in the train {np.linalg.norm(V_THROW):.3f} m/s, from "
    f"the platform {np.linalg.norm(V_THROW + U_TRAIN):.3f} m/s"
)

viz.use_style(notebook=False)
fig, (left, middle, right) = plt.subplots(
    1,
    3,
    figsize=(viz.FULL, 2.3),
    # each panel has equal aspect, so width ratios equal to x-span / y-span
    # give all three the same height
    gridspec_kw=dict(
        width_ratios=(1.8 / 3.5, 7.85 / 3.5, 6.9 / 6.0), wspace=0.22
    ),
)
falling = (np.zeros(N_MARKS), np.full(N_MARKS, -G))  # g at each mark

# (a) in the train: the path, the ball at each mark and its acceleration ---
left.plot(*r_train.T, color="black", lw=1.0)
left.quiver(
    rm_train[:, 0] - 0.25,  # beside the path, which runs up and down
    rm_train[:, 1],
    *falling,
    color=OXBLOOD,
    scale=ARROW_SCALE_A,
    **{**viz.ARROW, "width": 0.03},  # a narrow panel needs wider shafts
)
left.plot(*rm_train.T, "o", color="black", ms=3, zorder=4)
left.set_xlim(-0.9, 0.9)
left.set_ylim(-1.2, 2.3)
left.set_aspect("equal")
left.set_xticks([0])
left.set_xlabel(r"$x'$ / m")
left.set_ylabel(r"$y$ / m")
viz.panel_tag(left, "a")

# (b) from the platform: velocity, acceleration and the passenger's hand ---
middle.plot(*r_platform.T, color="black", lw=1.0)
middle.quiver(
    *rm_platform.T,
    *vm_platform.T,
    color=ACCENT,
    scale=ARROW_SCALE_V,
    **viz.ARROW,
)
middle.quiver(
    *rm_platform.T, *falling, color=OXBLOOD, scale=ARROW_SCALE_A, **viz.ARROW
)
middle.plot(*rm_platform.T, "o", color="black", ms=3, zorder=4)
middle.plot(
    U_TRAIN[0] * marks, np.zeros(N_MARKS), "s", color=REFERENCE, ms=4, zorder=2
)
middle.set_xlim(-0.4, 7.45)
middle.set_ylim(-1.2, 2.3)
middle.set_aspect("equal")
middle.set_xlabel(r"$x$ / m")
viz.panel_tag(middle, "b")

# (c) the velocities at the second mark: v = v' + u ------------------------
k = 1
v_seen_in_train, v_seen_from_platform = vm_train[k], vm_platform[k]
HEAD = dict(arrowstyle="-|>", lw=1.2, mutation_scale=8, shrinkA=0, shrinkB=0)
right.annotate("", U_TRAIN, (0, 0), arrowprops=dict(HEAD, color=REFERENCE))
right.annotate(
    "",
    U_TRAIN + v_seen_in_train,
    U_TRAIN,
    arrowprops=dict(HEAD, color=ACCENT, ls="--"),
)
right.annotate(
    "",
    v_seen_from_platform,
    (0, 0),
    arrowprops=dict(HEAD, color=ACCENT, shrinkB=3),
)
right.text(
    U_TRAIN[0] / 2,
    -0.45,
    r"$\mathbf{u}$",
    ha="center",
    va="top",
    color=REFERENCE,
)
right.text(
    U_TRAIN[0] + 0.25,
    v_seen_in_train[1] / 2,
    r"$\mathbf{v}'$",
    ha="left",
    va="center",
    color=ACCENT,
)
right.text(
    v_seen_from_platform[0] / 2 - 0.35,
    v_seen_from_platform[1] / 2 + 0.2,
    r"$\mathbf{v}$",
    ha="right",
    va="bottom",
    color=ACCENT,
)
right.set_xlim(-0.5, 6.4)
right.set_ylim(-1.6, 4.4)
right.set_aspect("equal")
right.set_xlabel(r"$v_x$ / m\,s$^{-1}$")
right.set_ylabel(r"$v_y$ / m\,s$^{-1}$")
viz.panel_tag(right, "c")
fig.align_xlabels()
print(
    f"second mark, t = {marks[k]:.4f} s: v' = {v_seen_in_train.round(3)}, "
    f"v = {v_seen_from_platform.round(3)} m/s"
)

print("wrote", viz.save(fig, figure_path("ch02_newton", "frames.pdf")))
