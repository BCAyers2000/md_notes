"""Figure fig:ro-remove: taking the drift and the rotation out of velocities.

A model ring of six carbon atoms, 1.4 Å from its centre, given velocities
in its plane made of a drift of (0.006, 0.003) Å/fs, a turn at
0.006 rad/fs and a random part of spread 0.003 Å/fs (seed 5).
(a) The velocities as made. (b) After
subtracting V = P/M from every atom: the total momentum is zero, but the
ring still turns. (c) After also subtracting the rigid rotation ω × r
with ω = I⁻¹L: what remains moves the atoms relative to one another.

Prints the total momentum, the angular momentum about the centre of mass
and the three parts of the kinetic energy, quoted in Section 5.6.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, energy, rotation, units, viz
from mdlab.viz import ACCENT, ARROW, ELEMENT, figure_path

N, RADIUS, MASS = 6, 1.4, 12.011
angles = 2 * np.pi * np.arange(N) / N
positions = RADIUS * np.column_stack(
    [np.cos(angles), np.sin(angles), np.zeros(N)]
)
masses = np.full(N, MASS)
rng = np.random.default_rng(5)
velocities = np.zeros((N, 3))
velocities[:, :2] = 0.003 * rng.normal(size=(N, 2))  # Å/fs, the random part
velocities += np.array([0.006, 0.003, 0.0])  # a drift
velocities += np.cross([0.0, 0.0, 0.006], positions)  # a turn, rad/fs

centre = dynamics.centre_of_mass(masses, positions)
drift_free = rotation.remove_rigid_motion(
    masses, positions, velocities, rotation=False
)
internal = rotation.remove_rigid_motion(masses, positions, velocities)
total = masses.sum()
drift = masses @ velocities / total
omega = rotation.angular_velocity(masses, positions, velocities)
inertia = rotation.inertia_tensor(masses, positions, centre)
stages = (("given", velocities), ("drift removed", drift_free))
for name, v in stages + (("rotation removed", internal),):
    p = dynamics.total_momentum(masses, v)
    ang = rotation.angular_momentum(masses, positions, v, centre)
    print(
        f"{name:17s} P = {np.array2string(p, precision=4)} amu Å/fs, "
        f"L_z = {ang[2]:+.4f} amu Å²/fs"
    )
k_total = energy.kinetic_energy(masses, velocities)
k_drift = energy.kinetic_energy([total], [drift])
k_turn = 0.5 * float(omega @ inertia @ omega) * units.MV2_TO_EV
k_rest = energy.kinetic_energy(masses, internal)
print(f"omega_z = {omega[2]:+.5f} rad/fs, I_zz = {inertia[2, 2]:.3f}")
p0 = dynamics.total_momentum(masses, velocities)
print(
    f"M = {total:.3f} amu, |P|² = {p0 @ p0:.5f}, from the four-figure "
    f"components {np.round(p0, 4) @ np.round(p0, 4):.4f} (amu Å/fs)²"
)
print(
    f"K = {k_total:.5f} eV = drift {k_drift:.5f} + turning {k_turn:.5f} "
    f"+ rest {k_rest:.5f} = {k_drift + k_turn + k_rest:.5f} eV"
)

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.4))
quivers = []
for ax, (v, tag, title) in zip(
    axes, ((velocities, "a", "initial velocities"),
           (drift_free, "b", "drift removed"),
           (internal, "c", "rotation removed")), strict=True
):
    ax.add_patch(plt.Circle((0, 0), RADIUS, fill=False, color="0.8", lw=0.6))
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=40,
        color=str(ELEMENT["C"]["colour"]),
        zorder=3,
    )
    arrows = ax.quiver(
        positions[:, 0],
        positions[:, 1],
        v[:, 0],
        v[:, 1],
        color=ACCENT,
        scale=0.01,
        **ARROW,
    )
    quivers.append(arrows)
    ax.plot(0, 0, "+", color="black", ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    viz.panel_tag(ax, tag)
    ax.set_title(title, fontsize=8, pad=15)
axes[0].quiverkey(quivers[0], 0.23, 0.08, 0.01,
                  r"$0.01$ \AA/fs", labelpos="E", coordinates="axes",
                  fontproperties={"size": 8})

print("wrote", viz.save(fig, figure_path("ch05_rotation", "remove.pdf")))
