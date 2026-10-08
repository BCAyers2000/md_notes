"""Write notebooks/14_pressure.ipynb, the companion to Chapter 14.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_14_pressure.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/14_pressure.ipynb

The runs are read from data/ch14_pressure/runs/, written by
scripts/ch14_pressure/runs.py, and Chapter 13's liquid from
data/ch13_thermostats/runs/. Sections 14.3, 14.4, 14.6, 14.7 and 14.8
end with a check against ASE; stochastic cell rescaling, which ASE lacks,
is checked against the exact volume density of an ideal gas.
"""

from nbtools import code, hidden, md, solution, write


def ase_check(name, what):
    """The closing cell of a section: mdlab against ASE."""
    return code(rf"""
    # Checked against ASE: {what}
    # (scripts/ch14_pressure/check_ase.py).
    check_ase.CHECKS["{name}"]()
    """)


SETUP = [
    md(r"""
    # Notebook 14: Pressure and barostats

    We calculate pressure from wall forces and from pair forces, then let
    the simulation cell respond to it. The useful checks are the volume
    distribution, the residual pressure and, for a layered solid, the
    separate changes within and across the sheets. Section numbers
    follow Chapter 14; run the cells in order.

    Longer trajectories are saved in `data/ch14_pressure/`. Pressure,
    Berendsen coupling and the MTK piston are compared with ASE. The exact
    ideal-gas volume distribution also checks stochastic cell rescaling.
    """),
    code(r"""
    %matplotlib inline
    import importlib.util
    import math
    import sys
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation

    from mdlab import barostats, cell, md, statmech, thermostats, units, viz
    from mdlab import virial
    from mdlab.exercise import check

    sys.path.insert(0, str(Path("..") / "scripts" / "ch14_pressure"))
    import ch14  # the shared set-up of the chapter's scripts

    # The checks against ASE, loaded from their file: the chapters'
    # script folders each have a check_ase.py.
    _spec = importlib.util.spec_from_file_location(
        "check_ase_14",
        Path("..") / "scripts" / "ch14_pressure" / "check_ase.py")
    check_ase = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(check_ase)

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    KB = units.KB
    GPA = units.EV_PER_A3_TO_GPA  # 1 eV/Å³ in GPa
    RNG = np.random.default_rng(14)


    def run(name):
        # Load a saved trajectory from scripts/ch14_pressure/runs.py.
        return np.load(ch14.RUNS / f"{name}.npz")
    """),
]

WALL = [
    md(r"""
    ## 14.1 Force on a wall

    Atoms that do not interact, between soft walls (springs of stiffness
    5 eV/Å² beyond each face), followed at fixed energy for 20 ps. The
    push of the atoms on the six walls is recorded at every step; its
    running mean per unit area is compared with $Nk_\mathrm{B}T/V$, with
    $T$ from the run's own kinetic energy.
    """),
    code(r"""
    class Walls:
        # The walls' energy and forces; the push is kept at each call.
        def __init__(self, side):
            self.side, self.push = side, []

        def __call__(self, r):
            u, f, push = virial.harmonic_walls(r, [self.side] * 3, 5.0)
            self.push.append(push.sum() / (6 * self.side**2))
            return u, f


    def walls(n=200, temperature=135, side=40.0):
        m = np.full(n, 39.948)
        rng = np.random.default_rng(1)
        r = rng.uniform(0.5, side - 0.5, (n, 3))
        v = statmech.thermal_velocities(m, temperature, rng,
                                        remove_drift=False)
        model = Walls(side)
        out = thermostats.run(model, m, r, v, 5.0, 4000, keep=())
        push = np.array(model.push)
        t_kin = 2 * out["kinetic"].mean() / (3 * n * KB)
        ideal = n * KB * t_kin / side**3
        running = np.cumsum(push) / np.arange(1, len(push) + 1)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.arange(len(push)) * 5e-3, running * GPA * 1e4,
                color=viz.ACCENT)
        ax.axhline(ideal * GPA * 1e4, **viz.REFERENCE_STYLE)
        ax.set_xlabel("time / ps")
        ax.set_ylabel("running mean of push per area / bar")
        plt.show()
        print(f"push per area {push.mean() * GPA * 1e4:.2f} bar; NkT/V "
              f"{ideal * GPA * 1e4:.2f} bar (T = {t_kin:.1f} K); ratio "
              f"{push.mean() / ideal:.3f}")


    widgets.interact(
        walls,
        n=widgets.IntSlider(200, min=50, max=400, step=50, **SLOW),
        temperature=widgets.IntSlider(135, min=50, max=400, step=25, **SLOW),
        side=widgets.FloatSlider(40.0, min=25.0, max=60.0, step=5.0, **SLOW));
    """),
    md(r"""
    The running mean settles within a few per cent
    of $Nk_\mathrm{B}T/V$ for every number of atoms, temperature and size;
    fewer atoms and larger boxes mean fewer bounces in 20 ps, so the curve
    takes longer to settle. The push is the momentum the bounces deliver.
    """),
]

VIRIAL = [
    md(r"""
    ## 14.2 The virial

    The interacting liquid between the same walls (cached, 100 ps after
    20 ps of settling): the running means of the push per unit area and
    of $(2K + \mathcal{W})/3V$, which equipartition for the positions says
    are the same.
    """),
    code(r"""
    lj = run("wall_lj")
    side = float(lj["length"])
    keep = lj["times"] >= 20000
    p_wall = lj["push"][keep].sum(1) / (6 * side**2)
    p_vir = (2 * lj["kinetic"][keep] + lj["virial"][keep]) / (3 * side**3)
    t_kin = 2 * lj["kinetic"][keep].mean() / (3 * 256 * KB)
    print(f"push per area {p_wall.mean() * GPA:.4f} GPa; virial pressure "
          f"{p_vir.mean() * GPA:.4f} GPa; NkT/V "
          f"{256 * KB * t_kin / side**3 * GPA:.4f} GPa")
    """),
]

PERIODIC = [
    md(r"""
    ## 14.3 Pressure in a periodic box

    Move the origin of a periodic box and wrap the positions again: the
    pressure from pair separations stays put, while $\sum_i
    \mathbf{r}_i\cdot\mathbf{F}_i$ from the wrapped positions does not. A
    frame of Chapter 13's liquid.
    """),
    code(r"""
    LIQ = np.load(ch14.ch13.RUNS / "csvr_1000_s0.npz")
    H, M = LIQ["cell"], LIQ["masses"]
    FRAME, VEL = LIQ["positions"][100], LIQ["velocities"][100]
    MODEL = ch14.argon_model(H)


    def origin(sx=0.0, sy=0.0, sz=0.0):
        shift = np.array([sx, sy, sz]) * H[0, 0]
        x = cell.wrap(FRAME + shift, H)
        _, f = MODEL(x)
        vol = cell.cell_volume(H)
        two_k = np.trace(virial.kinetic_tensor(M, VEL))
        pair = (two_k + np.trace(MODEL.virial)) / (3 * vol)
        naive = (two_k + virial.wrapped_virial(x, f)) / (3 * vol)
        print(f"pair separations: {pair * GPA:.4f} GPa; wrapped positions: "
              f"{naive * GPA:.4f} GPa")


    widgets.interact(origin, **{k: widgets.FloatSlider(0.0, min=0.0, max=1.0,
                                                       step=0.05, **SLOW)
                                for k in ("sx", "sy", "sz")});
    """),
    md(r"""
    The first number never changes; the second
    changes with every move of the origin that wraps some atom to the other
    side, since wrapping moves an atom by a lattice vector without
    changing its force.
    """),
    code(r"""
    def crystal(a=5.267):
        spacings = np.linspace(5.0, 5.8, 41)

        def state(spacing):
            r, h = ch14.fcc(4, spacing)
            model = ch14.argon_model(h)
            u, _ = model(r)
            vol = cell.cell_volume(h)
            return u / len(r), np.trace(model.virial) / (3 * vol)

        curve = np.array([state(s) for s in spacings])
        here = state(a)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(spacings, curve[:, 1] * GPA, color=viz.ACCENT)
        ax.axhline(0, **viz.THRESHOLD_STYLE)
        ax.plot(a, here[1] * GPA, "o", color=viz.OCHRE)
        ax.set_xlabel("a / Å")
        ax.set_ylabel("P / GPa")
        plt.show()
        print(f"a = {a:.3f} Å: P = {here[1] * GPA:+.4f} GPa, U/N = "
              f"{here[0]:.6f} eV")


    widgets.interact(crystal, a=widgets.FloatSlider(5.267, min=5.0, max=5.8,
                                                    step=0.01, **SLOW));
    """),
    md(r"""
    The pressure is positive when the crystal is
    squeezed below $a_0 = 5.267$ Å, zero there, where the energy is
    least, and negative, a tension, beyond it.
    """),
    ase_check("pressure", "the virial pressure of a strained, disordered "
              "crystal against\n    # ASE's LennardJones"),
]

TENSOR = [
    md(r"""
    ## 14.4 The pressure tensor

    Stretch the crystal from $a_0$ along $x$ by $\epsilon_{xx}$ and inspect
    the whole pressure tensor. Compare the component along the stretch
    with the two transverse components.
    """),
    code(r"""
    def stretch(exx=0.02):
        r, h = ch14.fcc(4, 5.26714)
        strain = np.eye(3)
        strain[0, 0] += exx
        model = ch14.argon_model(strain @ h)
        model(r @ strain.T)
        tensor = model.virial / cell.cell_volume(strain @ h) * GPA
        print("pressure tensor / GPa:")
        print(np.array2string(tensor, precision=4, suppress_small=True))


    widgets.interact(stretch, exx=widgets.FloatSlider(0.02, min=-0.03,
                                                      max=0.03, step=0.005,
                                                      **SLOW));
    """),
    md(r"""
    Stretched, $P_{xx}$ falls furthest below zero
    and $P_{yy} = P_{zz}$ less far; squeezed, the signs turn over. The
    off-diagonal components stay at zero, since a stretch along an axis of
    a cubic crystal shears nothing.
    """),
    ase_check("tensor", "the whole tensor against −(ASE's stress)"),
]

NPT = [
    md(r"""
    ## 14.5 What a barostat must do

    The exact density of the volume of $N$ ideal-gas atoms at constant
    pressure, $V^N e^{-\beta P_0V}$, and the compressibility of the liquid
    from three runs at fixed volume.
    """),
    code(r"""
    def ideal_density(n=20):
        kt, p0 = KB * 135, 1e-4
        v = np.linspace(1, (n + 1) * kt / p0 * 3, 400)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(v, virial.ideal_gas_volume_density(v, n, p0, 135.0),
                color=viz.ACCENT)
        ax.set_xlabel("V / Å³")
        ax.set_ylabel("density / Å⁻³")
        plt.show()
        print(f"mean {(n + 1) * kt / p0:.0f} Å³, relative spread "
              f"{1 / math.sqrt(n + 1):.3f}")


    widgets.interact(ideal_density, n=widgets.IntSlider(20, min=1, max=200,
                                                        **SLOW));
    """),
    md(r"""
    With few atoms the density is broad and leans to
    the right; with more it narrows, as $1/\sqrt{N + 1}$, towards a
    Gaussian.
    """),
    code(r"""
    volumes, means, errors = ch14.fixed_volume_pressures()
    for v, p, e in zip(volumes, means, errors):
        print(f"V {v:.0f} Å³: <P> {p * GPA:.4f} ± {e * GPA:.4f} GPa")
    kappa_t = ch14.compressibility_at(12306.0)
    print(f"κ_T at 12306 Å³: {kappa_t / GPA:.2f} GPa⁻¹; predicted spread of "
          f"V {math.sqrt(KB * 135 * 12306 * kappa_t):.0f} Å³")
    """),
]

RESCALING = [
    md(r"""
    ## 14.6 Weak coupling and stochastic cell rescaling

    20 ideal-gas atoms at 135 K and $P_0 = 10^{-4}$ eV/Å³ under each
    first-order barostat for 300 ps, with Langevin friction on the atoms,
    against the exact density.
    """),
    code(r"""
    def first_order(kind="cell rescaling", tau_ps=0.2):
        n, p0, temperature = 20, 1e-4, 135.0
        side = ch14.ideal_side(n, temperature, p0)
        m = np.full(n, 39.948)
        rng = np.random.default_rng(3)
        r = rng.uniform(0, side, (n, 3))
        v = statmech.thermal_velocities(m, temperature, rng,
                                        remove_drift=False)
        tau = tau_ps * 1000
        baro = (barostats.Berendsen(p0, tau, 1 / p0)
                if kind == "Berendsen" else
                barostats.StochasticCellRescaling(p0, temperature, tau,
                                                  1 / p0))
        friction = thermostats.Langevin(temperature, 0.01)
        out = barostats.run(md.NoForces(side * np.eye(3)), m, r, v, 10.0,
                            30000, baro, friction, np.random.default_rng(4),
                            every=10)
        vol = out["volume"][200:]
        grid = np.linspace(500, 6000, 300)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(vol, bins=np.linspace(500, 6000, 45), density=True,
                color=viz.BAROSTAT["Berendsen barostat" if kind == "Berendsen"
                                   else "stochastic cell rescaling"],
                alpha=0.5)
        ax.plot(grid, virial.ideal_gas_volume_density(grid, n, p0,
                                                      temperature),
                **viz.REFERENCE_STYLE)
        ax.set_xlabel("V / Å³")
        ax.set_ylabel("density / Å$^{-3}$")
        plt.show()
        exact = math.sqrt(n + 1) * KB * temperature / p0
        print(f"mean {vol.mean():.0f} Å³ (exact "
              f"{(n + 1) * KB * temperature / p0:.0f}); spread "
              f"{vol.std():.0f} Å³, {vol.std() / exact:.2f} of the exact")


    widgets.interact(first_order,
                     kind=widgets.ToggleButtons(options=["cell rescaling",
                                                         "Berendsen"]),
                     tau_ps=widgets.SelectionSlider(
                         options=[0.1, 0.2, 0.5, 1.0], value=0.2, **SLOW));
    """),
    md(r"""
    Under cell rescaling the histogram follows the
    dashed density at every $\tau_P$, its spread within about a tenth of
    the exact one in a run this short; under Berendsen's barostat it is a
    narrow spike, between a sixth and a half of the exact spread and
    narrower the longer $\tau_P$, centred a little low, near
    $Nk_\mathrm{B}T/P_0$.
    """),
    ase_check("berendsen", "Berendsen's barostat and thermostat, 40 steps "
              "of 5 fs, against\n    # NPTBerendsen"),
]

PISTON = [
    md(r"""
    ## 14.7 A piston with mass

    A piston resting on a two-dimensional gas of atoms that bounce off the
    walls and off it, pressed down by a fixed force and released from above
    its balance height: the gas is a spring and the piston swings on it.
    """),
    code(r"""
    def piston_animation(n=60, mass=10.0, force=60.0, start=1.4, frames=100,
                         per_frame=25):
        # Unit masses push on a lid of length 1 with n<v_y²>/h on average;
        # the lid starts above its balance and warms the gas as it falls.
        rng = np.random.default_rng(2)
        x = rng.uniform(0.05, 0.95, (n, 2)) * np.array([1.0, start])
        v = rng.normal(0, 1.0, (n, 2))
        top, top_v, dt = start, 0.0, 0.002
        heights, snapshots = [], []
        for step in range(frames * per_frame):
            x += v * dt
            for k in (0, 1):  # side and bottom walls
                low = x[:, k] < 0
                v[low, k] = np.abs(v[low, k])
            right = x[:, 0] > 1
            v[right, 0] = -np.abs(v[right, 0])
            hit = x[:, 1] > top
            if hit.any():  # elastic bounces off the piston
                for i in np.flatnonzero(hit):
                    u, w = v[i, 1], top_v
                    v[i, 1] = ((1 - mass) * u + 2 * mass * w) / (1 + mass)
                    top_v = ((mass - 1) * w + 2 * u) / (1 + mass)
                    x[i, 1] = top
            top_v -= force / mass * dt
            top += top_v * dt
            if step % per_frame == 0:
                heights.append(top)
                snapshots.append(x.copy())
        fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3), dpi=72)
        dots, = a.plot([], [], "o", ms=3, color=viz.ACCENT)
        lid, = a.plot([], [], color="black", lw=3)
        a.set_xlim(0, 1)
        a.set_ylim(0, max(heights) * 1.05)
        a.set_xticks([])
        a.set_ylabel("height")
        trace, = b.plot([], [], color=viz.ACCENT)
        b.set_xlim(0, frames)
        b.axhline(np.mean(heights[frames // 2:]), **viz.THRESHOLD_STYLE)
        b.set_ylim(min(heights) * 0.9, max(heights) * 1.05)
        b.set_xlabel("frame")
        b.set_ylabel("piston height")

        def draw(k):
            dots.set_data(snapshots[k][:, 0], snapshots[k][:, 1])
            lid.set_data([0, 1], [heights[k]] * 2)
            trace.set_data(np.arange(k + 1), heights[:k + 1])

        animation = FuncAnimation(fig, draw, frames=frames, interval=60)
        plt.close(fig)
        return HTML(animation.to_jshtml())


    piston_animation()
    """),
    md(r"""
    Released from above, the piston falls,
    overshoots and swings about the height at which the gas's push
    balances the force (dotted, the mean of the later swings), a mass on
    the spring of the gas. That height lies above $n k_\mathrm{B}T/F$ for
    the starting temperature, since the fall has warmed the gas.
    """),
    code(r"""
    def piston(tau_ps=1.0):
        names = {0.3: "npt_mtk_pd0.3_s0", 1.0: "npt_mtk_s0",
                 3.0: "npt_mtk_pd3_s0", 10.0: "npt_mtk_pd10_s0"}
        r = run(names[tau_ps])
        keep = r["times"] >= 10000
        period = ch14.crossings_period(r["times"][keep], r["volume"][keep])
        volume = r["volume"][keep].mean()
        kappa = ch14.compressibility_at(volume)
        predicted = 2 * math.pi * tau_ps * math.sqrt(
            257 * KB * 135 * kappa / (3 * volume))
        early = r["times"] <= 30000
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(r["times"][early] / 1000, r["volume"][early],
                color=viz.BAROSTAT["MTK piston"])
        ax.set_xlabel("time / ps")
        ax.set_ylabel("V / Å³")
        plt.show()
        print(f"period {period / 1000:.3f} ps, predicted {predicted:.3f} ps")


    widgets.interact(piston, tau_ps=widgets.SelectionSlider(
        options=[0.3, 1.0, 3.0, 10.0], value=1.0));
    """),
    md(r"""
    The volume swings faster for a lighter piston,
    with a period between 0.81 and 0.88 of $\tau_P$ at every setting,
    close to the $0.865\,\tau_P$ predicted; at $\tau_P = 10$ ps only a
    handful of swings fit in the first 30 ps.
    """),
    ase_check("mtk", "the isotropic MTK piston with chains, 40 steps of 5 "
              "fs, against\n    # IsotropicMTKNPT"),
]

SHAPE = [
    md(r"""
    ## 14.8 Isotropic, semi-isotropic and anisotropic coupling

    The layered model with guests inserted between its sheets, under each
    coupling (cached): how the height and width of the cell change, and the
    stress left in the plane and across it.
    """),
    code(r"""
    def coupling(couple="semi-isotropic"):
        r = run(f"layered_{couple}")
        strain = r["cells"] / np.diag(r["start_cell"]) - 1
        t = (np.arange(len(strain)) + 1) * 0.5
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t, strain[:, 2] * 100, color=viz.ACCENT, label="across the sheets, c")
        ax.plot(t, strain[:, 0] * 100, color=viz.OCHRE, ls="--", label="in the plane, a")
        ax.set_xlabel("time / ps")
        ax.set_ylabel("change / %")
        ax.legend()
        plt.show()
        late = r["tensors"][-20:]
        plane = 0.5 * (late[:, 0, 0] + late[:, 1, 1]).mean() * GPA
        print(f"in the plane {plane:+.3f} GPa, across "
              f"{late[:, 2, 2].mean() * GPA:+.3f} GPa")


    widgets.interact(coupling, couple=widgets.ToggleButtons(
        options=list(barostats.COUPLINGS), value="semi-isotropic"));
    """),
    md(r"""
    Under semi-isotropic and anisotropic coupling
    the height opens by about 7 to 8% while the width barely moves, and
    both pressures settle at zero; under isotropic coupling both grow by
    about 1% and the solid is left pulled in its plane and squeezed across
    it.
    """),
    ase_check("anisotropic", "anisotropic Berendsen coupling, 40 steps, "
              "against\n    # Inhomogeneous_NPTBerendsen"),
]

TRAJECTORY = [
    md(r"""
    ## 14.9 What the trajectory looks like

    The six faults of the chapter, each against a healthy run (dashed).
    """),
    code(r"""
    def _wrapped():
        p_pair, p_wrapped = [], []
        vol = cell.cell_volume(H)
        for k in range(0, len(LIQ["positions"]), 4):
            x = cell.wrap(LIQ["positions"][k], H)
            _, f = MODEL(x)
            two_k = np.trace(virial.kinetic_tensor(M, LIQ["velocities"][k]))
            p_pair.append((two_k + np.trace(MODEL.virial)) / (3 * vol))
            p_wrapped.append((two_k + virial.wrapped_virial(x, f)) / (3 * vol))
        t = np.arange(len(p_pair)) * 2.0
        return t, np.array(p_wrapped) * GPA, np.array(p_pair) * GPA, "P / GPa"


    def _volume(fault, healthy):
        a, b = run(fault), run(healthy)
        early = b["times"] <= a["times"][-1]
        return (a["times"] / 1000, a["volume"], b["times"][early] / 1000,
                b["volume"][early], "V / Å³")


    def _running_pressure(fault, healthy):
        out = []
        for name in (fault, healthy):
            r = run(name)
            p = np.trace(r["pressure"], axis1=1, axis2=2) / 3 * GPA
            out.append((r["times"] / 1000,
                        np.cumsum(p) / np.arange(1, len(p) + 1)))
        early = out[1][0] <= out[0][0][-1]
        return (*out[0], out[1][0][early], out[1][1][early],
                "running mean of P / GPa")


    FAULTS = {
        "(a) the virial from wrapped positions": _wrapped,
        "(b) the kinetic part dropped": lambda: _volume("fault_kinetic_s0",
                                                        "npt_scr_s0"),
        "(c) a narrow volume": lambda: _volume("npt_berendsen_s0",
                                               "npt_scr_s0"),
        "(d) a heavy piston": lambda: _volume("npt_mtk_pd10_s0",
                                              "npt_mtk_s0"),
        "(f) κ in the wrong unit": lambda: _running_pressure(
            "fault_kappa_s0", "npt_berendsen_s0"),
    }


    def fault(name="(b) the kinetic part dropped"):
        result = FAULTS[name]()
        fig, ax = plt.subplots(figsize=(6, 3))
        if len(result) == 4:
            t, bad, good, label = result
            ax.plot(t, good, label="reference", **viz.REFERENCE_STYLE)
            ax.plot(t, bad, color=viz.ACCENT, label="selected fault")
        else:
            t, bad, t_good, good, label = result
            ax.plot(t_good, good, label="reference", **viz.REFERENCE_STYLE)
            ax.plot(t, bad, color=viz.ACCENT, label="selected fault")
        ax.set_xlabel("time / ps")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        plt.show()


    widgets.interact(fault, name=widgets.Dropdown(options=list(FAULTS)));
    """),
    md(r"""
    (a) The pressure from wrapped positions sits far
    below the pair form and jumps from frame to frame. (b) Without the
    kinetic part the liquid settles at a smaller volume. (c) Berendsen's
    volume wanders within a narrower band than cell rescaling's. (d) The
    heavy piston swings slowly through the light one's band. (f) With
    $\kappa$ ten thousand times too small the running mean of the pressure
    stays at 0.09 GPa or below, short of the 0.1 GPa target. Fault (e),
    isotropic coupling of a layered solid, is the isotropic choice of
    Section 14.8.
    """),
]

EXERCISES = [
    md(r"""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises' works
    every exercise in full.

    Run the collapsed answer-key cell to prepare the checks. After
    each exercise a collapsed cell holds a worked solution in code. The
    derivations 14.7, 14.10 and 14.15 are checked by SymPy; 14.3, 14.5,
    14.6, 14.8, 14.12, 14.16, 14.17 and 14.21 ask for reasoning, answered in
    the book.
    """),
    hidden(r"""
    # Reference values, calculated from the exercise data.
    _kt = KB * 135
    _kappa = 1.46 * GPA  # 1.46 GPa⁻¹ in Å³/eV
    _n_air = 1e5 / (units.ELEMENTARY_CHARGE * 1e30) / (KB * 300)
    TARGETS = {
        "14.1": np.array([GPA, 1e5 / (units.ELEMENTARY_CHARGE * 1e30)]),
        "14.2": np.array([_n_air, _n_air ** (-1 / 3)]),
        "14.4": 2 ** (1 / 6) * 3.4,
        "14.9": (3.920 + 2 * 2.324) / 3,
        "14.11": np.array([21 * _kt / 1e-4, math.sqrt(21) * _kt / 1e-4]),
        "14.13": np.array([1.46 / 2.5, 1.46 / 2.5e-4 / 1000]),
        "14.14": math.sqrt(_kt * _kappa / 12306),
        "14.18": 2 * math.pi * math.sqrt(257 * _kt * _kappa / (3 * 12306)),
        "14.19": 1.108 ** (1 / 3) - 1,
        "14.20": _kappa * 256 * _kt / 12312,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 14.1: 1 eV/Å³ in GPa, and 1 bar in eV/Å³.
    units_pressure = None

    check(units_pressure, TARGETS["14.1"], rtol=1e-4, name="units")
    """),
    solution(r"""
    # SOLUTION 14.1. e J / 1e-30 m³ in Pa, over 1e9; and 1e5 Pa back.
    e = units.ELEMENTARY_CHARGE
    units_pressure = np.array([e * 1e30 / 1e9, 1e5 / (e * 1e30)])

    check(units_pressure, TARGETS["14.1"], rtol=1e-4, name="units");
    """),
    code(r"""
    # EXERCISE 14.2: air at 1 bar and 300 K: molecules per Å³, and the
    # spacing (V/N)^(1/3) in Å.
    air = None

    check(air, TARGETS["14.2"], rtol=1e-3, name="air")
    """),
    solution(r"""
    # SOLUTION 14.2. N/V = P/kT, with P in eV/Å³.
    density = 6.2415e-7 / (KB * 300)
    air = np.array([density, density ** (-1 / 3)])

    check(air, TARGETS["14.2"], rtol=1e-3, name="air");
    """),
    code(r"""
    # EXERCISE 14.4: the distance (Å) at which -r φ'(r) changes sign, argon.
    r_sign = None

    check(r_sign, TARGETS["14.4"], rtol=1e-4, name="distance")
    """),
    solution(r"""
    # SOLUTION 14.4. 12 (σ/r)^12 = 6 (σ/r)^6, so r = 2^(1/6) σ.
    r_sign = 2 ** (1 / 6) * 3.4

    check(r_sign, TARGETS["14.4"], rtol=1e-4, name="distance");
    """),
    code(r"""
    # EXERCISE 14.7, checked by SymPy: det(I + ε) to first order.
    e = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"e{i}{j}"))
    t = sp.Symbol("t")
    det = (sp.eye(3) + t * e).det()
    first = sp.expand(det).coeff(t, 1)
    print("first order:", first)
    assert sp.simplify(first - e.trace()) == 0
    """),
    code(r"""
    # EXERCISE 14.9: (C11 + 2 C12)/3 for C11 = 3.920, C12 = 2.324 GPa.
    b_cubic = None

    check(b_cubic, TARGETS["14.9"], rtol=1e-4, name="bulk modulus")
    """),
    solution(r"""
    # SOLUTION 14.9.
    b_cubic = (3.920 + 2 * 2.324) / 3

    check(b_cubic, TARGETS["14.9"], rtol=1e-4, name="bulk modulus");
    """),
    code(r"""
    # EXERCISE 14.10, checked by SymPy: the integral of V^n e^(-bV).
    V, b = sp.symbols("V b", positive=True)
    for n in range(6):
        value = sp.integrate(V**n * sp.exp(-b * V), (V, 0, sp.oo))
        assert sp.simplify(value - sp.factorial(n) / b ** (n + 1)) == 0
    print("n!/b^(n+1) for n = 0 to 5")
    """),
    code(r"""
    # EXERCISE 14.11: 20 atoms at 135 K and 1e-4 eV/Å³: mean volume and
    # its spread, in Å³.
    ideal = None

    check(ideal, TARGETS["14.11"], rtol=1e-3, name="volume")
    """),
    solution(r"""
    # SOLUTION 14.11. (N + 1) kT/P0 and √(N + 1) kT/P0.
    ideal = np.array([21 * KB * 135 / 1e-4, math.sqrt(21) * KB * 135 / 1e-4])

    check(ideal, TARGETS["14.11"], rtol=1e-3, name="volume");
    """),
    code(r"""
    # EXERCISE 14.13: relaxation times, in ps and in ns, for κ = 2.5 and
    # 2.5e-4 GPa⁻¹, with κ_T = 1.46 GPa⁻¹ and τ_P = 1 ps.
    relax = None

    check(relax, TARGETS["14.13"], rtol=1e-3, name="relaxation")
    """),
    solution(r"""
    # SOLUTION 14.13. τ_P κ_T/κ.
    relax = np.array([1.46 / 2.5, 1.46 / 2.5e-4 / 1000])

    check(relax, TARGETS["14.13"], rtol=1e-3, name="relaxation");
    """),
    code(r"""
    # EXERCISE 14.14: the spread of ln V for the liquid.
    log_spread = None

    check(log_spread, TARGETS["14.14"], rtol=1e-3, name="spread")
    """),
    solution(r"""
    # SOLUTION 14.14. √(kT κ_T/V), κ_T in Å³/eV.
    log_spread = math.sqrt(KB * 135 * 1.46 * GPA / 12306)

    check(log_spread, TARGETS["14.14"], rtol=1e-3, name="spread");
    """),
    code(r"""
    # EXERCISE 14.15, checked by SymPy: dK/dV at fixed π is −2K/3V.
    Vs, pi, m = sp.symbols("V pi m", positive=True)
    K = (pi * Vs ** sp.Rational(-1, 3)) ** 2 / (2 * m)
    assert sp.simplify(sp.diff(K, Vs) + 2 * K / (3 * Vs)) == 0
    print("dK/dV = -2K/(3V)")
    """),
    code(r"""
    # EXERCISE 14.18: the factor of τ_P in the piston's period.
    period_factor = None

    check(period_factor, TARGETS["14.18"], rtol=1e-3, name="factor")
    """),
    solution(r"""
    # SOLUTION 14.18. 2π √((N + 1) kT κ_T/3V).
    period_factor = 2 * math.pi * math.sqrt(257 * KB * 135 * 1.46 * GPA
                                            / (3 * 12306))

    check(period_factor, TARGETS["14.18"], rtol=1e-3, name="factor");
    """),
    code(r"""
    # EXERCISE 14.19: each length's growth if graphite's volume growth were
    # shared equally.
    share = None

    check(share, TARGETS["14.19"], rtol=1e-3, name="growth")
    """),
    solution(r"""
    # SOLUTION 14.19.
    share = 1.108 ** (1 / 3) - 1

    check(share, TARGETS["14.19"], rtol=1e-3, name="growth");
    """),
    code(r"""
    # EXERCISE 14.20: the fraction by which the volume shrinks, estimated
    # from κ_T and Nk_BT/V at the healthy volume, 12312 Å³.
    shrink = None

    check(shrink, TARGETS["14.20"], rtol=3e-3, name="shrink")
    """),
    solution(r"""
    # SOLUTION 14.20. κ_T N kT/V.
    shrink = 1.46 * GPA * 256 * KB * 135 / 12312

    check(shrink, TARGETS["14.20"], rtol=3e-3, name="shrink");
    """),
]

NOTES = [md(r"""
    ## Working notes

    Compare isotropic and semi-isotropic coupling for the layered solid.
    Record the changes in height and width and the two residual pressures.
    Explain why a correct mean scalar pressure can coexist with a solid
    that remains compressed in one direction.
    """)]

CELLS = (SETUP + WALL + VIRIAL + PERIODIC + TENSOR + NPT + RESCALING
         + PISTON + SHAPE + TRAJECTORY + EXERCISES + NOTES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "14_pressure.ipynb"))
