"""Figure fig:ip-learned: a foundation model run by ASE and by LAMMPS.

(a) The time of one force call of MACE-MP-0 small and medium against the
number of atoms, LiC₆ cells of 28 to 1008, on five CPU threads in float64
and float32 and on the GPU in float32, D3 included, and D3 alone
(timing_mace.py, this machine). (b) LiC₆ (56 atoms) at fixed energy from
the relaxed cell with velocities drawn at 600 K, which settle near 245 K
(runs_mace.py, nve_*): the spread of K + U over that of K against the
step, in float64 (circles) and float32 (squares), and scaled to 1200 K,
settling near 600 K, in float64 (triangles; nve_600_*), with a line ∝ δt²
through the step of 1 fs. (c) The energy of graphite
against the length c of its cell, a held at 2.464 Å, with MACE-MP-0
alone and with D3, against Trucano and Chen's c (dotted).

Prints the times at 126 atoms and the GPU's gain; each run's
temperature, its spread with and without its trend, and the drift of its
mean energy; the ratios of the spreads between steps; the minima of
(c); and the energy and largest force difference between ASE and LAMMPS
(fix external) for MACE-MP-0 on a 56-atom LiC₆ cell.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from ch17 import C_GRAPHITE, RUNS, graphite, lic6, mace
from scipy.optimize import minimize_scalar

from mdlab import units, viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE, THRESHOLD_STYLE

timing = np.load(RUNS / "mace_timing.npz")
n = timing["atoms"]
k126 = int(np.flatnonzero(n == 126)[0])
for key in ("small_cpu_float64", "small_cpu_float32", "small_mps_float32",
            "medium_cpu_float64", "medium_cpu_float32", "medium_mps_float32",
            "d3"):
    print(f"{key}: {timing[key][k126] * 1000:.0f} ms at 126 atoms")
for model in ("small", "medium"):
    gain = timing[f"{model}_cpu_float32"] / timing[f"{model}_mps_float32"]
    print(f"{model}: the GPU is {gain.min():.1f} to {gain.max():.1f} times "
          f"faster than five CPU threads in float32")

steps = {"float64": (0.5, 1.0, 2.0, 3.0), "float32": (1.0, 2.0),
         "hot": (1.0, 2.0)}
FREEDOMS = 3 * 56 - 3  # the drift was removed at the start
spread = {}
for dtype, dts in steps.items():
    for dt in dts:
        name = (f"nve_600_{dt:g}_float64" if dtype == "hot"
                else f"nve_{dt:g}_{dtype}")
        if not (RUNS / f"{name}.npz").exists():
            continue
        run = np.load(RUNS / f"{name}.npz")
        e = run["kinetic"] + run["potential"]
        spread[dtype, dt] = e.std() / run["kinetic"].std()
        first, last = e[: len(e) // 4].mean(), e[-len(e) // 4:].mean()
        # the spread left once a straight line through the energy is removed
        line = np.polyval(np.polyfit(run["times"], e, 1), run["times"])
        flat = (e - line).std() / run["kinetic"].std()
        z = stats.drift_test(run["times"], e)[2]
        temperature = 2 * run["kinetic"] / (FREEDOMS * units.KB)
        print(f"{name}: at {temperature[300:].mean():.0f} K after the first "
              f"300 steps (drawn at {temperature[0]:.0f} K); spread of K + U "
              f"{spread[dtype, dt]:.4f} of K's, {flat:.4f} with its trend "
              f"removed; mean of the last quarter less the first "
              f"{(last - first) * 1000:+.2f} meV over 2 ps; drift "
              f"{abs(z):.1f} errors")
for a, b in ((0.5, 1.0), (1.0, 2.0), (2.0, 3.0)):
    print(f"float64 spread ratio {b:g}/{a:g} fs: "
          f"{spread['float64', b] / spread['float64', a]:.2f} against "
          f"(δt ratio)² {(b / a) ** 2:.2f}")

cs = np.linspace(5.9, 9.0, 32)
curves = {}
for label, dispersion in (("MACE-MP-0", False), ("with D3", True)):
    calc = mace(dtype="float64", dispersion=dispersion)

    def energy(c, calc=calc):
        g = graphite(c=c)
        g.calc = calc
        return g.get_potential_energy() / len(g)

    curves[label] = np.array([energy(c) for c in cs])
    best = minimize_scalar(energy, bounds=(5.9, 9.0), method="bounded",
                           options={"xatol": 1e-3})
    print(f"{label}: least energy at c = {best.x:.3f} Å, "
          f"{100 * (best.x / C_GRAPHITE - 1):+.1f}% on 6.711; binding "
          f"against c = 9 Å {1000 * (curves[label][-1] - best.fun):.1f} "
          f"meV per atom")

# MACE-MP-0 in LAMMPS through fix external, against ASE
os.environ.setdefault("FI_PROVIDER", "tcp")
from ase import Atoms  # noqa: E402
from ase.io import write  # noqa: E402
from lammps import lammps  # noqa: E402

calc = mace(dtype="float64")
atoms = lic6().repeat((2, 2, 2))
atoms.rattle(0.05, seed=1)
atoms.calc = calc
e_ase, f_ase = atoms.get_potential_energy(), atoms.get_forces()
folder = RUNS / "lammps"
folder.mkdir(parents=True, exist_ok=True)
write(folder / "lic6.data", atoms, format="lammps-data",
      specorder=["Li", "C"], masses=True)
lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
lmp.commands_string(f"""
units metal
atom_style atomic
atom_modify map array
read_data {folder / 'lic6.data'}
pair_style zero 6.0
pair_coeff * *
fix ext all external pf/callback 1 1
fix_modify ext energy yes
""")
symbols = {1: "Li", 2: "C"}


def callback(caller, step, nlocal, tag, x, f):
    types = lmp.numpy.extract_atom("type")[:nlocal]
    lo, hi, xy, yz, xz = lmp.extract_box()[:5]
    h = np.array([[hi[0] - lo[0], 0, 0], [xy, hi[1] - lo[1], 0],
                  [xz, yz, hi[2] - lo[2]]])
    frame = Atoms([symbols[t] for t in types],
                  positions=np.array(x[:nlocal]) - lo, cell=h, pbc=True)
    frame.calc = calc
    f[:] = frame.get_forces()
    lmp.fix_external_set_energy_global("ext", frame.get_potential_energy())


lmp.set_fix_external_callback("ext", callback, None)
lmp.command("run 0")
order = np.argsort(lmp.numpy.extract_atom("id")[:len(atoms)])
f_lmp = lmp.numpy.extract_atom("f")[:len(atoms)][order]
# LAMMPS turns a cell whose first vector is not along x into its own frame,
# so ASE's forces are taken on the cell as LAMMPS holds it
from ase.io import read  # noqa: E402

turned = read(folder / "lic6.data", format="lammps-data", style="atomic",
              units="metal")
turned.set_chemical_symbols(atoms.get_chemical_symbols())
turned.calc = calc
print(f"compared with ASE's forces on the cell as built, not as LAMMPS holds "
      f"it: largest difference {np.abs(f_lmp - f_ase).max():.1f} eV/Å")
f_ase = turned.get_forces()
print(f"MACE-MP-0 through fix external: energy {lmp.get_thermo('pe'):.6f} "
      f"against ASE's {e_ase:.6f} eV; largest force difference "
      f"{np.abs(f_lmp - f_ase).max():.1e} eV/Å")
lmp.close()

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.6))
ax = axes[0]
style = {"cpu_float64": dict(color=OXBLOOD, marker="o"),
         "cpu_float32": dict(color=OCHRE, marker="s"),
         "mps_float32": dict(color=ACCENT, marker="^")}
labels = {"cpu_float64": "CPU, float64", "cpu_float32": "CPU, float32",
          "mps_float32": "GPU, float32"}
for setting, kw in style.items():
    ax.loglog(n, timing[f"medium_{setting}"], lw=0.9, ms=3,
              label=labels[setting], **kw)
    ax.loglog(n, timing[f"small_{setting}"], lw=0.6, ms=2, ls="--",
              **{**kw, "marker": None})
ax.loglog(n, timing["d3"], **REFERENCE_STYLE, lw=0.8, label="D3 alone")
ax.set_xlabel(r"$N$")
ax.set_ylabel("s per force call")
ax.set_ylim(1e-3, 3e3)
ax.legend(fontsize=5.5, loc="upper left", handlelength=1.5)
viz.panel_tag(ax, "a")

ax = axes[1]
dts = np.array(steps["float64"])
ax.loglog(dts, [spread["float64", d] for d in dts], "o", color=OXBLOOD,
          ms=3.5, label="float64")
ax.loglog(steps["float32"], [spread["float32", d] for d in steps["float32"]],
          "s", mfc="none", color=ACCENT, ms=5, label="float32")
hot = [d for d in steps["hot"] if ("hot", d) in spread]
if hot:
    ax.loglog(hot, [spread["hot", d] for d in hot], "^", color=OCHRE, ms=4,
              label="near 600 K")
ax.loglog(dts, spread["float64", 1.0] * dts**2, **THRESHOLD_STYLE)
ax.set_xticks(dts, [f"{d:g}" for d in dts])
ax.set_xticks([], minor=True)
ax.set_yticks([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"])
ax.set_yticks([], minor=True)
ax.set_xlabel(r"$\delta t$ / fs")
ax.set_ylabel(r"spread of $K + U$ over $K$'s")
ax.legend(fontsize=6, loc="upper left")
viz.panel_tag(ax, "b")

ax = axes[2]
for label, colour in (("MACE-MP-0", OCHRE), ("with D3", ACCENT)):
    e = curves[label]
    ax.plot(cs, (e - e.min()) * 1000, color=colour, lw=1.0, label=label)
ax.axvline(C_GRAPHITE, **THRESHOLD_STYLE)
ax.set_xlabel(r"$c$ / \AA")
ax.set_ylabel("energy per atom / meV")
ax.set_ylim(0, 60)
ax.legend(fontsize=6, loc="upper right")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch17_practice", "learned.pdf")))
