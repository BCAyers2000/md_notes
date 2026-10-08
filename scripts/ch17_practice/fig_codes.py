"""Figure fig:ip-codes: one model of liquid argon in three codes.

runs_argon.py: the switched Lennard-Jones model of Chapter 12, 256
atoms, run by mdlab, by ASE (through io.mdlab_calculator) and by LAMMPS
(through a table of 20 001 points). (a) At fixed energy from one start,
the largest difference of any coordinate from mdlab's, against time.
(b) D from the MSD over 2 to 20 ps of 200 ps runs at 135 K, from five
blocks of 38 ps after the first 10 ps, under Langevin friction of
1 ps⁻¹, CSVR and a Nosé-Hoover chain (τ_T = 1 ps), with one standard
error. (c) The time of one step of 10 fs, the least over the runs.

Prints the energy spreads at fixed energy, the differences at 1, 2 and
5 ps, the mean temperature with its error and the spread of the kinetic
temperature against the canonical √(2/N_f)T, D for each run, the 1 ns
Langevin runs (LAMMPS with zero no and zero yes), and the times.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from ch17 import RUNS, T_LIQUID, argon_model, ch12, liquid_start
from runs_argon import TABLE_POINTS

from mdlab import statmech, units, viz
from mdlab.analysis import stats, transport
from mdlab.cell import minimum_image
from mdlab.viz import ACCENT, OCHRE, OXBLOOD

CODES = {"mdlab": ACCENT, "ase": OCHRE, "lammps": OXBLOOD}
NAMES = {"mdlab": "mdlab", "ase": "ASE", "lammps": "LAMMPS"}
KINDS = {"langevin": "Langevin", "csvr": "CSVR", "nhc": "NH chain"}
NF = statmech.degrees_of_freedom(256)


def diffusion(run, blocks=5):
    """D with its standard error from blocks, after the first 10 ps."""
    late = run["times"] > 10000
    lag = run["times"][1] - run["times"][0]
    n = int(round(20000 / lag))
    ds = []
    for block in np.array_split(run["positions"][late], blocks):
        msd = transport.msd(block, n, remove_drift=True).sum(1)
        ds.append(transport.diffusion_coefficient(lag * np.arange(n + 1),
                                                  msd, 2000, 20000))
    ds = np.array(ds)
    return ds.mean(), ds.std(ddof=1) / np.sqrt(blocks)


load = {(c, k): np.load(RUNS / f"argon_{c}_{k}.npz")
        for c in CODES for k in ("nve", *KINDS)}
reference = load["mdlab", "nve"]["positions"]
for c in CODES:
    run = load[c, "nve"]
    e = run["kinetic"] + run["potential"]
    gap = np.abs(run["positions"] - reference).max(axis=(1, 2))
    print(f"{NAMES[c]} at fixed energy: spread of K + U {e.std() * 1000:.3f} "
          f"meV; largest difference from mdlab at 1, 2, 5 ps: {gap[100]:.1e},"
          f" {gap[200]:.1e}, {gap[-1]:.1e} Å")

results = {}
for k in KINDS:
    for c in CODES:
        run = load[c, k]
        t = 2 * run["kinetic"] / (NF * units.KB)
        late = run["times"] > 10000
        mean, err, _ = stats.standard_error(t[late])
        spread = t[late].std(ddof=1) / (np.sqrt(2 / NF) * T_LIQUID)
        d, d_err = diffusion(run)
        results[c, k] = (d, d_err)
        print(f"{KINDS[k]:9s} {NAMES[c]:6s}: T = {mean:.2f} ± {err:.2f} K, "
              f"spread {spread:.3f} of canonical; D = ({d * 1e4:.3f} ± "
              f"{d_err * 1e4:.3f})e-4 Å²/fs")
long = {}
for name, label in (("mdlab_langevin_long", "mdlab"),
                    ("lammps_langevin_long", "LAMMPS, zero no"),
                    ("lammps_langevin_zero", "LAMMPS, zero yes")):
    long[label] = diffusion(np.load(RUNS / f"argon_{name}.npz"), blocks=10)
    d, d_err = long[label]
    print(f"1 ns Langevin, {label}: D = ({d * 1e4:.3f} ± {d_err * 1e4:.3f})"
          f"e-4 Å²/fs")
for a_, b_ in (("LAMMPS, zero yes", "LAMMPS, zero no"),
               ("LAMMPS, zero yes", "mdlab"), ("LAMMPS, zero no", "mdlab")):
    gap = (long[a_][0] - long[b_][0]) / np.hypot(long[a_][1], long[b_][1])
    print(f"  {a_} against {b_}: {gap:.1f} combined errors")

# the table: LAMMPS's force on one pair against the model's
def table_miss():
    """LAMMPS's force on a pair against the model's, beyond 3 Å.

    LAMMPS splines the file's points and builds its own table of
    TABLE_POINTS entries equally spaced in r², then interpolates F/r
    linearly in r²; its error is largest midway between two entries, so
    the pair is placed at every such midpoint beyond 3 Å.
    """
    from lammps import lammps
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_string(f"""
units metal
atom_style atomic
atom_modify map array
region box block 0 40 0 40 0 40
create_box 1 box
create_atoms 1 single 5 20 20
create_atoms 1 single 9 20 20
mass 1 39.948
pair_style table linear {TABLE_POINTS}
pair_coeff 1 1 {RUNS / "lammps" / "argon.table"} ARGON {ch12.R_CUT}
""")
    rsq = np.linspace(0.5**2, ch12.R_CUT**2, TABLE_POINTS)
    middle = np.sqrt(0.5 * (rsq[1:] + rsq[:-1]))
    r = middle[middle > 3.0]
    got = np.empty(len(r))
    for k, x in enumerate(r):
        lmp.command(f"set atom 2 x {5 + x:.12f}")
        lmp.command("run 0 post no")
        ids = lmp.numpy.extract_atom("id")[:2]
        got[k] = lmp.numpy.extract_atom("f")[np.argmax(ids), 0]
    lmp.close()
    want = -argon_model(liquid_start(0)[2]).pair(r)[1]
    return r, got, want, rsq[1] - rsq[0]


# The cached trajectory comparisons suffice to redraw the figure.
# Add --check-table to repeat the independent LAMMPS force check.
if "--check-table" in sys.argv:
    r_pair, got, want, step_sq = table_miss()
    miss = np.abs(got - want)
    k = np.argmax(miss)
    print(f"LAMMPS's table: {TABLE_POINTS} entries {step_sq:.4f} Å² apart in "
          f"r², {step_sq / (2 * 3.0):.1e} Å in r at 3 Å; largest miss of the "
          f"force beyond 3 Å {miss[k]:.2e} eV/Å at {r_pair[k]:.4f} Å, "
          f"{miss[k] / abs(want[k]):.2e} of the force there")

h0 = liquid_start(0)[2]
closest = min(
    np.linalg.norm(minimum_image((frame[:, None] - frame[None]).reshape(
        -1, 3), h0), axis=1).reshape(len(frame), -1)[
        np.triu_indices(len(frame), 1)].min()
    for frame in np.load(RUNS / "argon_mdlab_csvr.npz")["positions"])
print(f"the closest two atoms come in mdlab's 200 ps under CSVR: "
      f"{closest:.2f} Å")
walls = {c: min(float(load[c, k]["wall"]) for k in ("nve", *KINDS))
         for c in CODES}
print("time per step: " + ", ".join(f"{NAMES[c]} {w * 1000:.2f} ms"
                                    for c, w in walls.items())
      + f"; mdlab / LAMMPS = {walls['mdlab'] / walls['lammps']:.0f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
for c in ("ase", "lammps"):
    run = load[c, "nve"]
    gap = np.abs(run["positions"] - reference).max(axis=(1, 2))
    ax.semilogy(run["times"][1:] / 1000, gap[1:], color=CODES[c], lw=0.9,
                label=NAMES[c])
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"largest difference / \AA")
ax.set_xlim(0, 5)
ax.legend(fontsize=7, loc="lower right")
viz.panel_tag(ax, "a")

ax = axes[1]
for j, c in enumerate(CODES):
    x = np.arange(len(KINDS)) + 0.22 * (j - 1)
    d = np.array([results[c, k][0] for k in KINDS]) * 1e4
    e = np.array([results[c, k][1] for k in KINDS]) * 1e4
    ax.errorbar(x, d, e, fmt="o", ms=3, color=CODES[c], label=NAMES[c],
                elinewidth=0.8, capsize=0)
ax.set_xticks(range(len(KINDS)), list(KINDS.values()), fontsize=7)
ax.set_ylabel(r"$D$ / $10^{-4}$ \AA$^2$\,fs$^{-1}$")
ax.set_ylim(2, 4.6)
ax.legend(fontsize=7, loc="lower right")
viz.panel_tag(ax, "b")

ax = axes[2]
ax.bar(range(3), [walls[c] * 1000 for c in CODES],
       color=[CODES[c] for c in CODES], width=0.6)
ax.set_xticks(range(3), [NAMES[c] for c in CODES], fontsize=7)
ax.set_ylabel("ms per step")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch17_practice", "codes.pdf")))
