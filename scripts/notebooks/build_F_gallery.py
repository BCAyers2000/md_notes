"""Write notebooks/F_gallery.ipynb, the companion to Appendix F.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_F_gallery.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/F_gallery.ipynb

It reads the cached runs of Chapters 11 to 17 from data/ and passes each
healthy and each broken record through diagnostics.health_report.
"""

from nbtools import code, md, write

CELLS = [
    md(r"""
    # Notebook F: A gallery of faults

    Use the saved faulty trajectories to practise reading the diagnostics
    before changing a simulation. Compare each record with its healthy
    counterpart, then explain which observation supports the diagnosis.
    `diagnostics.health_report` runs the checks permitted by each record;
    a missing check leaves that question open. The examples follow
    Appendix F. Run the cells in order.
    """),
    code(r"""
    %matplotlib inline
    import math
    import sys
    from pathlib import Path

    import ipywidgets as widgets
    import numpy as np

    from mdlab import diagnostics, statmech, units, viz
    from mdlab.cell import wrap

    viz.use_style()
    DATA = Path(viz.THEORY, "data")
    sys.path.insert(0, str(Path("..") / "scripts" / "ch11_constraints"))
    import ch11  # the water of Chapter 11


    def from_npz(path, ensemble="nve", temperature=None, every=1):
        # A record cached by Chapters 12 and 13: everything is in the file.
        r = np.load(path)
        n = r["positions"].shape[1]
        keys = dict(positions=r["positions"][::every], cell=r["cell"],
                    masses=r["masses"], kinetic=r["kinetic"],
                    ensemble=ensemble, symbols=["Ar"] * n)
        if "velocities" in r.files:
            keys["velocities"] = r["velocities"][::every]
        if ensemble == "nve":
            keys["times"] = r["times"]
            keys["potential"] = r["potential"]
        if temperature is not None:
            keys["temperature"] = temperature
        return keys
    """),
    md(r"""
    ## The records

    Water of Chapter 11 at fixed energy, with seven of its faults; argon
    of Chapter 12 started from a liquid, from a lattice, with its drift
    kept and with velocities spread evenly; Chapter 13's flying ice cube
    under rescaling and Berendsen coupling against CSVR; and Chapter 16's
    argon with its positions wrapped and its centre of mass carried along.
    """),
    code(r"""
    WATER = ch11.load()


    def water(name):
        r = np.load(DATA / "ch11_constraints" / "runs" / f"{name}.npz")
        # water at 300 K: 3N less the drift, and less the held distances
        # for rigid water; the run with too long a step is flexible
        n_free = 573 if name == "fault_step" else 381
        return dict(positions=r["frames"], cell=WATER["cell"],
                    masses=WATER["masses"], times=r["times"],
                    potential=r["potential"], kinetic=r["kinetic"],
                    symbols=WATER["symbols"], temperature=300.0,
                    n_free=n_free)


    RECORDS = {f"11 water, {name}": water(f"fault_{name}")
               for name in ("step", "cutoff", "units", "wrapped", "tolerance",
                            "drift")}
    RECORDS["11 water, healthy"] = water("rigid_healthy") if (
        DATA / "ch11_constraints" / "runs" / "rigid_healthy.npz").exists() \
        else None
    RUNS12 = DATA / "ch12_ensembles" / "runs"
    for name, label in (("liquid", "healthy liquid"), ("lattice",
                        "lattice start"), ("drift", "drift kept"),
                        ("uniform", "uniform velocities")):
        asked = float(np.load(RUNS12 / f"{name}.npz")["asked"])
        RECORDS[f"12 argon, {label}"] = from_npz(RUNS12 / f"{name}.npz",
                                                 temperature=asked)
    RUNS13 = DATA / "ch13_thermostats" / "runs"
    for name in ("ice_csvr", "ice_rescale", "ice_berendsen"):
        RECORDS[f"13 argon, {name}"] = from_npz(RUNS13 / f"{name}.npz",
                                                ensemble="nvt")
    liquid = np.load(RUNS12 / "liquid.npz")
    h = liquid["cell"]
    wrapped = np.array([wrap(x, h) for x in liquid["positions"][::10]])
    RECORDS["16 argon, wrapped"] = dict(positions=wrapped, cell=h,
                                        masses=liquid["masses"])
    kt_m = units.KB * 135.0 / (39.948 * units.MV2_TO_EV)
    shift = math.sqrt(3 * kt_m / 256) * liquid["frame_times"]
    RECORDS["16 argon, centre carried"] = dict(
        positions=liquid["positions"] + shift[:, None, None]
        * np.array([1.0, 0, 0]), cell=h, masses=liquid["masses"])
    RECORDS = {k: v for k, v in RECORDS.items() if v is not None}
    REPORTS = {k: diagnostics.health_report(**v) for k, v in RECORDS.items()}
    print(f"{len(REPORTS)} records")
    """),
    md(r"""
    ## The table

    Each row is a record, each column a check; `ok` passes, `FAIL` fails,
    and a dot means the record holds too little for that check.
    """),
    code(r"""
    CHECKS = []
    for report in REPORTS.values():
        for name in report.checks:
            if name not in CHECKS:
                CHECKS.append(name)
    SHORT = {name: f"c{k + 1}" for k, name in enumerate(CHECKS)}
    for name, short in SHORT.items():
        print(f"{short}: {name}")
    print()
    print(f"{'record':34s} " + " ".join(f"{s:>4s}" for s in SHORT.values()))
    for label, report in REPORTS.items():
        row = [("ok" if report.checks[c][2] else "FAIL")
               if c in report.checks else "." for c in CHECKS]
        print(f"{label:34s} " + " ".join(f"{x:>4s}" for x in row))
    """),
    md(r"""
    Every healthy record passes; each fault fails at
    least one check, and the check it fails names the fault: a step too
    long or a jumping cutoff the spread of the energy, wrong units the
    temperature against the value asked, a loose tolerance the drift,
    wrapping the moves between frames, a kept
    drift the centre of mass, a lattice start the temperature against the
    value asked, uniform velocities the kurtosis. Chapter 13's three runs
    all keep a drift on purpose, so all three fail the centre of mass;
    the flying ice cube itself, the drift's energy growing under rescaling
    and Berendsen coupling and not under CSVR, shows only over time, in
    the comparison of Chapter 13. Faults that no single record can show,
    a box too small or a thermostat of the wrong ensemble, need the
    comparisons of their chapters too and are listed in Appendix F with
    them. Water wrapped into its cell broke in its first steps and kept a
    single frame, so only its energy can be checked, and fails.
    """),
    code(r"""
    def show(record=list(REPORTS)[0]):
        print(REPORTS[record])


    widgets.interact(show, record=list(REPORTS));
    """),
]

if __name__ == "__main__":
    print("wrote", write(CELLS, "F_gallery.ipynb"))
