"""ASE as a source of frames and of forces; observers of a run (Ch. 16)."""

import numpy as np
import pytest
from ase import Atoms
from ase import units as ase_units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.io import Trajectory
from ase.md.verlet import VelocityVerlet

from mdlab import io, md, units


def _argon():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(2)
    rng = np.random.default_rng(0)
    atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
    return atoms


def test_ase_model_runs_as_ase_does():
    atoms = _argon()
    calc = LennardJones(sigma=3.4, epsilon=0.01034, rc=5.0, smooth=True)
    v = np.random.default_rng(1).normal(0, 0.005, (len(atoms), 3))  # Å/fs
    model = io.AseModel(calc, atoms.get_chemical_symbols(),
                        np.array(atoms.get_cell()).T)
    ours = md.run(model, atoms.get_masses(), atoms.get_positions(), v,
                  np.array(atoms.get_cell()).T, 1.0, 20)
    theirs = atoms.copy()
    theirs.calc = LennardJones(sigma=3.4, epsilon=0.01034, rc=5.0,
                               smooth=True)
    theirs.set_velocities(v / np.sqrt(units.FORCE_TO_ACCEL))
    VelocityVerlet(theirs, timestep=1.0 * ase_units.fs).run(20)
    # ASE's femtosecond differs from the CODATA one by 5 parts in 10⁹
    assert ours["positions"][-1] == pytest.approx(theirs.get_positions(),
                                                  abs=1e-8)


def test_read_frames_converts_velocities(tmp_path):
    atoms = _argon()
    v = np.random.default_rng(2).normal(0, 0.005, (len(atoms), 3))
    path = str(tmp_path / "run.traj")
    with Trajectory(path, "w") as traj:
        for k in range(3):
            frame = atoms.copy()
            frame.positions += k * 0.1
            frame.set_velocities(v / np.sqrt(units.FORCE_TO_ACCEL))
            traj.write(frame)
    frames = io.read_frames(path)
    assert len(frames) == 3
    assert frames[2]["positions"] == pytest.approx(
        atoms.get_positions() + 0.2)
    assert frames[0]["velocities"] == pytest.approx(v, rel=1e-12)
    assert frames[0]["cell"] == pytest.approx(np.array(atoms.get_cell()).T)


def test_observers_see_each_recorded_frame():
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.8, 0, 0]],
                  cell=20 * np.eye(3))
    model = io.AseModel(LennardJones(sigma=3.4, epsilon=0.01034, rc=8.0),
                        ["Ar", "Ar"], 20 * np.eye(3))
    seen = []
    out = md.run(model, [39.948, 39.948], atoms.get_positions(),
                 np.zeros((2, 3)), 20 * np.eye(3), 5.0, 10, every=5,
                 observers=[lambda t, r, v: seen.append((t, r[1, 0]))])
    assert [t for t, _ in seen] == [0.0, 25.0, 50.0]
    assert [x for _, x in seen] == pytest.approx(
        out["positions"][:, 1, 0])


def test_thermostat_run_observers_and_last_pairs():
    from mdlab import potentials, thermostats

    atoms = _argon()
    h = np.array(atoms.get_cell()).T
    pair = potentials.with_cutoff(
        lambda r: potentials.lennard_jones(r, 0.01034, 3.4), 4.5, "switch")
    model = md.PairModel(pair, h, 4.5, 0.5)
    seen = []

    def observe(t, r, v):
        i, j, dist = model.last_pairs
        seen.append((t, len(i), float(np.min(dist))))

    thermostats.run(model, atoms.get_masses(), atoms.get_positions(),
                    np.zeros((len(atoms), 3)), 2.0, 6, every=3,
                    observers=[observe])
    assert [t for t, _, _ in seen] == [0.0, 6.0, 12.0]
    assert all(n > 0 for _, n, _ in seen)


def test_read_frames_keeps_extxyz_velocities(tmp_path):
    """Velocities written by write_extxyz come back unchanged, in Å/fs."""
    rng = np.random.default_rng(7)
    frames = [rng.uniform(0, 5, (4, 3)) for _ in range(3)]
    velocities = [rng.normal(0, 0.01, (4, 3)) for _ in range(3)]
    path = tmp_path / "run.extxyz"
    io.write_extxyz(str(path), ["Ar"] * 4, frames, 5.0 * np.eye(3),
                    velocities=velocities)
    read = io.read_frames(str(path))
    for frame, r, v in zip(read, frames, velocities, strict=True):
        assert frame["positions"] == pytest.approx(r, abs=1e-9)
        assert frame["velocities"] == pytest.approx(v, abs=1e-9)


def _switched_argon():
    from mdlab import potentials

    atoms = _argon()
    h = np.array(atoms.get_cell()).T
    pair = potentials.with_cutoff(
        lambda r: potentials.lennard_jones(r, 0.01034, 3.4), 4.5, "switch")
    return atoms, h, pair


def test_mdlab_calculator_energy_forces_and_stress():
    """The model's own numbers, and a stress equal to ASE's finite
    differences of the energy under strain."""
    atoms, h, pair = _switched_argon()
    model = md.PairModel(pair, h, 4.5, 0.5)
    u, f = model(atoms.get_positions())
    atoms.calc = io.mdlab_calculator(md.PairModel(pair, h, 4.5, 0.5))
    assert atoms.get_potential_energy() == pytest.approx(u, rel=1e-12)
    assert atoms.get_forces() == pytest.approx(f, abs=1e-12)
    numerical = atoms.calc.calculate_numerical_stress(atoms, d=1e-5)
    assert atoms.get_stress() == pytest.approx(numerical, abs=1e-8)


@pytest.mark.external
def test_lammps_table_gives_the_model_energy_and_forces(tmp_path):
    """LAMMPS with the table reproduces the model's energy and forces."""
    import os

    # MPICH's default network provider fails on some Macs at exit
    os.environ.setdefault("FI_PROVIDER", "tcp")
    lammps = pytest.importorskip("lammps")
    from ase.io import write

    atoms, h, pair = _switched_argon()
    model = md.PairModel(pair, h, 4.5, 0.5)
    u, f = model(atoms.get_positions())
    table = tmp_path / "argon.table"
    io.lammps_table(pair, 0.5, 4.5, 20001, str(table), "ARGON")
    data = tmp_path / "argon.data"
    write(str(data), atoms, format="lammps-data", masses=True)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_string(f"""
units metal
atom_style atomic
read_data {data}
pair_style table linear 20001
pair_coeff 1 1 {table} ARGON 4.5
run 0
""")
    order = np.argsort(lmp.numpy.extract_atom("id")[:len(atoms)])
    forces = lmp.numpy.extract_atom("f")[:len(atoms)][order]
    assert lmp.get_thermo("pe") == pytest.approx(u, rel=1e-6)
    assert forces == pytest.approx(f, abs=1e-5)
    lmp.close()


def test_read_run_from_an_ase_trajectory(tmp_path):
    atoms, h, pair = _switched_argon()
    v = np.random.default_rng(3).normal(0, 0.005, (len(atoms), 3))
    path = str(tmp_path / "run.traj")
    with Trajectory(path, "w") as traj:
        for k in range(3):
            frame = atoms.copy()
            frame.positions += 0.05 * k
            frame.set_velocities(v / np.sqrt(units.FORCE_TO_ACCEL))
            frame.calc = io.mdlab_calculator(md.PairModel(pair, h, 4.5, 0.5))
            frame.get_potential_energy()
            traj.write(frame)
    run = io.read_run(path)
    assert run["positions"].shape == (3, len(atoms), 3)
    assert run["velocities"][1] == pytest.approx(v, rel=1e-12)
    kinetic = 0.5 * units.MV2_TO_EV * np.sum(atoms.get_masses()[:, None]
                                             * v * v)
    assert run["kinetic"] == pytest.approx(kinetic, rel=1e-12)
    assert len(run["potential"]) == 3
