"""Tests of mdlab.barostats: the loop, the ideal gas's volume, ASE."""

import math

import numpy as np
import pytest

from mdlab import barostats, md, potentials, statmech, thermostats, units
from mdlab.cell import cell_volume

EPS, SIG = 0.01034, 3.4
R_CUT, R_SWITCH = 8.5, 6.8
PAIR = potentials.with_cutoff(
    lambda d: potentials.lennard_jones(d, EPS, SIG), R_CUT, "switch",
    R_SWITCH)
ASE_FS = 1 / math.sqrt(units.FORCE_TO_ACCEL)  # one ASE time unit, in fs
KAPPA = 1 / (2.0 / units.EV_PER_A3_TO_GPA)  # 1/(2 GPa), in Å³/eV
P0 = 0.1 / units.EV_PER_A3_TO_GPA  # 0.1 GPa


def argon():
    """256 argon atoms near fcc in a cell under uneven stress, warm."""
    a = 5.26
    s = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(4) for j in range(4)
                     for k in range(4)])
    h = np.diag([4 * a * 1.03, 4 * a, 4 * a * 0.98])
    rng = np.random.default_rng(1)
    r = ((grid[:, None, :] + s[None]).reshape(-1, 3) / 4) @ h.T
    r = r + 0.05 * rng.standard_normal(r.shape)
    v = 0.003 * rng.standard_normal(r.shape)
    return h, r, v - v.mean(0), np.full(256, 39.948)


def _ase_atoms(h, r, v, m):
    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes

    class Wrapped(Calculator):
        implemented_properties = ["energy", "forces", "free_energy", "stress"]

        def calculate(self, atoms=None, properties=None,
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            cell = np.array(self.atoms.cell).T
            model = md.PairModel(PAIR, cell, R_CUT, 0.5)
            u, f = model(self.atoms.positions)
            s = -model.virial / cell_volume(cell)
            self.results = {"energy": u, "free_energy": u, "forces": f,
                            "stress": np.array([s[0, 0], s[1, 1], s[2, 2],
                                                s[1, 2], s[0, 2], s[0, 1]])}

    atoms = ase.Atoms("Ar256", positions=r, cell=h.T, pbc=True, masses=m)
    atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
    atoms.calc = Wrapped()
    return atoms


def _compare(atoms, r, v, h):
    v_ase = atoms.get_velocities() * math.sqrt(units.FORCE_TO_ACCEL)
    # the two libraries' values of k_B differ by 3.4e-7
    assert np.allclose(atoms.positions, r, atol=1e-6)
    assert np.allclose(v_ase, v, rtol=1e-4, atol=1e-8)
    assert np.allclose(np.array(atoms.cell).T, h, atol=1e-6)


@pytest.mark.parametrize("couple", ["isotropic", "anisotropic"])
def test_berendsen_against_ase(couple):
    pytest.importorskip("ase")
    from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen

    h, r, v, m = argon()
    atoms = _ase_atoms(h, r, v, m)
    dynamics = {"isotropic": NPTBerendsen,
                "anisotropic": Inhomogeneous_NPTBerendsen}[couple]
    dynamics(atoms, 5.0 / ASE_FS, temperature_K=100.0, pressure_au=P0,
         taut=100 / ASE_FS, taup=500 / ASE_FS, compressibility_au=KAPPA,
         fixcm=False).run(40)
    out = barostats.run(md.PairModel(PAIR, h, R_CUT, 0.5), m, r, v, 5.0, 40,
                        barostats.Berendsen(P0, 500.0, KAPPA, couple),
                        thermostats.Berendsen(100.0, 100.0, 768),
                        keep=("positions", "velocities"))
    _compare(atoms, out["positions"][-1], out["velocities"][-1],
             out["cell"][-1])


def test_mtk_against_ase():
    pytest.importorskip("ase")
    from ase.md.nose_hoover_chain import IsotropicMTKNPT

    h, r, v, m = argon()
    atoms = _ase_atoms(h, r, v, m)
    IsotropicMTKNPT(atoms, 5.0 / ASE_FS, temperature_K=100.0,
                    pressure_au=P0, tdamp=100 / ASE_FS,
                    pdamp=500 / ASE_FS).run(40)
    out = barostats.run_mtk(md.PairModel(PAIR, h, R_CUT, 0.5), m, r, v, 5.0,
                            40, 100.0, P0, 100.0, 500.0,
                            keep=("positions", "velocities"))
    scale = (out["volume"][-1] / cell_volume(h)) ** (1 / 3)
    _compare(atoms, out["positions"][-1], out["velocities"][-1], h * scale)


def test_mtk_keeps_its_extended_energy():
    h, r, v, m = argon()
    out = barostats.run_mtk(md.PairModel(PAIR, h, R_CUT, 0.5), m, r, v, 2.0,
                            500, 100.0, P0, 100.0, 300.0)
    # the piston and the chains trade eV with the atoms; the sum holds still
    assert np.ptp(out["volume"]) > 0.02 * out["volume"][0]
    exchanged = np.ptp(out["potential"] + out["kinetic"])
    assert exchanged > 1.0
    assert np.ptp(out["conserved"]) < 1e-4 * exchanged


def test_no_barostat_is_the_thermostat_loop():
    h, r, v, m = argon()
    out = barostats.run(md.PairModel(PAIR, h, R_CUT, 0.5), m, r, v, 5.0, 30,
                        keep=("positions",))
    ref = thermostats.run(md.PairModel(PAIR, h, R_CUT, 0.5), m, r, v, 5.0, 30)
    assert np.allclose(out["positions"][-1], ref["positions"][-1], atol=1e-12)
    assert np.allclose(out["volume"], cell_volume(h))


def test_berendsen_effective_energy():
    h, r, v, m = argon()
    out = barostats.run(md.PairModel(PAIR, h, R_CUT, 0.5), m, r, v, 2.0, 300,
                        barostats.Berendsen(P0, 200.0, KAPPA),
                        thermostats.CSVR(100.0, 100.0, 765),
                        np.random.default_rng(2))
    effective = (out["potential"] + out["kinetic"] + P0 * out["volume"]
                 - out["heat"] - out["work"])
    assert abs(out["work"][-1]) > 0.05
    assert np.ptp(effective) < 5e-3


def _ideal_gas_volumes(barostat, n_steps=60000, seed=4):
    n, temperature = 20, 300.0
    kt = units.KB * temperature
    side = ((n + 1) * kt / 1e-4) ** (1 / 3)
    rng = np.random.default_rng(seed)
    m = np.full(n, 39.948)
    r = rng.uniform(0, side, (n, 3))
    v = statmech.thermal_velocities(m, temperature, rng, remove_drift=False)
    out = barostats.run(md.NoForces(side * np.eye(3)), m, r, v, 10.0,
                        n_steps, barostat, thermostats.Langevin(300.0, 0.01),
                        np.random.default_rng(seed + 1), every=10)
    exact_mean = (n + 1) * kt / 1e-4
    exact_spread = math.sqrt(n + 1) * kt / 1e-4
    vol = out["volume"][100:]
    return vol.mean() / exact_mean, vol.std() / exact_spread


def test_cell_rescaling_samples_the_ideal_gas():
    scr = barostats.StochasticCellRescaling(1e-4, 300.0, 200.0, 1e4)
    mean, spread = _ideal_gas_volumes(scr)
    assert abs(mean - 1) < 0.03
    assert abs(spread - 1) < 0.08


def test_berendsen_narrows_the_ideal_gas():
    mean, spread = _ideal_gas_volumes(barostats.Berendsen(1e-4, 200.0, 1e4))
    assert spread < 0.6


def test_semi_isotropic_cell_rescaling_samples_the_volume():
    # the two equations add to the isotropic one, so V is sampled alike
    scr = barostats.StochasticCellRescaling(1e-4, 300.0, 200.0, 1e4,
                                            "semi-isotropic")
    mean, spread = _ideal_gas_volumes(scr)
    assert abs(mean - 1) < 0.03
    assert abs(spread - 1) < 0.08
