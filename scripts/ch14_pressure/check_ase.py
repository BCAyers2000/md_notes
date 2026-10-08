"""mdlab's pressure and barostats against ASE's.

Run from scripts/ch14_pressure/: python check_ase.py [check ...], with the
checks pressure, tensor, berendsen, mtk and anisotropic (all if none).

- pressure, tensor: 256 argon atoms near fcc in a cell stretched along x
  and squeezed along z, each displaced at random by about 0.1 Å, with the
  Lennard-Jones energy shifted to zero at 8.5 Å; the virial pressure and
  the whole tensor against ASE's LennardJones calculator, whose stress is
  −W/V, and the tensor against −dU/dε by finite differences.
- berendsen, mtk, anisotropic: the same atoms, warm, 40 steps of 5 fs
  from the same start under mdlab's barostats and ASE's NPTBerendsen,
  IsotropicMTKNPT and Inhomogeneous_NPTBerendsen, with mdlab's switched
  model inside ASE as a calculator; the largest differences in position,
  velocity and cell.
"""

import math
import sys
import warnings

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ch14 import EPS, GPA, R_CUT, SIG, argon_model, argon_pair

from mdlab import barostats, md, potentials, thermostats, units
from mdlab.cell import cell_volume

warnings.simplefilter("ignore", DeprecationWarning)
ASE_FS = 1 / math.sqrt(units.FORCE_TO_ACCEL)  # one ASE time unit, in fs
KAPPA = 1 / (2.0 / GPA)  # 1/(2 GPa), in Å³/eV
P0 = 0.1 / GPA


def crystal(stretch=(1.02, 1.0, 0.97), a=5.4, noise=0.1, seed=1):
    s = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(4) for j in range(4)
                     for k in range(4)])
    h = np.diag(4 * a * np.array(stretch))
    rng = np.random.default_rng(seed)
    r = ((grid[:, None, :] + s[None]).reshape(-1, 3) / 4) @ h.T
    return r + noise * rng.standard_normal(r.shape), h, rng


def check_pressure(full_tensor=False):
    r, h, _ = crystal()
    pair = potentials.with_cutoff(
        lambda d: potentials.lennard_jones(d, EPS, SIG), R_CUT, "shift")
    model = md.PairModel(pair, h, R_CUT, 1.0)
    model(r)
    ours = model.virial / cell_volume(h)
    atoms = Atoms("Ar256", positions=r, cell=h.T, pbc=True)
    atoms.calc = LennardJones(sigma=SIG, epsilon=EPS, rc=R_CUT, smooth=False)
    stress = atoms.get_stress(voigt=False)
    if full_tensor:
        print(f"pressure tensor: largest difference from −(ASE's stress) "
              f"{np.abs(ours + stress).max() * GPA:.1e} GPa, of components "
              f"up to {np.abs(ours).max() * GPA:.4f} GPa")
        step, worst = 1e-6, 0.0
        for a in range(3):
            for b in range(3):
                e = np.zeros((3, 3))
                e[a, b] = step
                energies = []
                for sign in (1, -1):
                    strain = np.eye(3) + sign * e
                    m = md.PairModel(pair, strain @ h, R_CUT, 1.0)
                    energies.append(m(r @ strain.T)[0])
                slope = (energies[0] - energies[1]) / (2 * step)
                worst = max(worst, abs(slope + model.virial[a, b]))
        print(f"virial tensor against −dU/dε by finite differences (step "
              f"{step:g}): largest difference {worst:.1e} eV")
    else:
        mine, theirs = np.trace(ours) / 3, -np.trace(stress) / 3
        print(f"virial pressure: mdlab {mine * GPA:.6f} GPa, ASE "
              f"{theirs * GPA:.6f} GPa")


class Wrapped(Calculator):
    """An ASE calculator that returns mdlab's switched argon model."""

    implemented_properties = ["energy", "forces", "free_energy", "stress"]

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """Energy, forces and stress −W/V in the atoms' current cell."""
        super().calculate(atoms, properties, system_changes)
        h = np.array(self.atoms.cell).T
        model = argon_model(h, 0.5)
        u, f = model(self.atoms.positions)
        s = -model.virial / cell_volume(h)
        self.results = {"energy": u, "free_energy": u, "forces": f,
                        "stress": np.array([s[0, 0], s[1, 1], s[2, 2],
                                            s[1, 2], s[0, 2], s[0, 1]])}


def warm_start():
    r, h, rng = crystal((1.03, 1.0, 0.98), 5.26, 0.05)
    v = 0.003 * rng.standard_normal(r.shape)
    m = np.full(len(r), 39.948)
    atoms = Atoms("Ar256", positions=r, cell=h.T, pbc=True, masses=m)
    atoms.set_velocities((v - v.mean(0)) / math.sqrt(units.FORCE_TO_ACCEL))
    atoms.calc = Wrapped()
    return r, v - v.mean(0), h, m, atoms


def report(label, atoms, r, v, h):
    v_ase = atoms.get_velocities() * math.sqrt(units.FORCE_TO_ACCEL)
    print(f"{label}: largest difference in position "
          f"{np.abs(atoms.positions - r).max():.1e} Å, relative in velocity "
          f"{np.abs(v_ase - v).max() / np.abs(v).max():.1e}, in the cell "
          f"{np.abs(np.array(atoms.cell).T - h).max():.1e} Å")


def check_berendsen(couple="isotropic"):
    r, v, h, m, atoms = warm_start()
    kind = NPTBerendsen if couple == "isotropic" else \
        Inhomogeneous_NPTBerendsen
    kind(atoms, 5.0 / ASE_FS, temperature_K=100.0, pressure_au=P0,
         taut=100 / ASE_FS, taup=500 / ASE_FS, compressibility_au=KAPPA,
         fixcm=False).run(40)
    out = barostats.run(md.PairModel(argon_pair(), h, R_CUT, 0.5), m, r, v,
                        5.0, 40, barostats.Berendsen(P0, 500.0, KAPPA, couple),
                        thermostats.Berendsen(100.0, 100.0, 3 * len(m)),
                        keep=("positions", "velocities"))
    report(f"Berendsen, {couple}", atoms, out["positions"][-1],
           out["velocities"][-1], out["cell"][-1])


def check_mtk():
    r, v, h, m, atoms = warm_start()
    IsotropicMTKNPT(atoms, 5.0 / ASE_FS, temperature_K=100.0, pressure_au=P0,
                    tdamp=100 / ASE_FS, pdamp=500 / ASE_FS).run(40)
    out = barostats.run_mtk(md.PairModel(argon_pair(), h, R_CUT, 0.5), m, r,
                            v, 5.0, 40, 100.0, P0, 100.0, 500.0,
                            keep=("positions", "velocities"))
    scale = (out["volume"][-1] / cell_volume(h)) ** (1 / 3)
    report("MTK piston", atoms, out["positions"][-1], out["velocities"][-1],
           h * scale)


CHECKS = {
    "pressure": check_pressure,
    "tensor": lambda: check_pressure(full_tensor=True),
    "berendsen": check_berendsen,
    "mtk": check_mtk,
    "anisotropic": lambda: check_berendsen("anisotropic"),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or CHECKS:
        CHECKS[name]()
