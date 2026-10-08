"""Appendix E: the version table and the numbers quoted in section E.2.

    python scripts/appE_environments/versions.py

Writes versions_table.tex and numbers.tex beside the appendix sources and
prints everything it writes.
"""

import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

OUT = (
    Path(__file__).resolve().parents[2]
    / "book"
    / "appendices"
    / "appE_environments"
)

PACKAGES = [
    ("Python", None),
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("ASE", "ase"),
    ("pymatgen", "pymatgen"),
    ("LAMMPS", "lammps"),
    ("PyTorch", "torch"),
    ("e3nn", "e3nn"),
    ("mace-torch", "mace-torch"),
    ("Manim", "manim"),
]

# A cell of about a thousand atoms has a total energy of order 10^3 eV.
TOTAL_ENERGY = 5000.0


def version(dist):
    if dist is None:
        return platform.python_version()
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        if dist == "lammps":
            import lammps

            return (
                lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
                .version()
                .__str__()
            )
        raise


def mps_error():
    if not torch.backends.mps.is_available():
        return None
    try:
        torch.zeros(1, dtype=torch.float64, device="mps")
    except (TypeError, RuntimeError) as err:
        return f"{type(err).__name__}: {err}"
    return "no error"


rows = [(name, version(dist)) for name, dist in PACKAGES]
for name, v in rows:
    print(f"{name:12s} {v}")

table = [
    "\\begin{tabular}{ll}",
    "\\toprule",
    "Package & Version\\\\",
    "\\midrule",
]
table += [f"{name} & {{\\liningfigs {v}}}\\\\" for name, v in rows]
table += ["\\bottomrule", "\\end{tabular}"]
(OUT / "versions_table.tex").write_text("\n".join(table) + "\n")

gap64 = np.spacing(np.float64(TOTAL_ENERGY))
gap32 = np.spacing(np.float32(TOTAL_ENERGY))
err = mps_error()
print(
    f"gap at {TOTAL_ENERGY} eV: float64 {gap64:.2e} eV, float32 {gap32:.2e} eV"
)
print("mps float64:", err)
kind = err.split(":")[0] if err and ":" in err else "an error"


def sci(x):
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"\\num{{{mantissa}e{int(exponent)}}}"


numbers = [
    f"\\newcommand{{\\EnvTotalEnergy}}{{\\SI{{{TOTAL_ENERGY:.0f}}}{{\\electronvolt}}}}",
    f"\\newcommand{{\\EnvGapDouble}}{{{sci(gap64)}\\,\\si{{\\electronvolt}}}}",
    f"\\newcommand{{\\EnvGapSingle}}{{{sci(gap32)}\\,\\si{{\\electronvolt}}}}",
    f"\\newcommand{{\\EnvMpsError}}{{a \\texttt{{{kind}}}}}",
]
(OUT / "numbers.tex").write_text("\n".join(numbers) + "\n")
print("\n".join(numbers))
