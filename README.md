# Molecular Dynamics and Learned Propagators

Brad Ayers

I started this book to work through molecular dynamics for myself: derive the
equations, write the calculation and check whether the result makes physical
sense. The explanations and exercises are written so that someone else can
follow the same route. The current text covers mechanics, molecular dynamics
and the foundations of learning from data, through Chapter 18, with an
animation gallery in Appendix F.

The book and notebooks belong together. Each notebook gives room to change a
parameter, inspect a trajectory and try the calculation before opening the
worked solution. `mdlab` is the accompanying Python package; its functions are
the implementations developed in the chapters, with tests and small examples
in their docstrings.

## Start the notebooks

Clone or download this repository, then open a terminal in its top-level
directory, alongside this README. Install Conda or Mamba if needed. The
environment uses Python 3.12.

```sh
conda env create -f environment.yml
conda activate mdbook
python -m pip install -e ./mdlab
make lab
```

Without `make`, the last command is `python -m jupyter lab`. In JupyterLab,
open `notebooks/01_motion.ipynb` and select the environment's Python kernel.
The notebooks through Chapter 11 have their small inputs included. Later
notebooks use saved trajectories, described below.

Read Chapter 0 for the route through the book. Chapters 1–7 develop the
mechanics; Chapters 8–17 turn it into molecular dynamics; Chapter 18 begins
the machine-learning material. Basic calculus and some familiarity with
Python help, but the code used in an example is available to inspect.

## Work through an example

Run the cells in order. Use **Restart Kernel and Run All Cells** when checking
that an edited notebook still works from the beginning. The sliders redraw
the figures, so it is useful to predict a change before moving them.

Exercise cells leave answers as `None`. Replace these with an expression or
calculation and run the check beneath it. A collapsed cell following the
exercise contains a worked code solution: click its collapsed bar to open it.
The reference functions in `mdlab/src/mdlab/` are also readable implementations
to compare against your own.

For personal notes, duplicate a notebook in the same folder and give it a name
ending in `_work.ipynb`, such as `01_motion_work.ipynb`. Git ignores these
copies. Keep them directly inside `notebooks/`, since some examples locate
data relative to that folder. Try changing a timestep, initial condition or
potential, then use the energy and other diagnostics to decide whether the
new result is trustworthy.

The package also works outside Jupyter. For example, a harmonic oscillator
with unit mass and spring constant makes a quarter turn in phase space:

```python
import numpy as np
from mdlab.hamiltonian import flow

q, p = flow(lambda q, p: q, lambda q, p: p,
            q0=[1.0], p0=[0.0], times=[0.0, np.pi / 2])
print(q[-1].round(8), p[-1].round(8))  # [0.] [-1.]
```

## Saved trajectories

The complete saved data occupy about 9.3 GiB. Small inputs are part of the Git
repository; large trajectories are separate release downloads. The original
arrays are retained, so the numerical examples use the same data as the book.

```sh
python scripts/manage_data.py list
python scripts/manage_data.py fetch
python scripts/manage_data.py check
```

`fetch` uses the GitHub repository configured as `origin` and its `data-v1`
release. That release must have been published by the maintainer. For a ZIP
download of the source, pass `--repo OWNER/REPOSITORY` using this repository's
GitHub name. No account or Git LFS setup is needed for a public data release.

It is possible to fetch just one part, for example
`python scripts/manage_data.py fetch 18`. Some notebooks use data from more
than one chapter; the [data guide](data/README.md) lists the required groups,
download sizes and an offline option. Allow about 19 GiB of free space for
the complete download and extraction. Compiling the book needs none of these
downloads.

## Compile the book

Install a full TeX Live or MacTeX distribution, including `latexmk`, pdfLaTeX
and BibTeX, and make sure they are on your terminal's path. The supplied
figure PDFs are ready to use.

```sh
make book
```

The result is `book/build/book.pdf`. To work on a single chapter:

```sh
make chapter CH=04
```

This writes `book/build/ch04.pdf`. References to other chapters may appear as
`??` in an isolated build; use the full book to resolve them. `make clean`
removes the main book's compilation files while keeping its PDF. On Windows,
WSL provides the shell and `make` used by these commands; the Python notebooks
can also be run directly with JupyterLab.

## Check or change the code

```sh
make test
```

This runs the package tests, docstring examples and data-helper checks. The
optional compiler and LAMMPS comparisons are selected by `make test-external`.
The editable installation
means that changes in `mdlab/src/mdlab/` are used by the environment; restart
a notebook kernel after changing an imported module.

Figure scripts live under `scripts/chNN_*`. Read the relevant script before
running it: some use saved data, while others perform a new calculation.
For example, `python scripts/ch18_learning/fig_basis.py` rebuilds the basis
function figure. The book can then be compiled again to include the change.

The notebook builders are in `scripts/notebooks/`. Running `make notebooks`
recreates the distributed notebooks and overwrites edits to those files.
Keep exercise attempts in separate `_work.ipynb` copies first. To prepare
changes for Git, run `python scripts/setup_git.py` once in your clone. It
keeps execution output out of commits while preserving output in your
working notebooks.

## Optional calculations and animations

The standard notebooks use saved external-calculator results. To repeat
Chapter 17's LAMMPS or MACE calculations, add their dependencies:

```sh
conda env update -n mdbook -f environment-simulations.yml
```

MACE calculations also obtain model weights on first use. In Notebook 17,
leave `RUN_EXTERNAL_CALCULATORS = False` when working from saved results.
The optional LAMMPS integration check requires a working LAMMPS and MPI
installation. FHI-aims must be installed separately.
Set `AIMS_ROOT` to its installation, or set `AIMS_BINARY` and `AIMS_SPECIES`
explicitly; `MPIRUN` and `AIMS_RANKS` select the MPI launcher and rank count.
The script `scripts/ch17_practice/runs_aims.py --help` lists its calculations.
The carbon regeneration scripts also use the `SiC.tersoff` parameter file
shipped with ASE's test data.

Notebook 10 compares Python, NumPy and Numba by default. Its optional Fortran
comparison needs a Fortran compiler, Meson and Ninja. After installing those,
set `RUN_FORTRAN = True`; the extension is built locally on first use.
If several compilers are installed, set `FC` to the full path of the intended
`gfortran` executable before starting Jupyter. On macOS it must be able to
locate the installed macOS SDK. To run just the optional compiler check:

```sh
make test-external TEST_ARGS="-q -k fortran"
```

The optional Manim example needs Manim, FFmpeg and TeX with `dvisvgm`. With
those installed, run:

```sh
python -m manim -qm --media_dir animations/media animations/ch00_smoke/smoke.py Lissajous
```

The notebook animations work without Manim.

The source for the saved water and energy-exchange GIFs is in `renders/`.
The energy-exchange animation can be rebuilt with the normal environment:

```sh
python renders/ch12_ensembles/exchange.py
```

The atom renderings need Blender. Put `blender` on your path or set `BLENDER`
to its executable, then run `python renders/ch11_constraints/water_frames.py
--movie` to rebuild the water GIF. Rendering these frames takes longer than
viewing the supplied animation.

## Repository layout

| Path | Contents |
| --- | --- |
| `book/` | LaTeX chapters, references, styles and saved figures |
| `notebooks/` | Chapter workbooks, exercises and worked solutions |
| `mdlab/` | Python source and tests |
| `scripts/` | Figure calculations, notebook builders and data helper |
| `data/` | Small inputs and the manifest for larger downloads |
| `animations/` | Optional Manim source |
| `renders/` | Saved-GIF scripts and optional Blender rendering source |

The [publishing notes](docs/publishing.md) describe the initial Git upload and
the separate data release. Build products, personal workbook copies and local
working notes stay out of Git.
