"""Write notebooks/00_template.ipynb, the layout used by the chapter notebooks.
Execute it afterwards with

    jupyter nbconvert --execute --to notebook --inplace notebooks/00_template.ipynb
"""

from pathlib import Path

import nbformat as nbf
from nbtools import set_cell_ids

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "00_template.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        "# Notebook 00: how the notebooks are laid out\n\n"
        "Every chapter of the book has a notebook like this one. Its sections "
        "carry the same numbers as the chapter's sections, so a pointer "
        "such as *notebook 09, shadow energy* in the book leads to the "
        "section here of the same name. The book carries the derivations; "
        "the notebook carries the same ideas as code to run and vary.\n\n"
        "Run the cells from the top. The first cell loads the house plotting "
        "style and fixes the random seed, so every figure and number below is "
        "reproducible."
    ),
    code(
        "%matplotlib inline\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import ipywidgets as widgets\n"
        "\n"
        "from mdlab import units, viz\n"
        "from mdlab.exercise import check\n"
        "\n"
        "viz.use_style()\n"
        "rng = np.random.default_rng(2026)"
    ),
    md(
        "## 1 Units\n\n"
        "`mdlab` measures energy in eV, length in Å, time in fs and mass in "
        "amu. Newton's second law then needs one conversion factor, because "
        "a force in eV/Å divided by a mass in amu is not an acceleration in "
        "Å/fs². The factor is computed from CODATA constants in "
        "`mdlab.units`, never typed in."
    ),
    code(
        "print(f'1 eV/(Å amu) = {units.FORCE_TO_ACCEL:.6e} Å/fs²')\n"
        "print(f'1 amu Å²/fs²  = {units.MV2_TO_EV:.4f} eV')\n"
        "print(f'k_B           = {units.KB:.6e} eV/K')"
    ),
    md(
        "## 2 A figure with a slider\n\n"
        "Sliders change a parameter and redraw. Here a point moves as "
        "$(\\sin a t, \\sin b t)$; the sliders set $a$ and $b$."
    ),
    code(
        "t = np.linspace(0.0, 2.0 * np.pi, 1000)\n"
        "\n"
        "def draw(a=3, b=2):\n"
        "    fig, ax = plt.subplots(figsize=(viz.HALF, viz.HALF))\n"
        "    ax.plot(np.sin(a * t), np.sin(b * t), color=viz.ACCENT)\n"
        "    ax.set_aspect('equal')\n"
        "    ax.set_xlabel('$x$')\n"
        "    ax.set_ylabel('$y$')\n"
        "    plt.show()\n"
        "\n"
        "widgets.interact(draw, a=(1, 6), b=(1, 6));"
    ),
    md(
        "The sliders use whole numbers, so every curve closes within the "
        "displayed interval. Set $a = b$: both coordinates are then the same "
        "and the path lies on a diagonal line. Before trying another pair, "
        "sketch the shape you expect. Keep your sketch and a short note of "
        "what changed alongside the calculation in your own copy."
    ),
    md(
        "## Exercises\n\n"
        "Each exercise cell sets `answer = None`. Replace `None` with your "
        "result and run the cell: the check says whether it is right, and "
        "prints *not attempted yet* while it is still `None`."
    ),
    code(
        "# EXERCISE 0.1\n"
        "# The thermal energy k_B T at 300 K, in meV.\n"
        "answer = None\n"
        "\n"
        "check(answer, units.KB * 300.0 * 1e3, rtol=1e-3)"
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
set_cell_ids(nb)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
