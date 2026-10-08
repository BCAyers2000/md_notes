"""House style for figures and notebooks.

Every figure in the book and every plot in a notebook is drawn through
:func:`use_style`, which loads ``book/style/mdbook.mplstyle``. Book
figures are drawn at their final width and saved through the PGF backend
with :func:`save`, so that their labels are set in the body font of the
book. Notebook figures cannot go through LaTeX on every redraw, so
:func:`use_style` gives them Palatino, or the nearest installed match.

The colour law lives here, once, for notebooks and figure scripts alike:

- ``ACCENT`` (teal) is the method or quantity under discussion;
  ``REFERENCE`` (grey, dashed) is what it is measured against; thresholds
  and guide lines are dotted.
- ``CYCLE`` is the order of colours for further series.
- ``METHOD`` pins a colour to an integrator, thermostat or model the first
  time the book uses it, and keeps it for the rest of the book.
- ``ELEMENT`` gives the colour and display radius of each element.

Figure scripts also find here where to write (:func:`figure_path`), how
to mark an axis in multiples of π (:func:`pi_ticks`) and the arrow style
for vectors (``ARROW``).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

#: Root of the course tree, ``theory/``; mdlab is installed editable there.
THEORY = Path(__file__).resolve().parents[3]
#: The one stylesheet for book figures and notebook plots.
STYLE_FILE = THEORY / "book" / "style" / "mdbook.mplstyle"
#: Where book figures are written, one folder per chapter.
FIGURES = THEORY / "book" / "figures"

#: Final figure widths in inches: the full text block, and one of two.
FULL = 5.77
HALF = 2.83

ACCENT = "#00546D"
OCHRE = "#C28E0E"
OXBLOOD = "#7C2529"
GREEN = "#5B7553"
REFERENCE = "#6E7275"
#: Order of colours for series: teal, ochre, oxblood, green, then grey.
CYCLE = (ACCENT, OCHRE, OXBLOOD, GREEN, REFERENCE)

#: Line styles that carry a role rather than a series.
REFERENCE_STYLE = {"color": REFERENCE, "linestyle": "--"}
THRESHOLD_STYLE = {"color": REFERENCE, "linestyle": ":", "linewidth": 0.8}

#: Arrows for vectors drawn with ``ax.quiver``, in data units, so that an
#: arrow's length is its vector times ``1 / scale``.
ARROW = {
    "angles": "xy",
    "scale_units": "xy",
    "width": 0.007,
    "headwidth": 4,
    "headlength": 4.5,
    "headaxislength": 4,
    "zorder": 3,
}

#: Integrators, thermostats and models, with the colour each keeps for the
#: whole book. Add an entry in the session that first plots the method.
METHOD: dict[str, str] = {
    "forward Euler": OXBLOOD,  # Ch. 7
    "symplectic Euler": GREEN,  # Ch. 7
    "velocity Verlet": ACCENT,  # Ch. 9
    "RK4": OCHRE,  # Ch. 9
}

#: Barostats, with the colour each keeps for the whole book (Ch. 14). A
#: figure compares methods of one family, so colours repeat across
#: families but never within one.
BAROSTAT: dict[str, str] = {
    "stochastic cell rescaling": ACCENT,
    "Berendsen barostat": OCHRE,
    "MTK piston": OXBLOOD,
}

#: Thermostat colours used across Chapters 13--15.
THERMOSTAT: dict[str, str] = {
    "CSVR": ACCENT,
    "Berendsen": OCHRE,
    "Langevin": OXBLOOD,
    "Andersen": GREEN,
    "Nosé-Hoover": REFERENCE,
}

#: Element colours (sRGB hex) and display radii in Å for plots and renders.
#: The radii are for legibility, not physical size.
ELEMENT: dict[str, dict[str, str | float]] = {
    "H": {"colour": "#E8ECEF", "radius": 0.40},
    "Li": {"colour": "#A77AD6", "radius": 0.98},
    "C": {"colour": "#30363B", "radius": 0.75},
    "O": {"colour": "#E52A35", "radius": 0.68},
    "F": {"colour": "#6FA565", "radius": 0.59},
    "Si": {"colour": "#B39C7A", "radius": 0.90},
    "P": {"colour": "#E99131", "radius": 0.86},
}

_SCREEN_SERIFS = ("Palatino", "TeX Gyre Pagella", "Book Antiqua")

_PI_LABELS = {
    -4: r"$-2\pi$",
    -3: r"$-\frac{3\pi}{2}$",
    -2: r"$-\pi$",
    -1: r"$-\frac{\pi}{2}$",
    0: r"$0$",
    1: r"$\frac{\pi}{2}$",
    2: r"$\pi$",
    3: r"$\frac{3\pi}{2}$",
    4: r"$2\pi$",
    5: r"$\frac{5\pi}{2}$",
    6: r"$3\pi$",
}


def _screen_serif() -> str:
    """The first Palatino-like font installed, or DejaVu Serif."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _SCREEN_SERIFS:
        if name in installed:
            return name
    return "DejaVu Serif"


def use_style(notebook: bool = True) -> None:
    """Load the house stylesheet into matplotlib's settings.

    Parameters
    ----------
    notebook : bool
        If True, use an installed Palatino-like face for text and
        mathematics, so that plots on screen match the book. Figure
        scripts pass False and set their text through LaTeX in
        :func:`save`.
    """
    if not STYLE_FILE.is_file():
        raise FileNotFoundError(
            f"house stylesheet not found at {STYLE_FILE}; install mdlab "
            "editable from the course tree (pip install -e mdlab)"
        )
    plt.style.use(STYLE_FILE)
    if notebook:
        face = _screen_serif()
        mpl.rcParams.update(
            {
                "font.serif": [face],
                "mathtext.fontset": "custom",
                "mathtext.rm": face,
                "mathtext.it": f"{face}:italic",
                "mathtext.bf": f"{face}:bold",
                # the custom set builds every slot; the default 'cursive'
                # calligraphic slot warns about fonts macOS lacks
                "mathtext.cal": f"{face}:italic",
                "mathtext.bfit": f"{face}:italic:bold",
            }
        )


def figure_path(chapter: str, name: str) -> Path:
    """Where a book figure is written, creating its folder if needed.

    Parameters
    ----------
    chapter : str
        Chapter folder, such as ``"ch02_newton"``.
    name : str
        File name, such as ``"drag.pdf"``. LaTeX cannot include a figure
        with a dot in its stem, so write ``1p5`` rather than ``1.5``.

    Returns
    -------
    Path
        ``theory/book/figures/<chapter>/<name>``.
    """
    if "." in Path(name).stem:
        raise ValueError(f"dot in figure stem {name!r}; write 1p5, not 1.5")
    target = FIGURES / chapter / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save(fig: mpl.figure.Figure, path: str | Path) -> Path:
    """Save a book figure at its drawn size, with labels set by LaTeX.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A figure drawn at final width (``FULL`` or ``HALF`` inches).
    path : str or Path
        Output file. A ``.pdf`` goes through the PGF backend, so that its
        text is typeset like the book; other formats use matplotlib's
        default backend.

    Returns
    -------
    Path
        The file written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = "pgf" if path.suffix == ".pdf" else None
    fig.savefig(path, backend=backend)
    return path


def tint(colour: str, strength: float) -> tuple[float, float, float]:
    """Mix ``colour`` with white: strength 1 keeps it, 0 gives white.

    Shades of one colour tell apart the runs of a single method, so that
    the colours of ``METHOD`` keep their meaning.

    >>> tint("#000000", 0.25)
    (0.75, 0.75, 0.75)
    """
    rgb = mpl.colors.to_rgb(colour)
    return tuple(1 - strength * (1 - c) for c in rgb)


def panel_tag(ax: mpl.axes.Axes, letter: str) -> None:
    """Write the panel tag '(a)' above the axes, at the left."""
    ax.set_title(f"({letter})", loc="left")


def pi_ticks(ax: mpl.axes.Axes, lo: int, hi: int, axis: str = "x") -> None:
    """Put ticks at k π/2 for k from ``lo`` to ``hi``, labelled with π.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to mark.
    lo, hi : int
        First and last multiple of π/2, between −4 and 6.
    axis : {"x", "y"}
        Which axis to mark.
    """
    ks = range(lo, hi + 1)
    ticks = [k * np.pi / 2 for k in ks]
    labels = [_PI_LABELS[k] for k in ks]
    if axis == "x":
        ax.set_xticks(ticks, labels)
    else:
        ax.set_yticks(ticks, labels)
