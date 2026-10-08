"""Helpers for the scripts that write the chapter notebooks.

Each builder lists its cells as plain text, indented to match the
surrounding Python; ``md`` and ``code`` remove that indentation, so the
cells read in the builder exactly as they will in the notebook.
"""

from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import nbformat

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"


def set_cell_ids(notebook: nbformat.NotebookNode) -> None:
    """Give unchanged cells the same IDs each time the notebook is rebuilt."""
    occurrences = {}
    for cell in notebook.cells:
        content = f"{cell.cell_type}\n{cell.source}".encode("utf-8")
        digest = sha256(content).hexdigest()[:16]
        occurrence = occurrences.get(digest, 0)
        cell.id = f"{digest}-{occurrence}"
        occurrences[digest] = occurrence + 1


def md(text: str) -> nbformat.NotebookNode:
    """A markdown cell from indented text."""
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbformat.NotebookNode:
    """A code cell from indented text."""
    return nbformat.v4.new_code_cell(dedent(text).strip())


def hidden(text: str) -> nbformat.NotebookNode:
    """A code cell whose source starts collapsed, such as an answer key.

    Jupyter shows only a bar in place of the code until it is clicked, so
    the targets of the exercises stay out of sight while the reader works.
    """
    cell = code(text)
    cell.metadata["jupyter"] = {"source_hidden": True}
    cell.metadata["tags"] = ["hide-input"]
    return cell


def solution(text: str) -> nbformat.NotebookNode:
    """A worked solution: a collapsed code cell that checks its own answer.

    It follows the exercise it solves. Opened, it shows the calculation
    step by step; run, it confirms that the calculation passes the check.
    """
    cell = hidden(text)
    cell.metadata["tags"] = ["hide-input", "solution"]
    return cell


def write(cells: list, name: str) -> Path:
    """Write the cells to ``notebooks/<name>`` and return its path."""
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    set_cell_ids(notebook)
    path = NOTEBOOKS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path
