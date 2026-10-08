"""House style: the stylesheet loads and the colour law is complete."""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from mdlab import viz

HEX = re.compile(r"^#[0-9A-F]{6}$")


@pytest.mark.parametrize("notebook", [True, False])
def test_style_loads(notebook):
    viz.use_style(notebook=notebook)
    cycle = [
        c["color"].upper().lstrip("#") for c in plt.rcParams["axes.prop_cycle"]
    ]
    assert cycle == [c.lstrip("#") for c in viz.CYCLE]
    assert plt.rcParams["axes.spines.top"] is False


def test_colours_are_hex():
    for colour in (*viz.CYCLE, *viz.METHOD.values()):
        assert HEX.match(colour), colour
    for spec in viz.ELEMENT.values():
        assert HEX.match(spec["colour"])
        assert 0.2 < spec["radius"] < 1.5


def test_elements_of_the_running_example_present():
    assert {"Li", "C", "O", "H", "F", "P", "Si"} <= viz.ELEMENT.keys()


def test_method_colours_distinct():
    for family in (viz.METHOD, viz.BAROSTAT, viz.THERMOSTAT):
        assert len(set(family.values())) == len(family)


def test_figure_path_rejects_dotted_stems_and_creates_folders(tmp_path):
    with pytest.raises(ValueError):
        viz.figure_path("ch99_test", "step_1.5.pdf")
    path = viz.figure_path("ch99_test", "step_1p5.pdf")
    assert path.parent.is_dir()
    path.parent.rmdir()


def test_pi_ticks_label_multiples_of_half_pi():
    viz.use_style(notebook=True)
    fig, ax = plt.subplots()
    viz.pi_ticks(ax, 0, 4)
    ticks = ax.get_xticks()
    assert len(ticks) == 5
    assert abs(ticks[-1] - 2 * 3.141592653589793) < 1e-12
    plt.close(fig)
