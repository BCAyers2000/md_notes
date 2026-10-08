"""House style for Manim scenes: white ground, the book's colours, and
mathematics set in the book's Palatino through a mathpazo TeX template.

Import before building any mobject:

    from house import ACCENT, INK, REFERENCE, axes
"""

from manim import (BLACK, WHITE, Axes, DashedLine, Dot, Line, MathTex,
                   Tex, TexTemplate, Text, config)

ACCENT = "#00546D"
OCHRE = "#C28E0E"
OXBLOOD = "#7C2529"
GREEN = "#5B7553"
REFERENCE = "#6E7275"
INK = "#202A33"

TEMPLATE = TexTemplate()
TEMPLATE.add_to_preamble(r"\usepackage[T1]{fontenc}\usepackage[sc]{mathpazo}")

config.background_color = WHITE
config.tex_template = TEMPLATE

for _cls in (MathTex, Tex):
    _cls.set_default(color=INK, tex_template=TEMPLATE)
Text.set_default(color=INK, font="Palatino")
for _cls in (Line, DashedLine, Dot):
    _cls.set_default(color=INK)


def axes(**kwargs) -> Axes:
    """Axes in the house look: thin ink lines, no arrow tips, no grid."""
    style = dict(color=INK, stroke_width=2, include_tip=False,
                 include_ticks=False)
    style.update(kwargs.pop("axis_config", {}))
    return Axes(axis_config=style, **kwargs)
