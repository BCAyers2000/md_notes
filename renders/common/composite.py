"""Composite a transparent render on white and crop to its content."""

from pathlib import Path

import numpy as np
from PIL import Image


def on_white(raw: str | Path, out: str | Path, margin: float = 0.014,
             dpi: int = 400) -> Path:
    """Flatten ``raw`` onto white and crop around its opaque pixels.

    The crop is found from the alpha channel, which is geometry, so the
    framing does not move when the lighting changes; the margin is a
    fraction of the content width, so it does not move with resolution.
    Writes ``out`` (PNG) and a PDF beside it.
    """
    image = Image.open(raw).convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(white, image).convert("RGB")
    solid = np.asarray(image)[..., 3] > 128
    rows = np.flatnonzero(solid.any(axis=1))
    cols = np.flatnonzero(solid.any(axis=0))
    pad = int(round(margin * (cols[-1] - cols[0] + 1)))
    box = (max(cols[0] - pad, 0), max(rows[0] - pad, 0),
           min(cols[-1] + pad + 1, flat.width),
           min(rows[-1] + pad + 1, flat.height))
    flat = flat.crop(box)
    out = Path(out)
    flat.save(out, dpi=(dpi, dpi))
    flat.save(out.with_suffix(".pdf"), "PDF", resolution=float(dpi))
    return out
