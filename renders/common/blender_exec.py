"""Locate Blender when a rendering script needs to start it."""

import os
import shutil
from pathlib import Path


def blender_executable() -> str:
    """Use BLENDER if set, otherwise find the blender command on PATH."""
    requested = os.environ.get("BLENDER", "blender")
    executable = shutil.which(str(Path(requested).expanduser()))
    if executable is None:
        raise SystemExit(
            "Blender was not found. Install it and put blender on PATH, "
            "or set BLENDER to the full path of its executable."
        )
    return str(Path(executable).resolve())
