"""Read a notebook from stdin and write its source-only Git representation."""

import json
import sys
from hashlib import sha256

TRANSIENT_METADATA = {
    "widgets",
    "execution",
    "ExecuteTime",
    "workbook_verification",
}


def clean(notebook: dict) -> dict:
    """Keep source, attachments and exercise metadata; remove run output."""
    if notebook.get("nbformat") != 4 or not isinstance(
        notebook.get("cells"), list
    ):
        raise ValueError("expected a version 4 Jupyter notebook")

    metadata = notebook.setdefault("metadata", {})
    for key in TRANSIENT_METADATA:
        metadata.pop(key, None)

    occurrences = {}
    for cell in notebook["cells"]:
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("cell source must be text")

        # Match scripts/notebooks/nbtools.py so a rebuild keeps the same IDs.
        content = f"{cell['cell_type']}\n{source}".encode()
        digest = sha256(content).hexdigest()[:16]
        occurrence = occurrences.get(digest, 0)
        cell["id"] = f"{digest}-{occurrence}"
        occurrences[digest] = occurrence + 1

        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        metadata = cell.setdefault("metadata", {})
        for key in TRANSIENT_METADATA:
            metadata.pop(key, None)

    return notebook


def main() -> None:
    try:
        notebook = clean(json.load(sys.stdin))
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"Cannot clean notebook: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    json.dump(
        notebook, sys.stdout, ensure_ascii=False, indent=1, sort_keys=True
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
