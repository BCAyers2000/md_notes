"""Configure this checkout to keep notebook run output out of Git commits."""

import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        sys.exit("Git is not installed or is not on PATH.")
    except subprocess.CalledProcessError:
        sys.exit("This is not a Git checkout. Clone or initialise it first.")
    if Path(result.stdout.strip()).resolve() != project:
        sys.exit(
            "The project must have its own Git repository; "
            "its parent is a different checkout."
        )

    settings = {
        "filter.notebook-clean.clean": shlex.join(
            [sys.executable, "scripts/clean_notebook.py"]
        ),
        "filter.notebook-clean.smudge": shlex.join(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
            ]
        ),
        "filter.notebook-clean.required": "true",
    }
    for key, value in settings.items():
        subprocess.run(
            ["git", "-C", str(project), "config", "--local", key, value],
            check=True,
        )
    print("Notebook filtering is configured for this checkout.")
    print(
        "Git will store source and exercise metadata; "
        "working notebook outputs stay in place."
    )


if __name__ == "__main__":
    main()
