#!/usr/bin/env python3
"""Check, download or package the saved trajectories used by the workbooks.

Only the Python standard library is needed. With no chapter arguments,
commands select every chapter. A chapter argument selects its data folder;
see data/README.md for notebooks that also use an earlier chapter's data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.json"
CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    """Hash a file without holding a trajectory in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def data_path(name: str) -> Path:
    """Resolve a manifest path inside data/, rejecting links outside it."""
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or not relative.parts
    ):
        raise ValueError(f"Unsafe data path: {name!r}")
    path = DATA.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(DATA.resolve()):
        raise ValueError(f"Data path leaves the project: {name!r}")
    return path


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported data manifest version")
    archive_names = set()
    for archive in manifest["archives"]:
        name = archive["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.zip", name):
            raise ValueError(f"Unsafe archive name: {name!r}")
        if name in archive_names:
            raise ValueError(f"Duplicate archive: {name}")
        archive_names.add(name)
    names = set()
    for entry in manifest["files"]:
        data_path(entry["path"])
        if entry["path"] in names:
            raise ValueError(f"Duplicate data path: {entry['path']}")
        if "archive" in entry and entry["archive"] not in archive_names:
            raise ValueError(f"Unknown archive for {entry['path']}")
        names.add(entry["path"])
    return manifest


def select_files(manifest: dict, chapters: list[str]) -> list[dict]:
    available = {item["chapter"] for item in manifest["files"]}
    if not chapters or "all" in chapters:
        return manifest["files"]
    selected = set()
    for chapter in chapters:
        match = re.fullmatch(r"(?:ch)?(\d{1,2})(?:_[a-z]+)?", chapter)
        if not match:
            raise ValueError(f"Use a chapter number, such as 13: {chapter}")
        prefix = f"ch{int(match[1]):02d}_"
        found = {name for name in available if name.startswith(prefix)}
        if not found:
            raise ValueError(f"No saved data for chapter {chapter}")
        selected.update(found)
    return [item for item in manifest["files"] if item["chapter"] in selected]


def status(entry: dict) -> str:
    path = data_path(entry["path"])
    if not path.exists():
        return "missing"
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return "changed"
    return "ok" if sha256(path) == entry["sha256"] else "changed"


def list_data(manifest: dict, entries: list[dict]) -> None:
    print(
        f"{'chapter':24s} {'files':>6s} {'in Git / MiB':>14s} "
        f"{'download / MiB':>16s}"
    )
    for chapter in sorted({item["chapter"] for item in entries}):
        group = [item for item in entries if item["chapter"] == chapter]
        tracked = sum(item["size"] for item in group if "archive" not in item)
        archived = sum(item["size"] for item in group if "archive" in item)
        print(
            f"{chapter:24s} {len(group):6d} {tracked / CHUNK:14.1f} "
            f"{archived / CHUNK:16.1f}"
        )
    print(
        "\nBook compilation uses the included figures; "
        "no data download is needed."
    )


def check_data(entries: list[dict]) -> int:
    problems = []
    for entry in entries:
        state = status(entry)
        if state != "ok":
            problems.append((entry, state))
    for entry, state in problems:
        print(f"{state:7s} {entry['path']}")
    print(
        f"{len(entries) - len(problems)}/{len(entries)} "
        "files match the manifest."
    )
    if problems:
        print(
            "Use 'fetch' for missing archives. "
            "Changed files are preserved unless "
            "you explicitly pass --replace."
        )
    return int(bool(problems))


def github_repository(given: str | None) -> str:
    if given:
        repo = given
    else:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        remote = result.stdout.strip()
        match = re.fullmatch(
            r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
            r"([^/]+/[^/]+?)(?:\.git)?/?",
            remote,
        )
        if not match:
            raise ValueError(
                "Supply --repo OWNER/REPOSITORY, or configure a GitHub "
                "origin. The data release must already be published."
            )
        repo = match[1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("The repository must be OWNER/REPOSITORY")
    return repo


def verify_archive(path: Path, archive: dict) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != archive["size"]
        or sha256(path) != archive["sha256"]
    ):
        raise ValueError(
            f"Archive does not match its SHA-256 and size: {path}"
        )


def download_archive(archive: dict, repo: str, tag: str) -> Path:
    cache = DATA / ".downloads"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / archive["name"]
    if path.exists():
        verify_archive(path, archive)
        return path
    url = (
        f"https://github.com/{repo}/releases/download/{quote(tag, safe='')}/"
        f"{archive['name']}"
    )
    print(
        f"Downloading {archive['name']} ({archive['size'] / CHUNK:.1f} MiB)",
        flush=True,
    )
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=cache)
    try:
        with os.fdopen(fd, "wb") as target, urlopen(url, timeout=60) as source:
            shutil.copyfileobj(source, target, length=CHUNK)
        verify_archive(Path(temporary), archive)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def unpack_archive(path: Path, entries: list[dict], replace: bool) -> None:
    """Extract known regular files, checking their bytes before replacement."""
    expected = {entry["path"]: entry for entry in entries}
    with zipfile.ZipFile(path) as source:
        members = source.infolist()
        if len(members) != len(expected) or {
            item.filename for item in members
        } != set(expected):
            raise ValueError(
                f"Archive members differ from the manifest: {path}"
            )
        # Validate every name and type before writing any member.
        for member in members:
            data_path(member.filename)
            mode = member.external_attr >> 16
            if (
                member.is_dir()
                or stat.S_ISLNK(mode)
                or member.file_size != expected[member.filename]["size"]
            ):
                raise ValueError(f"Invalid archive member: {member.filename}")
        for member in members:
            entry = expected[member.filename]
            state = status(entry)
            if state == "ok":
                continue
            if state == "changed" and not replace:
                raise ValueError(
                    f"Preserving changed file {entry['path']}; "
                    "move it aside or use --replace"
                )
            destination = data_path(entry["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=".extract-", dir=destination.parent
            )
            try:
                with (
                    os.fdopen(fd, "wb") as target,
                    source.open(member) as item,
                ):
                    shutil.copyfileobj(item, target, length=CHUNK)
                if sha256(Path(temporary)) != entry["sha256"]:
                    raise ValueError(f"Wrong SHA-256 for {entry['path']}")
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)


def fetch_data(
    manifest: dict, entries: list[dict], args: argparse.Namespace
) -> int:
    needed = set()
    for entry in entries:
        state = status(entry)
        if state == "ok":
            continue
        if "archive" not in entry:
            raise ValueError(
                f"{entry['path']} is {state}; restore it from Git. "
                "It is not part of a data download."
            )
        if state == "changed" and not args.replace:
            raise ValueError(
                f"Preserving changed file {entry['path']}; "
                "move it aside or use --replace"
            )
        needed.add(entry["archive"])
    repo = (
        github_repository(args.repo)
        if needed and not args.archive_dir
        else None
    )
    for archive in manifest["archives"]:
        if archive["name"] not in needed:
            continue
        path = (
            args.archive_dir / archive["name"]
            if args.archive_dir
            else download_archive(archive, repo, args.tag)
        )
        verify_archive(path, archive)
        members = [
            entry
            for entry in manifest["files"]
            if entry.get("archive") == archive["name"]
        ]
        unpack_archive(path, members, args.replace)
        print(f"Ready: {archive['name']}", flush=True)
    return check_data(entries)


def write_archive(path: Path, entries: list[dict]) -> None:
    """Write reproducible ZIP_STORED bytes without recompressing NPZ files."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as target:
        for entry in sorted(entries, key=lambda item: item["path"]):
            if status(entry) != "ok":
                raise ValueError(
                    f"Source differs from the manifest: {entry['path']}"
                )
            info = zipfile.ZipInfo(
                entry["path"], date_time=(2026, 1, 1, 0, 0, 0)
            )
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED
            with (
                data_path(entry["path"]).open("rb") as source,
                target.open(info, "w") as out,
            ):
                shutil.copyfileobj(source, out, length=CHUNK)


def pack_data(manifest: dict, entries: list[dict], output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    selected = {entry.get("archive") for entry in entries}
    for archive in manifest["archives"]:
        if archive["name"] not in selected:
            continue
        destination = output / archive["name"]
        if destination.exists():
            verify_archive(destination, archive)
            print(f"Already packed: {destination.name}", flush=True)
            continue
        members = [
            entry
            for entry in manifest["files"]
            if entry.get("archive") == archive["name"]
        ]
        fd, temporary = tempfile.mkstemp(prefix=".pack-", dir=output)
        os.close(fd)
        try:
            write_archive(Path(temporary), members)
            verify_archive(Path(temporary), archive)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"Packed: {destination.name}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "check", "fetch", "pack"):
        command = commands.add_parser(name)
        command.add_argument(
            "chapters", nargs="*", help="chapter numbers; default: all"
        )
        if name == "fetch":
            command.add_argument(
                "--repo", help="GitHub OWNER/REPOSITORY; default: origin"
            )
            command.add_argument(
                "--tag", default="data-v1", help="release tag (data-v1)"
            )
            command.add_argument(
                "--archive-dir",
                type=Path,
                help="read local archives instead of downloading",
            )
            command.add_argument(
                "--replace",
                action="store_true",
                help="replace existing files that differ from the manifest",
            )
        if name == "pack":
            command.add_argument(
                "--output-dir", type=Path, default=ROOT / "release-assets"
            )
    args = parser.parse_args()
    try:
        manifest = load_manifest()
        entries = select_files(manifest, args.chapters)
        if args.command == "list":
            list_data(manifest, entries)
            return 0
        if args.command == "check":
            return check_data(entries)
        if args.command == "fetch":
            return fetch_data(manifest, entries, args)
        return pack_data(manifest, entries, args.output_dir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Data: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
