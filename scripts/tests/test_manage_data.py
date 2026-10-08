"""Integrity checks for the trajectory download and extraction helper."""

# ruff: noqa: D101, D102

import argparse
import hashlib
import importlib.util
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "manage_data", Path(__file__).resolve().parents[1] / "manage_data.py"
)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


class DataArchives(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.previous = data.DATA
        data.DATA = self.root / "data"
        data.DATA.mkdir()
        self.payload = b"A saved trajectory\x00\xff\n" * 500
        self.entry = {
            "path": "ch18_learning/example.npz",
            "chapter": "ch18_learning",
            "size": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "archive": "example.zip",
        }
        self.archive = self.root / "example.zip"

    def tearDown(self):
        data.DATA = self.previous
        self.folder.cleanup()

    def archive_metadata(self):
        return {
            "name": self.archive.name,
            "size": self.archive.stat().st_size,
            "sha256": data.sha256(self.archive),
        }

    def write_fixture(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(self.entry["path"], self.payload)

    def test_fetch_verifies_bytes_and_preserves_local_edits(self):
        self.write_fixture()
        manifest = {
            "files": [self.entry],
            "archives": [self.archive_metadata()],
        }
        args = argparse.Namespace(
            archive_dir=self.root, replace=False, repo=None, tag="data-v1"
        )
        self.assertEqual(data.fetch_data(manifest, [self.entry], args), 0)
        destination = data.data_path(self.entry["path"])
        self.assertEqual(destination.read_bytes(), self.payload)
        destination.write_bytes(b"my experiment")
        with self.assertRaisesRegex(ValueError, "Preserving changed file"):
            data.fetch_data(manifest, [self.entry], args)
        self.assertEqual(destination.read_bytes(), b"my experiment")
        args.replace = True
        self.assertEqual(data.fetch_data(manifest, [self.entry], args), 0)
        self.assertEqual(destination.read_bytes(), self.payload)

    def test_corrupt_archive_is_rejected_before_extraction(self):
        self.write_fixture()
        manifest = {
            "files": [self.entry],
            "archives": [self.archive_metadata()],
        }
        raw = bytearray(self.archive.read_bytes())
        raw[len(raw) // 2] ^= 1
        self.archive.write_bytes(raw)
        args = argparse.Namespace(
            archive_dir=self.root, replace=False, repo=None, tag="data-v1"
        )
        with self.assertRaisesRegex(ValueError, "Archive does not match"):
            data.fetch_data(manifest, [self.entry], args)
        self.assertFalse(data.data_path(self.entry["path"]).exists())

    def test_parent_path_is_rejected(self):
        entry = dict(self.entry, path="../escape.npz")
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(entry["path"], self.payload)
        with self.assertRaisesRegex(ValueError, "Unsafe data path"):
            data.unpack_archive(self.archive, [entry], replace=False)
        self.assertFalse((self.root / "escape.npz").exists())

    def test_symlink_archive_member_is_rejected(self):
        info = zipfile.ZipInfo(self.entry["path"])
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(info, self.payload)
        with self.assertRaisesRegex(ValueError, "Invalid archive member"):
            data.unpack_archive(self.archive, [self.entry], replace=False)

    def test_existing_directory_symlink_cannot_escape_data(self):
        outside = self.root / "outside"
        outside.mkdir()
        (data.DATA / "ch18_learning").symlink_to(
            outside, target_is_directory=True
        )
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "leaves the project"):
            data.unpack_archive(self.archive, [self.entry], replace=False)
        self.assertFalse((outside / "example.npz").exists())

    def test_member_hash_failure_does_not_install_or_leave_temporary_file(
        self,
    ):
        self.write_fixture()
        entry = dict(self.entry, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "Wrong SHA-256"):
            data.unpack_archive(self.archive, [entry], replace=False)
        self.assertFalse(data.data_path(entry["path"]).exists())
        self.assertEqual(list(data.DATA.rglob(".extract-*")), [])

    def test_packing_is_deterministic(self):
        source = data.data_path(self.entry["path"])
        source.parent.mkdir()
        source.write_bytes(self.payload)
        data.write_archive(self.archive, [self.entry])
        second = self.root / "second.zip"
        data.write_archive(second, [self.entry])
        self.assertEqual(self.archive.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
