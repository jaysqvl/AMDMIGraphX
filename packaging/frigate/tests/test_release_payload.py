"""Check large-asset splitting with small real files and independent hashes."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_payload import ASSET_LIMIT, export_archive


class ReleaseArchives(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.archive = self.root / "runtime.tar.gz"
        self.output = self.root / "release"
        self.output.mkdir()

    def check_archive(self, content, limit):
        self.archive.write_bytes(content)
        result = export_archive(self.archive, self.output, limit)
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(result["bytes"], len(content))
        self.assertFalse(self.archive.exists())
        return result

    def test_small_archive_is_kept_whole_at_boundary(self):
        result = self.check_archive(b"whole archive", 13)
        self.assertNotIn("parts", result)
        self.assertEqual((self.output / result["file"]).read_bytes(), b"whole archive")

    def test_split_parts_reconstruct_the_original_and_have_independent_hashes(self):
        content = bytes(range(113))
        result = self.check_archive(content, 17)
        restored = b""
        self.assertEqual(len(result["parts"]), 7)
        for index, part in enumerate(result["parts"], 1):
            self.assertEqual(part["file"], f"runtime.tar.gz.part{index:03d}")
            data = (self.output / part["file"]).read_bytes()
            self.assertEqual(part["bytes"], len(data))
            self.assertLessEqual(len(data), 17)
            self.assertEqual(part["sha256"], hashlib.sha256(data).hexdigest())
            restored += data
        self.assertEqual(restored, content)

    def test_empty_archive_or_invalid_part_limit_is_rejected_without_removal(self):
        for content, limit in ((b"", 17), (b"a", 0), (b"a", ASSET_LIMIT)):
            self.archive.write_bytes(content)
            with self.assertRaises(ValueError):
                export_archive(self.archive, self.output, limit)
            self.assertTrue(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
