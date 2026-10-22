"""Check that native retargeting never changes the kernel tuning payload."""

import tempfile
import unittest
from pathlib import Path

from retarget_logic import copy_logic, retarget


class TestNativeLogic(unittest.TestCase):
    def test_copy_keeps_original_architecture_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vega10"
            root.mkdir()
            original = b"- version\n- vega10\n- gfx900\n- tuning\n"
            for index in range(46):
                (root / f"{index}.yaml").write_bytes(original)
            first = copy_logic(root)
            self.assertEqual(first, copy_logic(root))
            self.assertEqual(len(first), 46)
            for path in root.glob("*.yaml"):
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(
                    (root.parent / "gfx90c" / path.name).read_bytes(),
                    retarget(original),
                )

    def test_only_the_architecture_header_changes(self):
        original = b"- {MinimumRequiredVersion: 4.0.0}\n- vega10\n- gfx900\n- [Vega 10]\n- ISA: [9, 0, 0]\n"
        expected = original.replace(b"- gfx900\n", b"- gfx90c\n")
        self.assertEqual(retarget(original), expected)

    def test_retargeting_is_idempotent(self):
        content = b"- version\n- vega10\n- gfx90c\n- tuning data\n"
        self.assertEqual(retarget(retarget(content)), content)

    def test_unexpected_sources_are_rejected(self):
        for content in (
            b"",
            b"- version\n- vega20\n- gfx906\n- data\n",
            b"- version\n- vega10\n- gfx90a\n- data\n",
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                retarget(content)


if __name__ == "__main__":
    unittest.main()
