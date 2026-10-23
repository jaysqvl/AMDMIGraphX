"""Keep native dependency checks strict without treating SDK plugins as executables."""

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "owned"))
import verify_runtime


class RuntimeLinking(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binding = self.native("lib/migraphx.cpython-311-x86_64-linux-gnu.so")
        self.detector = self.native("lib/migraphx/lib/libmigraphx_gpu.so.2")
        self.native("lib/migraphx/lib/libmigraphx_cpu.so.2")
        for tool in ("migraphx-driver", "migraphx-hiprtc-driver", "rocminfo"):
            self.native("bin/" + tool)
        self.plugin = self.native(
            "lib/python3.10/site-packages/rocpd/libpyrocpd.cpython-310-x86_64-linux-gnu.so"
        )
        self.module = types.SimpleNamespace(__file__=str(self.binding))

    def native(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELF" + bytes(14) + b">\x00")
        return path

    def check(self, ldd):
        with (
            patch.object(verify_runtime.subprocess, "run", side_effect=ldd),
            patch.object(
                verify_runtime.importlib, "import_module", return_value=self.module
            ),
        ):
            return verify_runtime.check_linking(self.root)

    def test_checks_detector_roots_with_relocations_and_imports_binding(self):
        calls = []

        def ldd(command, **kwargs):
            calls.append(command)
            # An SDK plugin would fail standalone, but is outside this closure.
            if command[-1] == str(self.plugin):
                return subprocess.CompletedProcess(
                    command, 0, "undefined symbol: PyExc_ImportError", ""
                )
            return subprocess.CompletedProcess(
                command, 0, "all dependencies resolved", ""
            )

        result = self.check(ldd)
        self.assertEqual(result["native_files_checked"], 6)
        self.assertEqual(result["python_import"], "passed")
        self.assertIn(["ldd", "-r", str(self.detector)], calls)
        self.assertIn(["ldd", str(self.binding)], calls)
        self.assertFalse(any(command[-1] == str(self.plugin) for command in calls))

    def test_rejects_unresolved_transitive_detector_symbols(self):
        def ldd(command, **kwargs):
            output = (
                "undefined symbol: hipMissingEntryPoint (/sdk/libamdhip64.so)"
                if command[-1] == str(self.detector)
                else "resolved"
            )
            return subprocess.CompletedProcess(command, 0, output, "")

        with self.assertRaises(AssertionError):
            self.check(ldd)

    def test_rejects_missing_transitive_detector_libraries(self):
        def ldd(command, **kwargs):
            return subprocess.CompletedProcess(
                command, 0, "librocblas.so.5 => not found", ""
            )

        with self.assertRaises(AssertionError):
            self.check(ldd)

    def test_rejects_a_binding_imported_from_another_environment(self):
        self.module.__file__ = str(self.root / "unrelated/migraphx.so")

        def ldd(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "resolved", "")

        with self.assertRaises(AssertionError):
            self.check(ldd)


if __name__ == "__main__":
    unittest.main()
