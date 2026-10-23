"""Reject runtimes containing host libraries without their external device data."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_data import check_kernel_data


class KernelData(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.profile = {"gpu_targets": ["gfx1100"], "hipblaslt_targets": ["gfx1100"]}

    def data(self, component, target="gfx1100", suffix=".dat"):
        path = (
            self.root / "lib" / component / "library" / ("Tensile_" + target + suffix)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path

    def populate(self):
        for component in ("rocblas", "hipblaslt"):
            self.data(component)
            self.data(component, suffix=".hsaco")

    def test_library_without_data_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Missing rocblas"):
            check_kernel_data(self.root, self.profile)

    def test_metadata_without_code_or_wrong_architecture_is_rejected(self):
        self.populate()
        self.data("hipblaslt", suffix=".hsaco").unlink()
        self.data("hipblaslt", "gfx11001", ".hsaco")
        with self.assertRaisesRegex(AssertionError, "Incomplete hipblaslt.*gfx1100"):
            check_kernel_data(self.root, self.profile)

    def test_per_device_directory_symlinks_are_followed(self):
        self.populate()
        original = self.root / "lib/hipblaslt/library"
        device = self.root / "device/lib/hipblaslt/library/gfx1100"
        device.parent.mkdir(parents=True)
        original.rename(device)
        original.mkdir()
        (original / "gfx1100").symlink_to(device, target_is_directory=True)
        result = check_kernel_data(self.root, self.profile)["blas_kernel_data"]
        self.assertEqual(result["hipblaslt"]["gfx1100"]["code_objects"], 1)

    def test_disabled_hipblaslt_uses_only_rocblas(self):
        self.profile["hipblaslt_targets"] = []
        self.data("rocblas")
        self.data("rocblas", suffix=".co")
        result = check_kernel_data(self.root, self.profile)["blas_kernel_data"]
        self.assertEqual(set(result), {"rocblas"})

    def test_modern_compressed_tuning_metadata_is_accepted(self):
        self.populate()
        self.data("hipblaslt").unlink()
        self.data("hipblaslt", suffix=".dat.zlib")
        result = check_kernel_data(self.root, self.profile)["blas_kernel_data"]
        self.assertEqual(result["hipblaslt"]["gfx1100"]["metadata_files"], 1)


if __name__ == "__main__":
    unittest.main()
