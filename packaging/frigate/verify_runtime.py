"""Verify the packaged runtime with no network, GPU, cameras, or private data."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from array import array
from pathlib import Path

from gfx90c_code import inspect_code
from kernel_data import check_kernel_data


def x86_elf(path: Path) -> bool:
    """Distinguish host binaries from GPU code and linker scripts."""
    with path.open("rb") as stream:
        header = stream.read(20)
    return header[:4] == b"\x7fELF" and header[18:20] == b">\x00"


def check_linking(root: Path) -> dict:
    """Resolve the detector's dependency closure, then import its CPython binding."""
    binding = root / "lib/migraphx.cpython-311-x86_64-linux-gnu.so"
    assert binding.is_file()
    # ldd -r follows DT_NEEDED dependencies recursively. Start with our detector
    # libraries and tools, rather than unrelated SDK profiler/Python plugins
    # whose symbols are intentionally provided by their loading application.
    paths = {
        path.resolve()
        for path in (root / "lib/migraphx/lib").rglob("*.so*")
        if path.is_file()
    }
    paths.add(binding.resolve())
    paths |= {
        root / "bin" / name
        for name in ("migraphx-driver", "migraphx-hiprtc-driver", "rocminfo")
    }
    checked = []
    for path in sorted(paths):
        if not x86_elf(path):
            continue
        # Extension module symbols are supplied by CPython during import.
        arguments = ["ldd"] + ([] if path.resolve() == binding.resolve() else ["-r"])
        result = subprocess.run(
            [*arguments, str(path)], capture_output=True, text=True, check=False
        )
        details = result.stdout + result.stderr
        assert result.returncode == 0 and "not found" not in details, (path, details)
        assert "undefined symbol:" not in details, (path, details)
        checked.append(
            str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        )
    assert checked
    migraphx = importlib.import_module("migraphx")
    assert Path(migraphx.__file__).resolve() == binding.resolve()
    return {
        "native_files_checked": len(checked),
        "unresolved_dependencies": 0,
        "python_import": "passed",
    }


def check_reference() -> dict:
    """Check both reference and oneDNN CPU arithmetic using public buffer data."""
    import migraphx

    left = array("f", [-3, 0, 2, 8])
    right = array("f", [1, -2, 4, 0.5])
    expected = [-2, -2, 6, 8.5]
    for target in ("ref", "cpu"):
        program = migraphx.program()
        module = program.get_main_module()
        shape = migraphx.shape(type="float_type", lens=[4])
        a = module.add_parameter("a", shape=shape)
        b = module.add_parameter("b", shape=shape)
        output = module.add_instruction(migraphx.op("add"), [a, b])
        module.add_return([output])
        program.compile(migraphx.get_target(target))
        values = program.run(
            {"a": migraphx.argument(left), "b": migraphx.argument(right)}
        )
        assert list(memoryview(values[0])) == expected, target
    return {"reference_arithmetic": "passed", "onednn_cpu_arithmetic": "passed"}


def check_compiler(profile: dict) -> dict:
    """Verify real device objects produced offline by the linked rocMLIR API."""
    objects = []
    for arch in profile["gpu_targets"]:
        for dtype in ("fp32", "fp16"):
            path = Path("/compiler-check") / f"{arch}-{dtype}.hsaco"
            objects.append({"arch": arch, "dtype": dtype, **inspect_code(path, arch)})
    return {"offline_mlir_objects": objects}


def main():
    """Write evidence without discovering or initializing any GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not Path("/dev/kfd").exists() and not Path("/dev/dri").exists()
    assert os.environ["HSA_ENABLE_SDMA"] == "0"
    assert sys.version_info[:2] == (3, 11)
    profile = json.loads(args.profile.read_text())
    root = Path("/opt/rocm").resolve()
    manifest = json.loads((root / "share/frigate/source-build.json").read_text())
    assert manifest["runtime"]["rocm"] == profile["rocm"]
    assert manifest["gpu_qualified"] is False
    result = {
        **check_kernel_data(root, profile),
        **check_linking(root),
        **check_reference(),
        **check_compiler(profile),
        "gpu_devices_exposed": False,
        "gpu_tested": False,
        "private_model_or_configuration_required": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
