"""Verify native code inside ELF and LLVM compressed offload containers."""

import os
import subprocess
import tempfile
from pathlib import Path

# llvm/BinaryFormat/ELF.h, EF_AMDGPU_MACH_AMDGCN_* (processor mask 0xff).
MACHINES = {
    "gfx900": 0x2C,
    "gfx906": 0x2F,
    "gfx908": 0x30,
    "gfx90a": 0x3F,
    "gfx90c": 0x32,
    "gfx942": 0x4C,
    "gfx950": 0x4F,
    "gfx1030": 0x36,
    "gfx1100": 0x41,
    "gfx1101": 0x46,
    "gfx1200": 0x48,
    "gfx1201": 0x4E,
}


def inspect_gfx90c_code(path: Path) -> dict:
    """Keep the original native gfx90c validation entry point."""
    return inspect_code(path, "gfx90c")


def inspect_code(path: Path, arch: str) -> dict:
    """Inspect actual ELF flags, unpacking compressed Tensile bundles first."""
    with path.open("rb") as stream:
        header = stream.read(64)
    if header[:4] == b"CCOB":
        candidates = [
            Path("/opt/rocm") / suffix
            for suffix in (
                "bin/clang-offload-bundler",
                "lib/llvm/bin/clang-offload-bundler",
                "llvm/bin/clang-offload-bundler",
            )
        ]
        tool = next(
            (candidate for candidate in candidates if os.access(candidate, os.X_OK)),
            None,
        )
        assert tool is not None, "The pinned SDK offload-bundle tool is missing"
        result = subprocess.run(
            [str(tool), "--type=o", "--list", "--input=" + str(path)],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        targets = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        device_targets = [target for target in targets if "amdgcn" in target]
        assert device_targets and all(
            "--" + arch in target for target in device_targets
        ), (path.name, targets)
        objects = []
        with tempfile.TemporaryDirectory(prefix="gfx90c-unbundle-") as temporary:
            for index, target in enumerate(device_targets):
                output = Path(temporary) / f"kernel-{index}.hsaco"
                result = subprocess.run(
                    [
                        str(tool),
                        "--type=o",
                        "--unbundle",
                        "--input=" + str(path),
                        "--targets=" + target,
                        "--output=" + str(output),
                    ],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                assert result.returncode == 0, result.stderr
                objects.append({"target": target, **inspect_code(output, arch)})
        return {"container": "compressed-clang-offload-bundle", "objects": objects}
    assert len(header) == 64 and header[:6] == b"\x7fELF\x02\x01", path.name
    machine = int.from_bytes(header[18:20], "little")
    flags = int.from_bytes(header[48:52], "little")
    assert machine == 224 and flags & 0xFF == MACHINES[arch], (
        path.name,
        arch,
        machine,
        flags,
    )
    return {"container": "elf", "machine": machine, "flags": flags}
