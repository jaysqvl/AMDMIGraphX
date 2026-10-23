"""Reject OS libraries newer than Frigate's Debian 12 ABI before source builds."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from source_inputs import selected_artifacts, verify

BASELINE = {"GLIBC": (2, 36), "GLIBCXX": (3, 4, 30)}


def version_requirements(symbols: str) -> dict:
    """Inspect undefined symbol versions, rather than executing a shared object."""
    requirements = {}
    for line in symbols.splitlines():
        if "*UND*" not in line:
            continue
        if "GLIBC_PRIVATE" in line:
            raise ValueError("OS library depends on private glibc symbols")
        for namespace, number in re.findall(
            r"\b(GLIBCXX|GLIBC)_(\d+(?:\.\d+)+)\b", line
        ):
            version = tuple(map(int, number.split(".")))
            if version > BASELINE[namespace]:
                raise ValueError(f"{namespace}_{number} exceeds the Debian 12 ABI")
            requirements[namespace] = max(requirements.get(namespace, ()), version)
    if "GLIBC" not in requirements:
        raise ValueError("No glibc symbol requirements found")
    return requirements


def check_library(path: Path) -> dict:
    """Check a pinned ELF's actual import requirements with binutils or LLVM."""
    with path.open("rb") as stream:
        header = stream.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b">\x00":
        raise ValueError(f"Expected an x86-64 shared object: {path}")
    result = subprocess.run(
        ["objdump", "-T", str(path)], capture_output=True, text=True, check=True
    )
    return version_requirements(result.stdout)


def main():
    """Audit exported OS libraries independently of expensive AMD compilation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name("runtime-os-lock.json").read_text())
    artifacts = selected_artifacts(lock, [])
    for artifact in artifacts:
        verify(args.root / artifact["path"], artifact)
    with tempfile.TemporaryDirectory(prefix="os-abi-") as directory:
        extracted = Path(directory)
        for artifact in artifacts:
            subprocess.run(
                ["dpkg-deb", "-x", str(args.root / artifact["path"]), directory],
                check=True,
            )
        checks = {}
        for pattern in (
            "usr/lib/x86_64-linux-gnu/libnuma.so.*",
            "usr/lib/llvm-*/lib/libomp.so*",
        ):
            paths = {
                path.resolve() for path in extracted.glob(pattern) if path.is_file()
            }
            if not paths:
                raise ValueError(f"Missing exported OS library: {pattern}")
            for path in sorted(paths):
                checks[str(path.relative_to(extracted))] = check_library(path)
    print(json.dumps({"debian_12_os_library_requirements": checks}, indent=2))


if __name__ == "__main__":
    main()
