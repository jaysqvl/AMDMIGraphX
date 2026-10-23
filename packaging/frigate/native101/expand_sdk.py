"""Expand the pinned AMD wheel layout without pip resolution or GPU discovery."""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def expand(inputs: Path, destination: Path, source: Path) -> None:
    """Use AMD's initialization code to combine host and per-device packages."""
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / ".expanded").exists():
        for path in sorted(inputs.glob("*.whl")):
            with zipfile.ZipFile(path) as wheel:
                wheel.extractall(destination)
                for entry in wheel.infolist():
                    mode = entry.external_attr >> 16 & 0o777
                    if mode and not entry.is_dir():
                        os.chmod(destination / entry.filename, mode)
        with tarfile.open(inputs / "rocm-10.1.0.tar.gz") as archive:
            prefix = next(
                member.name.rsplit("/src/rocm_sdk/", 1)[0]
                for member in archive
                if "/src/rocm_sdk/" in member.name
            )
            archive.extractall(source)
        shutil.copytree(
            source / prefix / "src/rocm_sdk",
            destination / "rocm_sdk",
            dirs_exist_ok=True,
        )
    subprocess.run(
        [sys.executable, "-m", "rocm_sdk", "init"],
        env={**os.environ, "PYTHONPATH": str(destination.resolve())},
        check=True,
    )
    (destination / ".expanded").touch()


def main():
    """Reuse the exact source-build expansion in the Linux packaging preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("/inputs/sdk"))
    parser.add_argument("--destination", type=Path, default=Path("/opt/owned-deps/sdk"))
    parser.add_argument("--source", type=Path, default=Path("/work/sdk-source"))
    args = parser.parse_args()
    assert not Path("/dev/kfd").exists() and not Path("/dev/dri").exists()
    expand(args.inputs, args.destination, args.source)


if __name__ == "__main__":
    main()
