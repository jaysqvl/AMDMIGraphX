"""Check external BLAS kernel data without loading a GPU library."""

import argparse
import json
import os
import re
from pathlib import Path


def check_kernel_data(root: Path, profile: dict, component_filter=None) -> dict:
    """Require both tuning metadata and device objects for each enabled target."""
    checked = {}
    for component, targets in (
        ("rocblas", profile["gpu_targets"]),
        ("hipblaslt", profile["hipblaslt_targets"]),
    ):
        if not targets or component_filter and component != component_filter:
            continue
        directory = root / "lib" / component / "library"
        assert directory.is_dir(), f"Missing {component} kernel data: {directory}"
        # AMD's SDK merges per-device wheel directories using symlinks.
        paths = [
            Path(parent) / name
            for parent, _, files in os.walk(directory, followlinks=True)
            for name in files
            if (Path(parent) / name).is_file()
            and (Path(parent) / name).stat().st_size > 0
        ]
        checked[component] = {}
        for target in targets:
            pattern = re.compile(rf"{re.escape(target)}(?![0-9a-f])")
            matching = [
                path
                for path in paths
                if pattern.search(str(path.relative_to(directory)))
            ]
            metadata = [
                path for path in matching if path.name.endswith((".dat", ".dat.zlib"))
            ]
            objects = [path for path in matching if path.suffix in {".co", ".hsaco"}]
            assert metadata and objects, (
                f"Incomplete {component} kernel data for {target}"
            )
            checked[component][target] = {
                "metadata_files": len(metadata),
                "code_objects": len(objects),
            }
    return {"blas_kernel_data": checked}


def main():
    """Reject missing target data before a long source compilation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--component", choices=("rocblas", "hipblaslt"))
    args = parser.parse_args()
    assert not Path("/dev/kfd").exists() and not Path("/dev/dri").exists()
    print(
        json.dumps(
            check_kernel_data(
                args.root, json.loads(args.profile.read_text()), args.component
            )
        )
    )


if __name__ == "__main__":
    main()
