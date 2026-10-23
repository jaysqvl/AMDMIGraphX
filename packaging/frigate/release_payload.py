"""Prepare checked runtime archives for GitHub's per-asset size limit."""

import argparse
import hashlib
import json
from pathlib import Path

ASSET_LIMIT = 2 * 1024**3
PART_BYTES = 1900 * 1024**2


def export_archive(archive: Path, output: Path, part_bytes=PART_BYTES) -> dict:
    """Preserve a whole-archive hash, splitting large payloads into ordered parts."""
    size = archive.stat().st_size
    if size <= 0 or not 0 < part_bytes < ASSET_LIMIT:
        raise ValueError("Use a nonempty archive and parts below GitHub's asset limit")
    result = {"file": archive.name, "bytes": size}
    whole = hashlib.sha256()
    if size <= part_bytes:
        with archive.open("rb") as stream:
            whole = hashlib.file_digest(stream, "sha256")
        archive.rename(output / archive.name)
    else:
        parts = []
        with archive.open("rb") as stream:
            remaining = size
            while remaining:
                name = f"{archive.name}.part{len(parts) + 1:03d}"
                count = min(remaining, part_bytes)
                checksum = hashlib.sha256()
                with (output / name).open("xb") as destination:
                    left = count
                    while left:
                        block = stream.read(min(left, 1024**2))
                        if not block:
                            raise ValueError("Archive changed while splitting")
                        destination.write(block)
                        whole.update(block)
                        checksum.update(block)
                        left -= len(block)
                parts.append(
                    {"file": name, "bytes": count, "sha256": checksum.hexdigest()}
                )
                remaining -= count
            if stream.read(1):
                raise ValueError("Archive changed while splitting")
        result["parts"] = parts
        archive.unlink()
    result["sha256"] = whole.hexdigest()
    return result


def main():
    """Attach the pushed OCI digest and real archive checksums to release metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    assert metadata["profile"] == args.profile
    inspected = json.loads(args.inspection.read_text())[0]
    matches = [
        ref
        for ref in inspected["RepoDigests"]
        if ref.startswith(args.image + "@sha256:")
    ]
    assert len(matches) == 1
    metadata["image"] = matches[0]
    args.output.mkdir()
    exported = export_archive(args.archive, args.output)
    metadata["archive"] = exported
    (args.output / f"{args.profile}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (args.output / f"{args.profile}.sha256").write_text(
        exported["sha256"] + "  " + exported["file"] + "\n"
    )
    if "parts" in exported:
        (args.output / f"{args.profile}.parts.sha256").write_text(
            "".join(
                part["sha256"] + "  " + part["file"] + "\n"
                for part in exported["parts"]
            )
        )


if __name__ == "__main__":
    main()
