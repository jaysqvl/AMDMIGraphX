"""Exercise root-owned output packaging from a non-root disposable Linux worker."""

import argparse
import json
import os
import stat
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import DEBIAN, ROOT, Runtime


def main():
    """Check filesystem ownership, OCI export, archive contents and exact cleanup."""
    if os.geteuid() == 0:
        raise RuntimeError("Run the packaging fixture as an unprivileged Docker user")
    os.umask(0o077)
    fixture = ROOT / "tmp/packaging-fixture"
    args = argparse.Namespace(
        profile="rocm-10.1-gfx90c",
        workspace=fixture / "work",
        receipts=fixture / "receipts",
        jobs=1,
        memory="512m",
        cpuset=None,
        ssh_host=None,
        socket="unix:///var/run/docker.sock",
        tag=None,
        archive=True,
    )
    runtime = Runtime(args)
    image = None
    try:
        runtime.prepare()
        runtime.run(
            "fixture",
            DEBIAN,
            "mkdir -p /runtime/usr/share\n"
            "chown 0:0 /runtime/usr/share\n"
            "chmod 0755 /runtime/usr/share\n"
            'printf "%s\\n" "$FIXTURE_OWNER" > /runtime/usr/share/fixture.txt\n'
            "mkdir -p /runtime/opt/fixture/share/frigate\n"
            'printf "%s\\n" "$FIXTURE_OWNER" > /runtime/opt/fixture/share/frigate/source-lock.json\n'
            "chmod 0600 /runtime/opt/fixture/share/frigate/source-lock.json\n"
            "chmod 0700 /runtime/opt/fixture/share/frigate\n",
            [(runtime.remote + "/runtime", "/runtime", False)],
            env=("FIXTURE_OWNER=" + runtime.journal["owner"],),
        )
        protected = Path(runtime.remote) / "runtime/usr/share"
        public_recipes = Path(runtime.remote) / "runtime/opt/fixture/share/frigate"
        assert protected.stat().st_uid == 0
        assert not os.access(protected, os.W_OK)
        assert not os.access(public_recipes / "source-lock.json", os.R_OK)
        # This fixture deliberately has no AMD binaries or compiler evidence.
        # Bypass only the prerequisite gate while exercising real packaging.
        # Its verification object cannot pass the Frigate runtime importer.
        runtime.journal["verification"] = {"fixture_only": True}
        with patch.object(runtime, "require_passed") as gate:
            runtime.package()
            gate.assert_called_once_with(["source", "compiler", "verify"])
        image = runtime.journal["candidate"]["id"]
        manifest = json.loads((protected / "frigate-rocm/runtime.json").read_text())
        assert manifest["verification"] == {"fixture_only": True}
        assert stat.S_IMODE(public_recipes.stat().st_mode) == 0o755
        assert (
            stat.S_IMODE((public_recipes / "source-lock.json").stat().st_mode) == 0o644
        )
        runtime.run(
            "fixture-consumer",
            DEBIAN,
            'test "$(id -u)" = 1000\n'
            "test -r /payload/usr/share/frigate-rocm/runtime.json\n"
            'test "$(cat /payload/usr/share/fixture.txt)" = "$FIXTURE_OWNER"\n'
            'test "$(cat /payload/opt/fixture/share/frigate/source-lock.json)" = "$FIXTURE_OWNER"\n',
            [(runtime.remote + "/runtime", "/payload", True)],
            user="1000:1000",
            env=("FIXTURE_OWNER=" + runtime.journal["owner"],),
        )
        archive = Path(runtime.remote) / "output" / (runtime.name + ".tar.gz")
        with tarfile.open(archive, "r:gz") as exported:
            assert exported.getmember(".").mode == 0o755
            metadata = exported.extractfile("./usr/share/frigate-rocm/runtime.json")
            assert metadata is not None and json.load(metadata) == manifest
            payload = exported.extractfile("./usr/share/fixture.txt")
            assert payload is not None
            assert payload.read().decode().strip() == runtime.journal["owner"]
            source_lock = exported.extractfile(
                "./opt/fixture/share/frigate/source-lock.json"
            )
            assert source_lock is not None
            assert source_lock.read().decode().strip() == runtime.journal["owner"]
            assert exported.getmember("./opt/fixture/share/frigate").mode == 0o755
            assert (
                exported.getmember("./opt/fixture/share/frigate/source-lock.json").mode
                == 0o644
            )
            assert all(
                member.uid == member.gid == member.mtime == 0 for member in exported
            )
        assert not os.access(protected, os.W_OK)
        print(json.dumps({"root_owned_packaging": "passed", "image_id": image}))
    finally:
        runtime.cleanup()
        image = image or runtime.journal.get("candidate", {}).get("id")
        if image:
            references = runtime.text(
                [
                    "container",
                    "ls",
                    "-aq",
                    "--no-trunc",
                    "--filter",
                    "ancestor=" + image,
                ]
            )
            if references:
                raise RuntimeError("A container still protects the fixture image")
            runtime.docker(["image", "rm", image])
            print(json.dumps({"removed_fixture_image": image}))


if __name__ == "__main__":
    main()
