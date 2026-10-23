"""Check the real AMD device-package layout in recorded, device-free containers."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import ROOT, Runtime

PYTHON = (
    "python@sha256:8fb0b110b022c70cbe09d3ff1fb313654f31f9c814abb0d48180af56d906a199"
)


def main():
    """Reuse source-build expansion and check data without host GPU exposure."""
    os.umask(0o077)
    fixture = ROOT / "tmp/sdk-fixture"
    runtime = Runtime(
        argparse.Namespace(
            profile="rocm-10.1",
            workspace=fixture / "work",
            receipts=fixture / "receipts",
            jobs=2,
            memory="4g",
            cpuset=None,
            ssh_host=None,
            socket="unix:///var/run/docker.sock",
            tag=None,
            archive=False,
        )
    )
    try:
        runtime.prepare()
        runtime.run(
            "sdk-inputs",
            PYTHON,
            "python3 /recipes/source_inputs.py fetch --lock /recipes/source-lock.json "
            "--root /inputs --group sdk --group device\n",
            [
                (runtime.remote + "/inputs", "/inputs", False),
                (runtime.remote + "/recipes", "/recipes", True),
            ],
            network="bridge",
        )
        runtime.run(
            "sdk-layout",
            PYTHON,
            "python3 /recipes/expand_sdk.py\n"
            "python3 /recipes/kernel_data.py --root /opt/owned-deps/sdk/_rocm_sdk_devel "
            "--profile /recipes/profile.json\n",
            runtime.mounts(),
            env=("HSA_ENABLE_SDMA=0",),
        )
        stage = runtime.journal["stages"]["sdk-layout"]
        print((runtime.directory / stage["log"]).read_text())
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    main()
