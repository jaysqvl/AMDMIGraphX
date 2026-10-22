#!/usr/bin/env python3
"""Build pinned AMD runtimes in device-free Linux containers."""

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SDK724 = "rocm/dev-ubuntu-22.04@sha256:a50a8547101ac9e7c35848ff609a56335214b789b468c9eb05dcd006a4a38c25"
DEBIAN = (
    "debian@sha256:a4672c0cb26fbdde88e38fa2dfb6c681942306680e41e4378b28770b6e79ee91"
)
PROFILES = json.loads((HERE / "profiles.json").read_text())


def validate_profile(name: str) -> dict:
    """Reject inconsistent target settings before contacting a Docker host."""
    profile = PROFILES[name]
    if profile["kind"] not in {"724", "101"}:
        raise ValueError("Unknown runtime recipe kind")
    if not profile["gpu_targets"] or len(set(profile["gpu_targets"])) != len(
        profile["gpu_targets"]
    ):
        raise ValueError("A profile needs distinct explicit GPU targets")
    if any(not re.fullmatch(r"gfx[0-9a-f]+", arch) for arch in profile["gpu_targets"]):
        raise ValueError("Invalid AMD GPU target")
    if profile["legacy_rocblas"] and (
        profile["hipblaslt"] or profile["composable_kernel"]
    ):
        raise ValueError("The legacy rocBLAS profile disables CK and hipBLASLt")
    if profile["python_abi"] != "cp311":
        raise ValueError("The current source locks require CPython 3.11")
    return profile


def now() -> str:
    """Return an unambiguous receipt timestamp."""
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    """Hash a build input without loading large archives into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def memory_bytes(value: str) -> int:
    """Accept explicit integer byte or binary K/M/G limits."""
    match = re.fullmatch(r"([1-9][0-9]*)([kmg]?)", value.lower())
    if not match:
        raise ValueError("Use an integer memory limit such as 12g")
    return int(match[1]) * 1024 ** {"": 0, "k": 1, "m": 2, "g": 3}[match[2]]


class Runtime:
    """Keep source inputs, bounded stages, and artifact identities together."""

    def __init__(self, args):
        self.args = args
        self.configuration = validate_profile(args.profile)
        self.profile = self.configuration["kind"]
        self.name = args.profile
        self.root = str(args.workspace.resolve())
        if self.root in {"/", "/mnt", "/opt", "/tmp", "/var", "/home"}:
            raise ValueError("Use a dedicated build workspace")
        if args.jobs < 1:
            raise ValueError("Compiler jobs must be positive")
        memory_bytes(args.memory)
        self.remote = self.root + "/" + self.name
        self.recipe = HERE / self.configuration["recipe"]
        self.lock = json.loads((self.recipe / "source-lock.json").read_text())
        self.directory = args.receipts.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / (self.name + ".json")
        self.guard = self.path.with_suffix(".lock").open("a")
        fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.journal = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "schema": 1,
                "owner": "amd-runtime-" + os.urandom(8).hex(),
                "profile": self.name,
                "workspace": self.root,
                "ssh_host": args.ssh_host,
                "created_at": now(),
                "containers": [],
                "images": [],
                "stages": {},
            }
        )
        if (
            self.journal["workspace"],
            self.journal["ssh_host"],
            self.journal["profile"],
        ) != (self.root, args.ssh_host, self.name):
            raise ValueError("Receipt belongs to a different build")
        info = self.json(["info", "--format", "{{json .}}"])
        if info["OSType"] != "linux" or info["Architecture"] not in {"x86_64", "amd64"}:
            raise ValueError("Requires a Linux amd64 Docker host")
        if self.journal.get("engine", info["ID"]) != info["ID"]:
            raise ValueError("Docker engine changed")
        if not args.cpuset:
            args.cpuset = ",".join(str(cpu) for cpu in range(info["NCPU"]))
        if not re.fullmatch(r"[0-9,-]+", args.cpuset):
            raise ValueError("Invalid CPU affinity")
        self.journal["engine"] = info["ID"]
        self.journal["resource_limits"] = {
            "cpuset": args.cpuset,
            "jobs": args.jobs,
            "memory": args.memory,
        }
        self.journal["repository_commit"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.save()

    def environment(self) -> tuple[str, ...]:
        """Pass the selected profile explicitly to every isolated build stage."""
        targets = ";".join(self.configuration["gpu_targets"])
        return (
            "GPU_TARGETS=" + targets,
            "OWNED_GPU_ARCHS=" + targets.replace(";", " "),
            "CPU_ARCH=" + self.configuration["cpu_arch"],
            "BUILD_LEGACY_ROCBLAS="
            + ("1" if self.configuration["legacy_rocblas"] else "0"),
            "USE_HIPBLASLT=" + ("ON" if self.configuration["hipblaslt"] else "OFF"),
            "USE_COMPOSABLEKERNEL="
            + ("ON" if self.configuration["composable_kernel"] else "OFF"),
        )

    def prepare(self):
        """Stage only runtime inputs into a marked, exclusively owned workspace."""
        existing = set(self.text(["container", "ls", "-aq", "--no-trunc"]).split())
        identities = [
            row["id"] for row in self.journal["containers"] if row["id"] in existing
        ]
        if identities:
            for container in self.json(["inspect", *identities]):
                if container["State"]["Status"] not in {"exited", "dead"}:
                    raise RuntimeError("An owned build container is still active")
        remote = shlex.quote(self.remote)
        marker = shlex.quote(self.remote + "/.runtime-owner")
        owner = shlex.quote(self.journal["owner"])
        self.host(
            [
                "bash",
                "-c",
                (
                    f"set -eu\ntest ! -L {remote}\n"
                    f'if [ -d {remote} ] && [ ! -f {marker} ] && [ -n "$(ls -A {remote})" ]; then '
                    "printf '%s\\n' 'Refusing an existing unowned build directory' >&2; exit 1; fi\n"
                    f"mkdir -p {remote}\n"
                    f'if [ -f {marker} ]; then test "$(cat {marker})" = {owner}; '
                    f'else printf "%s\\n" {owner} > {marker}; fi'
                ),
            ]
        )
        directories = (
            "inputs",
            "recipes",
            "work",
            "prefix",
            "runtime",
            "binding-work",
            "metadata",
            "output",
        )
        self.host(["mkdir", "-p", *(self.remote + "/" + name for name in directories)])
        entries = [
            (path, "recipes/" + path.name)
            for path in self.recipe.iterdir()
            if path.is_file()
            and not path.name.startswith("test_")
            and path.suffix in {".py", ".sh", ".json", ".txt", ".patch", ".sha256"}
        ]
        entries += [
            (HERE / "common/source_inputs.py", "recipes/source_inputs.py"),
            (HERE / "common/runtime-os-lock.json", "recipes/runtime-os-lock.json"),
            (HERE / "verify_runtime.py", "recipes/verify_runtime.py"),
            (HERE / "Dockerfile", "Dockerfile"),
            (HERE / ".dockerignore", ".dockerignore"),
        ]
        entries += [
            (HERE / "owned" / name, "recipes/" + name)
            for name in (
                "CMakeLists.compiler.txt",
                "compiler-probe.cpp",
                "build_compiler_probe.sh",
                "gfx90c_code.py",
            )
        ]
        hashes = {str(path.relative_to(HERE)): digest(path) for path, _ in entries}
        recipe_hash = hashlib.sha256(
            json.dumps(
                {"files": hashes, "profile": self.configuration}, sort_keys=True
            ).encode()
        ).hexdigest()
        previous = self.journal.get("runtime_recipe_sha256")
        if previous and previous != recipe_hash and self.journal["stages"]:
            raise ValueError("Runtime inputs changed; use a new empty build workspace")
        self.upload(entries)
        self.write_remote_json("recipes/profile.json", self.configuration)
        self.journal.update(
            {"runtime_recipe_sha256": recipe_hash, "input_files": hashes}
        )
        self.save()

    def write_remote_json(self, name: str, value: dict):
        """Keep generated metadata under the repository's ignored tmp directory."""
        temporary = ROOT / "tmp/public-builds/metadata" / self.journal["owner"]
        temporary.mkdir(parents=True, exist_ok=True)
        path = temporary / Path(name).name
        path.write_text(json.dumps(value, indent=2) + "\n")
        self.upload([(path, name)])

    def save(self):
        """Atomically persist ownership before the next mutation."""
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.journal, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def host(self, arguments, **kwargs):
        """Preserve argument boundaries when crossing SSH."""
        command = list(arguments)
        if self.args.ssh_host:
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                self.args.ssh_host,
                shlex.join(command),
            ]
        return subprocess.run(command, check=True, **kwargs)

    def docker(self, arguments, **kwargs):
        """Address the intended daemon explicitly."""
        return self.host(["docker", "-H", self.args.socket, *arguments], **kwargs)

    def text(self, arguments):
        """Read a small Docker result."""
        return self.docker(arguments, capture_output=True, text=True).stdout.strip()

    def json(self, arguments):
        """Read structured Docker state."""
        return json.loads(self.text(arguments))

    def upload(self, entries):
        """Stage versioned recipes without requiring host Python."""
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as output:
            for source, name in entries:
                output.add(source, arcname=name)
        self.host(["tar", "-xf", "-", "-C", self.remote], input=archive.getvalue())

    def image(self, reference):
        """Resolve an immutable Linux amd64 image."""
        owned = reference == self.journal.get("candidate", {}).get("id")
        if "@sha256:" not in reference and not owned:
            raise ValueError("Base images must use immutable manifest digests")
        image = self.json(["image", "inspect", reference])[0]
        if image["Architecture"] != "amd64" or image["Os"] != "linux":
            raise ValueError("Wrong base image architecture")
        return image["Id"]

    def pull(self, reference):
        """Acquire a missing pinned base; existing shared bases are not owned."""
        try:
            return self.image(reference)
        except subprocess.CalledProcessError:
            with (self.directory / f"pull-{reference.rsplit(':', 1)[1][:12]}.log").open(
                "w"
            ) as log:
                self.docker(
                    ["pull", "--platform", "linux/amd64", reference],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            identity = self.image(reference)
            self.journal["images"].append(
                {
                    "id": identity,
                    "reference": reference,
                    "purpose": "acquired pinned base",
                }
            )
            self.save()
            return identity

    def run(
        self,
        stage,
        image,
        script,
        mounts=(),
        network="none",
        user=None,
        env=(),
        copies=(),
    ):
        """Require actual termination and a zero exit, even after SSH detaches."""
        image_id = self.pull(image)
        name = self.journal["owner"] + "-" + stage + "-" + os.urandom(3).hex()
        arguments = [
            "create",
            "--name",
            name,
            "--label",
            "io.jaysqvl.amd-runtime=" + self.journal["owner"],
            "--network",
            network,
            "--cpuset-cpus",
            self.args.cpuset,
            "--cpus",
            str(self.args.jobs),
            "--cpu-shares",
            "256",
            "--memory",
            self.args.memory,
            "--memory-swap",
            self.args.memory,
            "--pids-limit",
            "2048",
            "--ulimit",
            "core=0",
            "--no-healthcheck",
            "--env",
            "BUILD_JOBS=" + str(self.args.jobs),
            "--entrypoint",
            "/bin/bash",
        ]
        if user:
            arguments += ["--user", user]
        for variable in (*self.environment(), *env):
            arguments += ["--env", variable]
        for source, target, readonly in mounts:
            if not source.startswith(self.remote + "/") or ".." in Path(source).parts:
                raise ValueError("Stage mounts must stay inside the owned workspace")
            arguments += [
                "--mount",
                f"type=bind,source={source},target={target}"
                + (",readonly" if readonly else ""),
            ]
        safety = "set -euo pipefail\ntest ! -e /dev/kfd\ntest ! -e /dev/dri\n"
        cid = self.text([*arguments, image_id, "-c", safety + script])
        if not re.fullmatch("[a-f0-9]{64}", cid):
            raise ValueError("Docker did not return a full container ID")
        obj = self.json(["inspect", cid])[0]
        self.journal["containers"].append(
            {"id": cid, "image": obj["Image"], "purpose": stage}
        )
        self.save()
        host = obj["HostConfig"]
        if host["Privileged"] or host.get("Devices") or host.get("DeviceRequests"):
            raise ValueError("Device isolation failed")
        if host["CpusetCpus"] != self.args.cpuset or host["NetworkMode"] != network:
            raise ValueError("Stage resource configuration changed")
        if (
            host["Memory"] != memory_bytes(self.args.memory)
            or host["NanoCpus"] != self.args.jobs * 1_000_000_000
            or host["CpuShares"] != 256
            or host["PidsLimit"] != 2048
        ):
            raise ValueError("Stage CPU, memory or process limit changed")
        self.journal["stages"][stage] = {
            "id": cid,
            "started_at": now(),
            "network": network,
            "gpu_devices_exposed": False,
            "state": "running",
            "runtime_recipe_sha256": self.journal["runtime_recipe_sha256"],
        }
        self.save()
        if copies:
            archive = io.BytesIO()

            def owner(member):
                member.uid = member.gid = 1000
                member.uname = member.gname = ""
                return member

            with tarfile.open(fileobj=archive, mode="w") as output:
                for source, name in copies:
                    output.add(source, arcname=name, filter=owner)
            self.docker(["cp", "-", cid + ":/"], input=archive.getvalue())
        print(
            json.dumps({"stage": stage, "container": cid, "state": "running"}),
            flush=True,
        )
        log_path = self.directory / f"{self.name}-{stage}-{cid[:12]}.log"
        self.journal["stages"][stage]["log"] = log_path.name
        self.save()
        with log_path.open("w") as log:
            try:
                self.docker(
                    ["start", "--attach", cid], stdout=log, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError:
                pass
        # Interrupted SSH can leave a running container with ExitCode=0.
        state = self.json(["inspect", cid])[0]["State"]
        self.journal["stages"][stage].update({"finished_at": now(), "state": state})
        self.journal["containers"][-1].update({"state": state, "log": log_path.name})
        self.save()
        if state["Status"] != "exited" or state["ExitCode"] != 0 or state["OOMKilled"]:
            raise RuntimeError(
                f"{stage} did not finish successfully; see {self.directory}"
            )
        print(
            json.dumps({"stage": stage, "state": "passed", "container": cid}),
            flush=True,
        )

    def compiler(self):
        """Compile through the owned MLIR API without using a GPU or a driver."""
        if self.profile == "724":
            image = SDK724
            setup = """python3 /recipes/source_inputs.py verify --lock /recipes/source-lock.json --root /inputs --group sdk --group source --group tool
    dpkg --unpack /inputs/sdk/*.deb
    dpkg --configure --pending
    """
        else:
            image = DEBIAN
            setup = """(cd /inputs && sha256sum -c /recipes/builder-inputs.sha256)
    apt-get -y --no-install-recommends -o Dir::Etc::sourcelist=/dev/null -o Dir::Etc::sourceparts=- install /inputs/builder/*.deb
    python3 /recipes/source_inputs.py verify --lock /recipes/source-lock.json --root /inputs
    """
        self.run(
            "compiler",
            image,
            setup + "bash /recipes/build_compiler_probe.sh\n",
            self.mounts(),
            env=("OWNED_PROFILE=" + self.profile,),
        )

    def mounts(self):
        """Map only owned inputs, output, and compiler caches."""
        return [
            (self.remote + "/" + name, target, readonly)
            for name, target, readonly in (
                ("inputs", "/inputs", True),
                ("recipes", "/recipes", True),
                ("work", "/work", False),
                ("prefix", "/opt/owned-deps", False),
                ("runtime", "/runtime", False),
            )
        ]

    def fetch(self):
        """Verify cached inputs and fetch absent files before entering offline mode."""
        script = (
            "python3 /recipes/source_inputs.py fetch --lock /recipes/source-lock.json --root /inputs\n"
            "python3 /recipes/source_inputs.py fetch --lock /recipes/runtime-os-lock.json --root /inputs\n"
        )
        self.run(
            "fetch",
            SDK724,
            script,
            [
                (self.remote + "/inputs", "/inputs", False),
                (self.remote + "/recipes", "/recipes", True),
            ],
            network="bridge",
        )

    def build(self):
        """Build both variants through the same stage runner."""
        if self.profile == "724":
            script = """python3 /recipes/source_inputs.py verify --lock /recipes/source-lock.json --root /inputs --group sdk --group source --group tool --group legacy-kernels
    dpkg --unpack /inputs/sdk/*.deb
    dpkg --configure --pending
    BUILD_JOBS=$BUILD_JOBS nice -n 10 bash /recipes/build_source.sh
    rm -rf /runtime/opt/rocm-7.2.4
    python3 /recipes/assemble_runtime.py
    cp -a /runtime/opt/rocm-7.2.4/share/frigate/. /metadata/
    """
            self.run(
                "source",
                SDK724,
                script,
                self.mounts() + [(self.remote + "/metadata", "/metadata", False)],
            )
            script = """(cd /inputs && sha256sum -c /metadata/binding-inputs.sha256)
    apt-get -y --no-install-recommends -o Dir::Etc::sourcelist=/dev/null -o Dir::Etc::sourceparts=- install /inputs/binding/*.deb
    ln -s rocm-7.2.4 /opt/rocm
    BUILD_JOBS=$BUILD_JOBS nice -n 10 bash /recipes/build_binding.sh
    python3 /recipes/source_inputs.py verify --lock /recipes/runtime-os-lock.json --root /inputs
    bash /recipes/assemble_os.sh
    """
            mounts = [
                mount
                for mount in self.mounts()
                if mount[1] not in {"/work", "/opt/owned-deps"}
            ]
            mounts += [
                (self.remote + "/binding-work", "/work", False),
                (self.remote + "/metadata", "/metadata", True),
                (self.remote + "/runtime/opt/rocm-7.2.4", "/opt/rocm-7.2.4", False),
            ]
            self.run("binding", DEBIAN, script, mounts)
        else:
            # Minimal Debian has no Python until the verified bootstrap is installed.
            script = """(cd /inputs && sha256sum -c /recipes/builder-inputs.sha256)
    apt-get -y --no-install-recommends -o Dir::Etc::sourcelist=/dev/null -o Dir::Etc::sourceparts=- install /inputs/builder/*.deb
    python3 /recipes/source_inputs.py verify --lock /recipes/source-lock.json --root /inputs
    BUILD_JOBS=$BUILD_JOBS nice -n 10 bash /recipes/build_source.sh
    python3 /recipes/assemble_runtime.py
    """
            self.run("source", DEBIAN, script, self.mounts())

    def require_passed(self, stages):
        """Reject publication after a failed, interrupted, or different recipe run."""
        for stage in stages:
            result = self.journal["stages"].get(stage, {})
            state = result.get("state", {})
            if (
                not isinstance(state, dict)
                or state.get("Status") != "exited"
                or state.get("ExitCode") != 0
                or state.get("OOMKilled")
            ):
                raise RuntimeError(f"{stage} must finish successfully first")
            if (
                result.get("runtime_recipe_sha256")
                != self.journal["runtime_recipe_sha256"]
            ):
                raise RuntimeError(f"{stage} belongs to a different recipe")

    def verify(self):
        """Check the packaged closure and CPU reference arithmetic without HSA."""
        self.require_passed(
            ["source", "compiler"] + (["binding"] if self.profile == "724" else [])
        )
        group = "binding" if self.profile == "724" else "builder"
        checksum = (
            "/metadata/binding-inputs.sha256"
            if self.profile == "724"
            else "/recipes/builder-inputs.sha256"
        )
        script = f"""(cd /inputs && sha256sum -c {checksum})
apt-get -y --no-install-recommends -o Dir::Etc::sourcelist=/dev/null -o Dir::Etc::sourceparts=- install /inputs/{group}/*.deb
cp -a /runtime/opt/. /opt/
cp -a /runtime/usr/. /usr/
ln -sfn rocm-{self.configuration["rocm"]} /opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/migraphx/lib:/opt/rocm/lib/llvm/lib
export PYTHONPATH=/opt/rocm/lib
python3 /recipes/verify_runtime.py --profile /recipes/profile.json --output /output/verification.json
"""
        mounts = [
            (self.remote + "/" + name, target, True)
            for name, target in (
                ("inputs", "/inputs"),
                ("recipes", "/recipes"),
                ("runtime", "/runtime"),
                ("work/compiler-check", "/compiler-check"),
                ("metadata", "/metadata"),
            )
        ] + [(self.remote + "/output", "/output", False)]
        self.run(
            "verify",
            DEBIAN,
            script,
            mounts,
            env=("HSA_ENABLE_SDMA=0", "OMP_NUM_THREADS=2"),
        )
        result = self.host(
            ["cat", self.remote + "/output/verification.json"],
            capture_output=True,
            text=True,
        )
        self.journal["verification"] = json.loads(result.stdout)
        self.save()

    def package(self):
        """Export only a checked runtime, independently of any Frigate image."""
        self.require_passed(
            ["source", "compiler", "verify"]
            + (["binding"] if self.profile == "724" else [])
        )
        manifest = {
            "schema": 1,
            "profile": self.name,
            "configuration": self.configuration,
            "source_repository": "https://github.com/jaysqvl/AMDMIGraphX",
            "source_commit": self.journal["repository_commit"],
            "runtime_recipe_sha256": self.journal["runtime_recipe_sha256"],
            "source_lock_sha256": digest(self.recipe / "source-lock.json"),
            "verification": self.journal["verification"],
            "gpu_tested": False,
        }
        self.host(["mkdir", "-p", self.remote + "/runtime/usr/share/frigate-rocm"])
        self.write_remote_json("runtime/usr/share/frigate-rocm/runtime.json", manifest)
        self.write_remote_json("output/runtime.json", manifest)
        self.upload([(HERE / "Dockerfile", "Dockerfile")])
        tag = self.args.tag or "amd-runtime:" + self.name + "-" + self.journal["owner"]
        with (self.directory / (self.name + "-package.log")).open("w") as log:
            self.docker(
                [
                    "build",
                    "--platform",
                    "linux/amd64",
                    "--network",
                    "none",
                    "--tag",
                    tag,
                    "--file",
                    self.remote + "/Dockerfile",
                    "--label",
                    "org.opencontainers.image.source=https://github.com/jaysqvl/AMDMIGraphX",
                    "--label",
                    "org.opencontainers.image.revision=" + manifest["source_commit"],
                    "--label",
                    "io.jaysqvl.amd-runtime.profile=" + self.name,
                    "--label",
                    "io.jaysqvl.amd-runtime.recipe="
                    + manifest["runtime_recipe_sha256"],
                    self.remote,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        image = self.json(["image", "inspect", tag])[0]
        self.journal["candidate"] = {"id": image["Id"], "tag": tag, "gpu_tested": False}
        self.journal["images"].append(
            {"id": image["Id"], "purpose": "verified runtime", "tag": tag}
        )
        self.save()
        manifest["local_image_id"] = image["Id"]
        (self.directory / (self.name + "-runtime.json")).write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if self.args.archive:
            self.host(
                [
                    "tar",
                    "--sort=name",
                    "--mtime=@0",
                    "--owner=0",
                    "--group=0",
                    "--numeric-owner",
                    "-czf",
                    self.remote + "/output/" + self.name + ".tar.gz",
                    "-C",
                    self.remote + "/runtime",
                    ".",
                ]
            )
        print(json.dumps(self.journal["candidate"]), flush=True)

    def cleanup(self):
        """Remove only recorded, stopped build containers on the same engine."""
        existing = set(self.text(["container", "ls", "-aq", "--no-trunc"]).split())
        for record in self.journal["containers"]:
            cid = record["id"]
            if cid not in existing:
                continue
            obj = self.json(["inspect", cid])[0]
            if (
                obj["Config"]["Labels"].get("io.jaysqvl.amd-runtime")
                != self.journal["owner"]
            ):
                raise RuntimeError("Container ownership changed")
            if obj["State"]["Status"] not in {"exited", "dead", "created"}:
                raise RuntimeError(
                    "Build container is still running; cleanup will not stop it"
                )
            self.docker(["rm", cid], capture_output=True)
            record["removed_at"] = now()
            self.save()


def main():
    """Use the same explicit profile interface locally and in public CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "plan",
            "prepare",
            "fetch",
            "build",
            "compiler",
            "verify",
            "package",
            "all",
            "cleanup",
            "status",
        ),
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT / "tmp/runtime")
    parser.add_argument("--receipts", type=Path, default=ROOT / "tmp/receipts")
    parser.add_argument("--ssh-host")
    parser.add_argument("--socket", default="unix:///var/run/docker.sock")
    parser.add_argument("--cpuset")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--memory", default="12g")
    parser.add_argument("--tag")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also export a normalized runtime tar archive",
    )
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps(validate_profile(args.profile), indent=2))
        return
    if args.mode == "status":
        print((args.receipts / (args.profile + ".json")).read_text())
        return
    os.umask(0o077)
    runtime = Runtime(args)
    if args.mode == "cleanup":
        runtime.cleanup()
        return
    runtime.prepare()
    for mode in ("fetch", "build", "compiler", "verify", "package"):
        if args.mode in {mode, "all"}:
            getattr(runtime, mode)()
            runtime.cleanup()


if __name__ == "__main__":
    main()
