# Pinned AMD runtimes for Frigate

This directory builds the AMD runtime separately from Frigate. Normal Frigate
updates reuse a runtime by immutable OCI digest. Rebuild this layer when changing
its source pins, SDK, compiler, target list, or build settings.

All public profiles are **experimental**. CI checks actual offline compiler
output, linking, Python imports, and CPU arithmetic. It does not initialize a GPU
or prove hardware support. Earlier Ryzen 7 5700G measurements used different
artifacts and do not qualify these broader public builds.

| Profile | Build targets | BLAS and compiler |
| --- | --- | --- |
| `rocm-10.1` | gfx1100, gfx1101, gfx1200, gfx1201 | AMD SDK rocBLAS and hipBLASLt, classic rocMLIR |
| `rocm-10.1-vega` | gfx900, gfx906, gfx90c | Source-built rocBLAS/Tensile, classic rocMLIR |
| `rocm-10.1-gfx90c` | gfx90c | The focused legacy recipe, source-built rocBLAS/Tensile |
| `rocm-7.2.4` | Gary's ten-target set in `profiles.json` | ROCm 7.2.4 libraries plus gfx900 kernel data, source-built MIGraphX/rocMLIR |

Every profile uses CPython 3.11 and the portable `x86-64-v2` CPU baseline.
The Vega profiles disable hipBLASLt and Composable Kernel. The modern profile
uses the SDK's BLAS and disables CK, which that SDK does not provide. The small
`native101/migraphx-rocblas-only.patch` fixes building without hipBLASLt.
`retarget_logic.py` preserves the original gfx900 tuning and adds its gfx90c
architecture variant; it does not rewrite compiled ISA tags.

## Build and release

Use a dedicated Linux amd64 Docker host, Python 3.11+, at least 16 GiB RAM, and
ample disk for the LLVM build. The defaults are four compile jobs and a 12 GiB
container limit. Source downloads are the only container stage with networking.
Compilation and verification use `--network=none` and expose no GPU devices.

```sh
python3 packaging/frigate/runtime.py plan --profile rocm-10.1-vega
python3 packaging/frigate/runtime.py all --profile rocm-10.1-vega --archive
```

`--workspace`, `--receipts`, `--jobs`, `--memory`, and `--cpuset` are explicit
options. An existing SSH alias can select a remote Docker worker with
`--ssh-host`; choose a dedicated absolute workspace on that worker. No service
is deployed. Separate `fetch`, `build`, `compiler`, `verify`, and `package` modes
use the same receipt. A changed recipe requires a new empty workspace, rather
than silently reusing objects built with different settings.

The **Frigate AMD runtimes** workflow runs cheap checks for changes here.
It verifies the exported NUMA/OpenMP libraries against Debian 12's symbol ABI
before starting the source builds. HIPRTC headers are packaged separately from
glibc's runtime libraries; the application OS is not upgraded.
A small fixture also packages root-owned files as an unprivileged Linux host,
checks its OCI/archive output, and removes its recorded containers and image.
The real runtime checks remain mandatory; this fixture contains no AMD binaries.
To publish all four profiles, dispatch it with a new `runtime-vVERSION` tag, or
push such a tag. It reserves a draft GitHub Release, builds each profile on a
disposable GitHub runner, pushes the checked runtime to GHCR, and uploads the
archive, checksum, and profile JSON. The release is published as a prerelease
only after all four profiles succeed. A failure keeps the release in draft.
Large runtime payloads go directly to Releases; only small build logs use
Actions artifact storage.

The profile JSON records the exact source commit, recipe and lock hashes,
compiler targets, verification results, archive checksum, and OCI digest.
Fresh builds reproduce pinned inputs and settings; identical OCI bytes across
builds are not promised. Set a newly created GHCR package public once and verify
anonymous access before advertising it as a public image.

Successful build containers are removed by recorded full ID. Failed runs retain
logs and can finish container cleanup with `runtime.py cleanup` using the same
arguments. Cleanup checks ownership and never stops a running container or
prunes shared images, volumes, or workspaces.

## Ownership and upstream sync

The source locks use these independently owned forks:

- [AMDMIGraphX](https://github.com/jaysqvl/AMDMIGraphX)
- [rocMLIR](https://github.com/jaysqvl/rocMLIR)
- [rocm-libraries](https://github.com/jaysqvl/rocm-libraries)

`develop` remains the upstream mirror. This fork's `main` adds only the packaging
directory and its workflow. Merge upstream `develop` into `main` to sync source;
updating the mirror does not change a locked runtime recipe. Advance each source
commit, archive size and SHA256 together in its lock, rebuild, and review the new
compiler/linking evidence before updating Frigate's runtime digest.

The current runtime uses direct MIGraphX Python bindings. It does not need a
custom ONNX Runtime wheel or a separate ONNX Runtime source fork.
