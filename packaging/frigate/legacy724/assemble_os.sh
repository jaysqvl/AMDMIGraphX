#!/bin/bash
# Extract only the pinned runtime libraries and HIPRTC development files.
set -euo pipefail
test ! -e /dev/kfd
test ! -e /dev/dri
mkdir -p /runtime/opt /runtime/usr/lib/x86_64-linux-gnu /runtime/usr/lib/gcc/x86_64-linux-gnu
mkdir -p /runtime/opt/rocm-7.2.4/share/frigate
mkdir -p /os-packages
for package in /inputs/runtime-os/*.deb; do
    dpkg-deb -x "$package" /os-packages
done
cp -a /os-packages/usr/include /runtime/usr/
cp -a /os-packages/usr/lib/gcc/x86_64-linux-gnu/14 /runtime/usr/lib/gcc/x86_64-linux-gnu/
cp -a /os-packages/usr/lib/x86_64-linux-gnu/libnuma.so.* /runtime/usr/lib/x86_64-linux-gnu/
cp -a /os-packages/usr/lib/llvm-19/lib/libomp.so* /runtime/usr/lib/x86_64-linux-gnu/
mkdir -p /runtime/usr/share/doc
cp -a /os-packages/usr/share/doc/. /runtime/usr/share/doc/
cp /recipes/runtime-os-lock.json /runtime/opt/rocm-7.2.4/share/frigate/
cp /metadata/source-build.json /runtime/opt/rocm-7.2.4/share/frigate/
# Pin the Bookworm builds; the Trixie variants require a newer glibc.
echo 'bb2bb9250f348587ea3b7594a0a22055312134e55d065434d1f7e22e99ebf1d5  /runtime/usr/lib/x86_64-linux-gnu/libnuma.so.1.0.0' | sha256sum -c -
echo 'a325b385161f49ce5593f1f341fe30c6f9c95a22502855ad7c3b1a83c93397bd  /runtime/usr/lib/x86_64-linux-gnu/libomp.so.5' | sha256sum -c -
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/runtime')
base = root / 'opt/rocm-7.2.4/share/frigate'
manifest = json.loads((base / 'source-build.json').read_text())
binding = Path('/opt/rocm-7.2.4/lib/migraphx.cpython-311-x86_64-linux-gnu.so')
manifest['python_binding_sha256'] = hashlib.sha256(binding.read_bytes()).hexdigest()
manifest['runtime_os_lock_sha256'] = hashlib.sha256((base / 'runtime-os-lock.json').read_bytes()).hexdigest()
(base / 'source-build.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
