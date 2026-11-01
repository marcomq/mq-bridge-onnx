#!/bin/sh
# Writes THIRD_PARTY_NOTICES for the ONNX Runtime the plugin links statically,
# and the licenses of the crates feature `asr` adds, read from cargo's registry.
# Usage: scripts/third_party_notices.sh 1.31.0
# The version is the `ms@X.Y.Z` in ort-sys's build/download/dist.tsv.
# Adding a crate to `asr`: add it to ASR_CRATES and rerun.
set -eu

VERSION=${1:?usage: third_party_notices.sh <onnxruntime version>}
SRC=https://raw.githubusercontent.com/microsoft/onnxruntime/v$VERSION
ROOT=$(dirname "$0")/..
OUT=$ROOT/THIRD_PARTY_NOTICES
ASR_CRATES="parakeet-rs ndarray matrixmultiply rawpointer realfft eyre indenter onig onig_sys"

{
    cat <<EOF
THIRD-PARTY NOTICES
===================

mq-bridge-onnx links ONNX Runtime $VERSION (https://github.com/microsoft/onnxruntime)
statically into its library. Its license and its third-party notices follow
unchanged, then the licenses of the crates the cargo feature asr adds,
including the Oniguruma C library. mq-bridge-onnx's own code is licensed MIT OR Apache-2.0, see
LICENSE-MIT and LICENSE-APACHE.

================================================================================
ONNX Runtime $VERSION: LICENSE
================================================================================

EOF
    curl -fsSL "$SRC/LICENSE"
    cat <<EOF

================================================================================
ONNX Runtime $VERSION: ThirdPartyNotices.txt
================================================================================

EOF
    curl -fsSL "$SRC/ThirdPartyNotices.txt"
    cargo metadata --format-version 1 --manifest-path "$ROOT/Cargo.toml" |
        ASR_CRATES=$ASR_CRATES python3 -c '
import json, os, sys

wanted = os.environ["ASR_CRATES"].split()
packages = {p["name"]: p for p in json.load(sys.stdin)["packages"] if p["name"] in wanted}

def section(title, path):
    rule = "=" * 80
    print(f"\n{rule}\n{title}\n{rule}\n")
    with open(path, encoding="utf-8") as f:
        print(f.read().rstrip())

for name in wanted:
    p = packages[name]
    folder = os.path.dirname(p["manifest_path"])
    file = next(f for f in ("LICENSE-MIT", "LICENSE", "LICENSE.md") if os.path.exists(os.path.join(folder, f)))
    version, license = p["version"], p["license"]
    section(f"{name} {version} ({license}, feature asr): {file}", os.path.join(folder, file))
    if name == "onig_sys":
        section("Oniguruma (bundled in onig_sys, feature asr): COPYING", os.path.join(folder, "oniguruma", "COPYING"))
'
} >"$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "wrote $OUT"
