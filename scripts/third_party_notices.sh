#!/bin/sh
# Writes THIRD_PARTY_NOTICES for the ONNX Runtime the plugin links statically.
# Usage: scripts/third_party_notices.sh 1.31.0
# The version is the `ms@X.Y.Z` in ort-sys's build/download/dist.tsv.
set -eu

VERSION=${1:?usage: third_party_notices.sh <onnxruntime version>}
SRC=https://raw.githubusercontent.com/microsoft/onnxruntime/v$VERSION
OUT=$(dirname "$0")/../THIRD_PARTY_NOTICES

{
    cat <<EOF
THIRD-PARTY NOTICES
===================

mq-bridge-onnx links ONNX Runtime $VERSION (https://github.com/microsoft/onnxruntime)
statically into its library. Its license and its third-party notices follow
unchanged. mq-bridge-onnx's own code is licensed MIT OR Apache-2.0, see
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
} >"$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "wrote $OUT"
