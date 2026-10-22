#!/bin/sh
# Installs the mq-bridge-onnx plugin from a GitHub release where mqb finds it.
#   curl -fsSL https://raw.githubusercontent.com/marcomq/mq-bridge-onnx/main/install.sh | sh
# Optional: a version as argument (`| sh -s -- 0.1.1`), MQB_ONNX_DIR for another directory.
set -eu

REPO=https://github.com/marcomq/mq-bridge-onnx
DIR=${MQB_ONNX_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/mq-bridge/plugins}

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) TARGET=aarch64-apple-darwin ;;
    Linux-x86_64) TARGET=x86_64-unknown-linux-gnu ;;
    Linux-aarch64) TARGET=aarch64-unknown-linux-gnu ;;
    *) echo "No build for $(uname -s) $(uname -m)." >&2 && exit 1 ;;
esac

VERSION=${1:-$(curl -fsSLo /dev/null -w '%{url_effective}' "$REPO/releases/latest")}
VERSION=${VERSION##*/}
ARCHIVE=mq-bridge-onnx-$VERSION-$TARGET.tar.gz
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cd "$TMP"
curl -fsSLO "$REPO/releases/download/$VERSION/$ARCHIVE"
curl -fsSLO "$REPO/releases/download/$VERSION/$ARCHIVE.sha256"
if command -v sha256sum >/dev/null; then sha256sum -c "$ARCHIVE.sha256"; else shasum -a 256 -c "$ARCHIVE.sha256"; fi
tar xzf "$ARCHIVE"

mkdir -p "$DIR/mq-bridge-onnx"
for LIB in libmq_bridge_onnx.*; do
    # Copy, then rename: overwriting a loaded library can crash a running mqb.
    cp "$LIB" "$DIR/.$LIB.new" && mv -f "$DIR/.$LIB.new" "$DIR/$LIB"
done
# Notices go in a subfolder, so they don't overwrite those of other plugins.
for F in THIRD_PARTY_NOTICES LICENSE-MIT LICENSE-APACHE; do
    if [ -f "$F" ]; then cp "$F" "$DIR/mq-bridge-onnx/"; fi
done
echo "Installed mq-bridge-onnx $VERSION to $DIR"
