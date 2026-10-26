#!/bin/sh
# Installs tmux and git for the agent's terminal and this agent's Python environment, in uv's own Python so the
# target's is never used, on Debian or Ubuntu, as root. The image build and install/v1 both run it, so an install
# into a task container matches the image.
set -eu
UV_VERSION=0.13.1
UV_SHA256_X86_64=6c86040a8f045ab9adc9e3ec4cdafa03fd00e9a988ed2bf28971c7fc2d0debd0
UV_SHA256_AARCH64=345a8308e4d1835cdfaadf692836fd3137a6d45864773f05ad7bf59442b1ba16
PYTHON_VERSION=3.12
here=$(cd "$(dirname "$0")" && pwd)

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git tmux
rm -rf /var/lib/apt/lists/*

case "$(uname -m)" in
  x86_64) arch=x86_64 sha256=$UV_SHA256_X86_64 ;;
  aarch64 | arm64) arch=aarch64 sha256=$UV_SHA256_AARCH64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
archive=uv-$arch-unknown-linux-musl.tar.gz
curl -fsSL --retry 3 "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$archive" -o "$stage/$archive"
echo "$sha256  $stage/$archive" | sha256sum -c -
tar -xzf "$stage/$archive" -C "$stage"
uv="$stage/uv-$arch-unknown-linux-musl/uv"

export UV_PYTHON_INSTALL_DIR=/opt/openhands-a2a-python UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR="$stage/cache"
"$uv" venv --quiet --clear --python "$PYTHON_VERSION" /opt/openhands-a2a-venv
"$uv" pip install --quiet --python /opt/openhands-a2a-venv/bin/python -r "$here/requirements.txt"
