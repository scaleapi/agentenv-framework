#!/bin/sh
# Installs Node, pi and this agent's Python environment on Debian or Ubuntu, as root. The image build and
# install/v1 both run it, so an install into a task container matches the image.
set -eu
NODE_VERSION=22.19.0
PI_VERSION=1.1.0
here=$(cd "$(dirname "$0")" && pwd)

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-venv ripgrep xz-utils
rm -rf /var/lib/apt/lists/*

case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${arch}.tar.xz" | tar -xJ -C /opt
ln -sfn "/opt/node-v${NODE_VERSION}-linux-${arch}" /opt/node
PATH="/opt/node/bin:$PATH" npm install --global --prefix /opt/pi --ignore-scripts \
  "@earendil-works/pi-coding-agent@${PI_VERSION}"

python3 -m venv /opt/pi-a2a-venv
/opt/pi-a2a-venv/bin/pip install --no-cache-dir -r "$here/requirements.txt"
