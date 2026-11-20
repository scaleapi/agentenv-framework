#!/bin/sh
# Installs Node, pi and this agent's Python environment on Debian or Ubuntu, as root. The image build and
# install/v1 both run it, so an install into a task container matches the image.
set -eu
NODE_VERSION=22.23.3
# nodejs.org's SHASUMS256.txt for node-v$NODE_VERSION-linux-<arch>.tar.xz.
NODE_SHA256_X64=df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de
NODE_SHA256_ARM64=a44aeb94849a299b22df10b9e622ec2f605c2183501bc40590705131de7c740f
PI_VERSION=1.1.0
here=$(cd "$(dirname "$0")" && pwd)

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-venv ripgrep xz-utils
rm -rf /var/lib/apt/lists/*

case "$(uname -m)" in
  x86_64) arch=x64 node_sha256=$NODE_SHA256_X64 ;;
  aarch64 | arm64) arch=arm64 node_sha256=$NODE_SHA256_ARM64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
node_archive=$(mktemp)
curl -fsSL --retry 3 -o "$node_archive" "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${arch}.tar.xz"
echo "$node_sha256  $node_archive" | sha256sum -c -
tar -xJf "$node_archive" -C /opt
rm -f "$node_archive"
ln -sfn "/opt/node-v${NODE_VERSION}-linux-${arch}" /opt/node
PATH="/opt/node/bin:$PATH" npm install --global --prefix /opt/pi --ignore-scripts \
  "@earendil-works/pi-coding-agent@${PI_VERSION}"

python3 -m venv /opt/pi-a2a-venv
/opt/pi-a2a-venv/bin/pip install --no-cache-dir -r "$here/requirements.txt"
