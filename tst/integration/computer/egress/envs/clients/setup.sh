#!/bin/sh
# Run as root once the app's files, packages and downloads are in: Node's official build under /opt/node.
set -e
cd "$(dirname "$0")"
mkdir -p /opt/node
tar -xJf node.tar.xz -C /opt/node --strip-components=1
ln -sf /opt/node/bin/node /usr/local/bin/node
