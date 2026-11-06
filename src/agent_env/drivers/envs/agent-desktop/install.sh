#!/bin/sh
# agent-desktop's release, from the tarball the install downloaded and checked, in bin/ beside the driver: on macOS as the
# session user, whom its helper must belong to, with the quarantine a download carries removed, and on Linux as root.
set -eu
mkdir -p bin
tar -xzf agent-desktop.tar.gz -C bin
[ "$(uname)" != Darwin ] || xattr -dr com.apple.quarantine bin || true
rm -f agent-desktop.tar.gz
