#!/bin/sh
# agent-desktop's release, from the tarball the install downloaded and checked, in bin/ beside the driver, as the session
# user, whom its macOS helper must belong to, with the quarantine a download carries on macOS removed.
set -eu
mkdir -p bin
tar -xzf agent-desktop.tar.gz -C bin
[ "$(uname)" != Darwin ] || xattr -dr com.apple.quarantine bin || true
rm -f agent-desktop.tar.gz
