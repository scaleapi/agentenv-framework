#!/bin/sh
# CuaDriver.app, from the release tarball the install downloaded and checked, in /Applications, where its LaunchAgent
# starts it, with the quarantine a download carries removed.
set -eu
tar -xzf CuaDriver.tar.gz -C "$TMPDIR"
rm -rf /Applications/CuaDriver.app
ditto "$TMPDIR/cua-driver-rs-0.38.0-darwin-arm64/CuaDriver.app" /Applications/CuaDriver.app
xattr -dr com.apple.quarantine /Applications/CuaDriver.app || true
rm -f CuaDriver.tar.gz
