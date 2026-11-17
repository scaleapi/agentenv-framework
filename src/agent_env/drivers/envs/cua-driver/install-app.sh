#!/bin/sh
# CuaDriver.app, from the release tarball the install downloaded and checked, in /Applications, where its LaunchAgent
# starts it, with the quarantine a download carries removed.
set -eu
work=$(mktemp -d)
tar -xzf CuaDriver.tar.gz -C "$work"
rm -rf /Applications/CuaDriver.app
ditto "$work/cua-driver-rs-0.47.10-darwin-arm64/CuaDriver.app" /Applications/CuaDriver.app
rm -rf "$work"
xattr -dr com.apple.quarantine /Applications/CuaDriver.app || true
rm -f CuaDriver.tar.gz
