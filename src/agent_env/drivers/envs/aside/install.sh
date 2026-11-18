#!/bin/sh
# The pinned daemon component and CLI, unpacked to where the grants and the daemon service name them, as the session
# user. The helper's runtime copy is placed before the daemon first starts, since macOS keys its approval to that path.
set -eu
daemon=aside-daemon.crx
cli=aside-cli.zip
dest=/opt/agentenv/aside
helper="Aside Computer Use.app"
rm -rf "$dest"
mkdir -p "$dest/bin" "$dest/home/runtime-computer-use"
# A CRX is a signature header and then a zip, which bsdtar reads.
tar -xf "$daemon" -C "$dest"
ditto -x -k "$cli" "$dest"
ln -s "../Aside CLI.app/Contents/MacOS/aside" "$dest/bin/aside"
rm -f "$daemon" "$cli"
xattr -dr com.apple.quarantine "$dest" || true
ditto "$dest/$helper" "$dest/home/runtime-computer-use/$helper"
# Usage analytics are off from the daemon's first start.
mkdir -p "$dest/home/u/0"
printf '%s\n' '{"analytics":{"enabled":false}}' > "$dest/home/u/0/settings.json"
