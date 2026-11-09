#!/bin/sh
# The pinned ChatGPT app, from the zip the install downloaded and checked, unpacked beside the driver and never run, so it
# never updates itself; and Codex Computer Use's helper app copied out of it to where the layer's grant names it, a path
# that doesn't depend on this folder's name. As the session user, whom the helper must belong to.
set -eu
ditto -x -k ChatGPT.zip .
rm -f ChatGPT.zip
helper="ChatGPT.app/Contents/Resources/cua_node/lib/node_modules/@oai/sky/Codex Computer Use.app"
mkdir -p /opt/agentenv/codex-computer-use
rm -rf "/opt/agentenv/codex-computer-use/Codex Computer Use.app"
ditto "$helper" "/opt/agentenv/codex-computer-use/Codex Computer Use.app"
xattr -dr com.apple.quarantine ChatGPT.app /opt/agentenv/codex-computer-use || true
