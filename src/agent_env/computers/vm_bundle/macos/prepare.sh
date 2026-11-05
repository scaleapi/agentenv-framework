#!/bin/bash
# prepare.sh: a macOS base readied for agent-env, once a composed VM bundle, before its layers, in the VM as its session
# user. /opt/agentenv holds the layers; the pinned Python, with a current Tk, which layers' venvs and the probe's fixture
# run on, is python3 first on the PATH of what services run; agentenv-service starts each service with the deploy's
# environment, and set-display follows the virtual display's size at login; host.docker.internal is the VM's own
# loopback, where a deploy forwards this Mac's grant server; and nothing of the base's own is in an agent's way: Cua
# Spacesd and the Login Items banner are off.
set -euo pipefail
. "/Volumes/My Shared Files/agentenv-build/prepare/lib.sh"

[ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || fail "a macOS VM bundle is composed on Apple silicon"
csrutil status | grep -q disabled || fail "the base's System Integrity Protection is on, so no TCC permission can be granted"
as_root mkdir -p /opt/agentenv/bin /opt/agentenv/drivers /opt/agentenv/apps /etc/agentenv
as_root chown -R "$(id -un):staff" /opt/agentenv
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/agentenv"

pinned_python
/opt/agentenv/bin/python3 -c 'import tkinter; assert tkinter.TkVersion >= 8.6' || fail "its Python has no Tk"
install -m 755 "$build/prepare/agentenv-service" /opt/agentenv/bin/agentenv-service
grep -q "[[:space:]]host[.]docker[.]internal$" /etc/hosts ||
    as_root sh -c 'echo "127.0.0.1 host.docker.internal" >> /etc/hosts'
command -v swiftc > /dev/null || fail "the base has no swiftc, which builds set-display"
swiftc -O "$build/prepare/set-display.swift" -o /opt/agentenv/bin/set-display

if [ -f /Library/LaunchAgents/com.trycua.spacesd.plist ]; then
    as_root mv /Library/LaunchAgents/com.trycua.spacesd.plist /Library/LaunchAgents/com.trycua.spacesd.plist.disabled
fi
launchctl bootout "gui/$(id -u)/com.trycua.spacesd" 2> /dev/null || true
pkill -x cua-spacesd 2> /dev/null || true
launchctl bootout "gui/$(id -u)/com.apple.backgroundtaskmanagement.agent" 2> /dev/null || true
launchctl disable "gui/$(id -u)/com.apple.backgroundtaskmanagement.agent"
