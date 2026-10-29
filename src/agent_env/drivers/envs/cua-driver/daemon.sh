#!/bin/sh
# cua-driver's daemon, once X answers: from the session user's home, where the apps it launches start too, on the
# socket the driver reaches it at, with approvals off, since nobody is at the computer to give them, and nothing sent
# to trycua.
set -eu
export DO_NOT_TRACK=1 CUA_DRIVER_RS_TELEMETRY_ENABLED=false CUA_DRIVER_RS_UPDATE_CHECK=false
venv="$(cd "$(dirname "$0")" && pwd)/venv"
display="${DISPLAY#:}"
while [ ! -S "/tmp/.X11-unix/X${display%%.*}" ]; do sleep 0.2; done
cd "$HOME"
exec "$venv/bin/cua-driver" serve --socket /run/agentenv/cua-driver.sock --dangerously-bypass-approvals --no-overlay
