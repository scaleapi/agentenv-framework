#!/bin/sh
# Starts the installed agent (install/v1) with the credentials install wrote to agent.env; the caller sets A2A_PORT.
here=$(cd "$(dirname "$0")" && pwd)
# The agent works where it's started, the task container's working directory, unless that's the root.
[ "$(pwd)" = / ] || export OPENHANDS_A2A_WORKSPACE="$(pwd)"
set -a
. "$here/agent.env"
set +a
export PATH="/opt/openhands-a2a-venv/bin:$PATH" OPENHANDS_SUPPRESS_BANNER=1
cd "$here" && exec python a2a_server.py >/tmp/openhands-a2a.log 2>&1
