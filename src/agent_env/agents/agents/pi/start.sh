#!/bin/sh
# Starts the installed agent (install/v1) with the credentials install wrote to agent.env; the caller sets A2A_PORT.
here=$(cd "$(dirname "$0")" && pwd)
# The agent works where it's started, the task container's working directory, unless that's the root.
[ "$(pwd)" = / ] || export PI_A2A_WORKSPACE="$(pwd)"
set -a
. "$here/agent.env"
set +a
export PATH="/opt/pi-a2a-venv/bin:/opt/pi/bin:/opt/node/bin:$PATH"
# The build context keeps the shared egress rule and the clock page's format in the folders it takes them as
# (shared_sources.toml), which the image copies beside the server instead.
export PYTHONPATH="$here/agent_clock:$here/clock${PYTHONPATH:+:$PYTHONPATH}"
cd "$here" && exec python3 a2a_server.py >/tmp/pi-a2a.log 2>&1
