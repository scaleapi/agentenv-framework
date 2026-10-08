#!/bin/sh
# Starts the installed agent (install/v1) with the credentials install wrote to agent.env; the caller sets A2A_PORT.
here=$(cd "$(dirname "$0")" && pwd)
set -a
. "$here/agent.env"
set +a
export PATH="/opt/pi-a2a-venv/bin:/opt/pi/bin:/opt/node/bin:$PATH"
cd "$here" && exec python3 a2a_server.py >/tmp/pi-a2a.log 2>&1
