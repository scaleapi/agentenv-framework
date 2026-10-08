#!/bin/sh
# Starts the installed agent (install/v1); the caller supplies LITELLM_* and A2A_PORT.
here=$(cd "$(dirname "$0")" && pwd)
export PATH="/opt/pi-a2a-venv/bin:/opt/pi/bin:/opt/node/bin:$PATH"
cd "$here" && exec python3 a2a_server.py >/tmp/pi-a2a.log 2>&1
