#!/bin/sh
# Starts the installed agent (install/v1) with the credentials install wrote to agent.env; the caller sets A2A_PORT.
here=$(cd "$(dirname "$0")" && pwd)
# The agent works where it's started, the task container's working directory, unless that's the root.
[ "$(pwd)" = / ] || export PYDANTIC_AI_A2A_WORKSPACE="$(pwd)"
set -a
. "$here/agent.env"
eval "${AGENT_TRUST_ENV:-}"
set +a
export PATH="/opt/pydantic-ai-a2a-venv/bin:$PATH"
cd "$here" && exec python a2a_server.py >/tmp/pydantic-ai-a2a.log 2>&1
