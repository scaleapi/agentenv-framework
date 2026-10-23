#!/bin/sh
# Start an installed agent from ROOT, in this shell's working directory: its environment from ROOT/agent.env, its own
# Python and the Codex CLI, and its Node for a target that has none.
set -e
root=$1
set -a
. "$root/agent.env"
set +a
echo $$ > "$root/agent.pid"
export PATH="$root/bin:$PATH:$root/node/bin" PYTHONPATH="$root/app"
exec "$root/venv/bin/uvicorn" a2a_server:app --host 0.0.0.0 --port "$A2A_PORT"
