#!/bin/sh
# Start an installed agent from ROOT, in this shell's working directory: its environment from ROOT/agent.env, its own
# Python and the Letta Code CLI, and its Node and Python for the model's commands on a target that has none.
set -e
root=$1
set -a
. "$root/agent.env"
set +a
echo $$ > "$root/agent.pid"
export PATH="$root/bin:$PATH:$root/node/bin:$root/venv/bin" PYTHONPATH="$root/app"
# Within install.sh's stop timeout, so a reinstall's stop reaches the shutdown that kills running turns' CLIs.
exec "$root/venv/bin/uvicorn" a2a_server:app --host 0.0.0.0 --port "$A2A_PORT" --timeout-graceful-shutdown 5
