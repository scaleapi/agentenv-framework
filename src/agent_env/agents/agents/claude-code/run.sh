#!/bin/sh
# Start an installed agent from ROOT, in this shell's working directory: its environment from ROOT/agent.env, and its
# own Python and Claude Code ahead of the target's.
set -e
root=$1
set -a
. "$root/agent.env"
set +a
# The CLI refuses --dangerously-skip-permissions as root unless it's told it's sandboxed.
if [ "$(id -u)" = "0" ]; then
    export IS_SANDBOX=1
fi
echo $$ > "$root/agent.pid"
export PATH="$root/bin:$PATH" PYTHONPATH="$root/app"
exec "$root/venv/bin/uvicorn" a2a_server:app --host 0.0.0.0 --port "$A2A_PORT"
