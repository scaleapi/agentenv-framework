#!/bin/sh
# Start an installed agent from ROOT, in this shell's working directory: its environment from ROOT/agent.env and its
# own Python, whose packages hold the Antigravity SDK and its harness.
set -e
root=$1
set -a
. "$root/agent.env"
set +a
echo $$ > "$root/agent.pid"
export PATH="$root/bin:$PATH:$root/venv/bin" PYTHONPATH="$root/app"
# Within install.sh's stop timeout, so a reinstall's stop reaches the shutdown that ends running turns' harnesses.
exec "$root/venv/bin/uvicorn" a2a_server:app --host 0.0.0.0 --port "$A2A_PORT" --timeout-graceful-shutdown 5
