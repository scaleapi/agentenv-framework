#!/bin/sh
set -e
# The CLI refuses --dangerously-skip-permissions as root unless it's told it's sandboxed.
if [ "$(id -u)" = "0" ]; then
    export IS_SANDBOX=1
fi
exec uvicorn a2a_server:app --host 0.0.0.0 --port "${A2A_PORT:-8000}"
