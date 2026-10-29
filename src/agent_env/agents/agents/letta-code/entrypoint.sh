#!/bin/sh
set -e
# A stop waits this long for open requests, then the server shuts down and kills the CLIs of turns still running.
exec uvicorn a2a_server:app --host 0.0.0.0 --port "${A2A_PORT:-8000}" --timeout-graceful-shutdown 5
