#!/bin/sh
set -e
exec uvicorn a2a_server:app --host 0.0.0.0 --port "${A2A_PORT:-8000}"
