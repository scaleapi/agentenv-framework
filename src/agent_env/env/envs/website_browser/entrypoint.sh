#!/usr/bin/env bash
set -euo pipefail

# MCP_PORT and MCP_HOST are injected by docker-compose (gateway convention).
# PLAYWRIGHT_MCP_VERSION is baked into the image at build time.

# The recorder serves the browser's card on RECORDER_PORT, restarted whenever it exits; the MCP server starts once
# it listens, so the card is there by the time the server is healthy, or after RECORDER_WAIT_S without it.
export RECORDER_PORT="${RECORDER_PORT:-18766}"
(while true; do /opt/recorder/bin/python /app/recorder.py || true; sleep 1; done) &
for _ in $(seq 1 "$(( ${RECORDER_WAIT_S:-20} * 10 ))"); do
    (exec 3<>"/dev/tcp/127.0.0.1/${RECORDER_PORT}") 2>/dev/null && break
    sleep 0.1
done

exec npx @playwright/mcp@${PLAYWRIGHT_MCP_VERSION} \
    --browser chromium \
    --headless \
    --isolated \
    --no-sandbox \
    --host "${MCP_HOST:-0.0.0.0}" \
    --port "${MCP_PORT:-18765}" \
    --allowed-hosts '*' \
    --caps vision
