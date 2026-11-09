#!/usr/bin/env bash
set -euo pipefail

# MCP_PORT and MCP_HOST are injected by docker-compose (gateway convention).
# PLAYWRIGHT_MCP_VERSION is baked into the image at build time.

# The recorder fronts the MCP port: it serves the browser's card and recording there and passes every other request to
# the MCP server, which listens on loopback PLAYWRIGHT_MCP_PORT behind it. It serves the card on RECORDER_PORT at once,
# opens the MCP port only once the MCP server listens, so the port answering means the server does, and is restarted
# whenever it exits; the MCP server starts once the card is served, or after RECORDER_WAIT_S without it.
export MCP_PORT="${MCP_PORT:-18765}"
export RECORDER_PORT="${RECORDER_PORT:-18766}"
export PLAYWRIGHT_MCP_PORT="${PLAYWRIGHT_MCP_PORT:-18767}"
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
    --host 127.0.0.1 \
    --port "${PLAYWRIGHT_MCP_PORT}" \
    --allowed-hosts '*' \
    --caps vision
