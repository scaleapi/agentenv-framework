#!/usr/bin/env bash
set -euo pipefail

# MCP_PORT and MCP_HOST are injected by docker-compose (gateway convention).
# PLAYWRIGHT_MCP_VERSION is baked into the image at build time.
exec npx @playwright/mcp@${PLAYWRIGHT_MCP_VERSION} \
    --browser chromium \
    --headless \
    --isolated \
    --no-sandbox \
    --host "${MCP_HOST:-0.0.0.0}" \
    --port "${MCP_PORT:-18765}" \
    --allowed-hosts '*' \
    --caps vision
