"""Constants for the Playwright MCP website browser, Microsoft's Playwright MCP server (``@playwright/mcp``)."""

# Docker image tag for the browser's image when built locally.
PLAYWRIGHT_MCP_IMAGE_TAG = "mcp-playwright-mcp"

# The env id and environment_name its put writes.
PLAYWRIGHT_MCP_ENVIRONMENT_NAME = "playwright-mcp"

# The port the image's recorder serves the browser's env card on (recorder.py, entrypoint.sh's RECORDER_PORT).
PLAYWRIGHT_MCP_RECORDER_PORT = 18766

__all__ = [
    "PLAYWRIGHT_MCP_IMAGE_TAG",
    "PLAYWRIGHT_MCP_ENVIRONMENT_NAME",
    "PLAYWRIGHT_MCP_RECORDER_PORT",
]
