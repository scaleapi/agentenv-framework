"""Website browser (Playwright) MCP server constants."""

# Docker image tag for the website browser server image.
WEBSITE_BROWSER_IMAGE_TAG = "mcp-website-browser"

# The environment_name used when creating MCPServerEnv.
WEBSITE_BROWSER_ENVIRONMENT_NAME = "website-browser"

# The port the image's recorder serves the browser's env card on (recorder.py, entrypoint.sh's RECORDER_PORT).
WEBSITE_BROWSER_RECORDER_PORT = 18766

__all__ = [
    "WEBSITE_BROWSER_IMAGE_TAG",
    "WEBSITE_BROWSER_ENVIRONMENT_NAME",
    "WEBSITE_BROWSER_RECORDER_PORT",
]
