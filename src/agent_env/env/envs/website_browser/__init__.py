"""Website browser (Playwright) MCP server constants."""

# Pinned version of @playwright/mcp to install in the Docker image.
# Update this when upgrading the Playwright MCP server.
PLAYWRIGHT_MCP_VERSION = "0.0.41"

# Docker image tag for the website browser server image.
WEBSITE_BROWSER_IMAGE_TAG = "mcp-website-browser"

# The environment_name used when creating MCPServerEnv.
WEBSITE_BROWSER_ENVIRONMENT_NAME = "website-browser"

# The service_version used when creating MCPServerEnv.
WEBSITE_BROWSER_SERVICE_VERSION = 1

__all__ = [
    "PLAYWRIGHT_MCP_VERSION",
    "WEBSITE_BROWSER_IMAGE_TAG",
    "WEBSITE_BROWSER_ENVIRONMENT_NAME",
    "WEBSITE_BROWSER_SERVICE_VERSION",
]
