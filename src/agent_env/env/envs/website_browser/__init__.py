"""Website browser (Playwright) MCP server constants."""

# Docker image tag for the website browser server image.
WEBSITE_BROWSER_IMAGE_TAG = "mcp-website-browser"

# The environment_name used when creating MCPServerEnv.
WEBSITE_BROWSER_ENVIRONMENT_NAME = "website-browser"

__all__ = [
    "WEBSITE_BROWSER_IMAGE_TAG",
    "WEBSITE_BROWSER_ENVIRONMENT_NAME",
]
