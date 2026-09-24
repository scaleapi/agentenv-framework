"""
Application configuration.
"""
import os
from typing import Optional


class Settings:
    """Configuration settings for the Slack Workspace Manager API."""

    def __init__(self):
        self.mcp_server_path = os.getenv("MCP_SERVER_PATH", "/mcp")
        self.sample_data_path = os.getenv("SAMPLE_DATA_PATH", "/mcp/sample_data.json")
        # CRITICAL FIX: Use PostgreSQL connection string, not SQLite path
        self.connection_string = os.getenv("DATABASE_URL", "postgresql://slack:slack@postgres:5432/slack")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"


settings = Settings()