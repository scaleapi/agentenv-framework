"""
Base Service abstraction for Synthetic MCP Servers.

This module provides a base class for MCP service implementations,
handling common functionality like health checks, data reset, and time utilities.
"""

import os
from datetime import datetime
from typing import Annotated, Any, Optional, TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Default port for MCPServerEnv compatibility
_DEFAULT_MCP_PORT = 18765

if TYPE_CHECKING:
    from .base_database import BaseDatabase


class BaseService:
    """
    Base class for MCP service implementations.

    Provides common functionality including:
    - Health check endpoint
    - Data reset with optional mock data reload
    - Current time utility
    - Database integration

    Example usage:
        class MyService(BaseService):
            def __init__(self):
                self.db = MyDatabase()
                super().__init__("my_service", db=self.db)

                # Register service-specific tools
                self._register_my_tools()
    """

    def __init__(
        self,
        default_service_name: str,
        db: Optional["BaseDatabase"] = None,
        mock_data_path: Optional[str] = None,
        **mcp_kwargs: Any,
    ):
        """
        Initialize the base service.

        Args:
            default_service_name: Default name of the MCP service (can be overridden by SERVICE_NAME env var).
            db: Optional database instance (must extend BaseDatabase).
            mock_data_path: Optional path to JSON file with initial mock data.
            **mcp_kwargs: Additional arguments to pass to FastMCP.
        """
        self.service_name = os.environ.get("SERVICE_NAME", default_service_name)
        self.mcp = FastMCP(self.service_name, **mcp_kwargs)
        self.db = db
        self._mock_data_path = mock_data_path

        # Configure MCP settings for HTTP transport (MCPServerEnv compatibility)
        self.mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        self.mcp.settings.port = int(os.environ.get("MCP_PORT", str(_DEFAULT_MCP_PORT)))
        self.mcp.settings.transport_security.enable_dns_rebinding_protection = False

        # --- Auto-Register Common Tools with Prefixes ---
        self.mcp.tool(name=f"{self.service_name}_health")(self.check_health)
        self.mcp.tool(name=f"{self.service_name}_reset")(self._reset_data_tool)

        # --- Register REST endpoints ---
        import json as _json
        from starlette.requests import Request as _Request
        from starlette.responses import Response as _Response

        if self.db:
            @self.mcp.custom_route("/export-state", methods=["GET"])
            async def export_state_endpoint(request: _Request) -> _Response:
                data = self.db.export_state()
                return _Response(content=_json.dumps(data, default=str), media_type="application/json")

        @self.mcp.custom_route("/api/reset", methods=["POST"])
        async def reset_endpoint(request: _Request) -> _Response:
            body = await request.body()
            mock_data_path = None
            if body:
                data = _json.loads(body)
                mock_data_path = data.get("mock_data_path")
            result = self.reset_data(mock_data_path or self._mock_data_path)
            return _Response(content=_json.dumps({"ok": True, "message": result}), media_type="application/json")

    def check_health(self) -> str:
        """
        Check the health of the service.

        Returns:
            Health status message including database connection info if available.
        """
        status = f"{self.service_name} is running and healthy."

        if self.db:
            try:
                conn_info = self.db.get_connection_info()
                status += f"\nDatabase: {conn_info['backend']} - Connected: {conn_info['is_connected']}"
            except Exception as e:
                status += f"\nDatabase: Error checking connection - {str(e)}"

        return status

    def _reset_data_tool(self, mock_data_path: Annotated[Optional[str], Field(description="Path to JSON file with mock data to reload")] = None) -> str:
        """
        Reset all data and optionally reload from a JSON file.
        """
        # Use provided path or fall back to default
        path = mock_data_path or self._mock_data_path
        return self.reset_data(path)

    def reset_data(self, mock_data_path: Optional[str] = None) -> str:
        """
        Reset all data in the database.

        This method can be overridden by subclasses for custom reset behavior.

        Args:
            mock_data_path: Optional path to JSON file with mock data to reload.

        Returns:
            Status message about the reset operation.
        """
        if not self.db:
            return f"No database configured for {self.service_name}."

        # Reset the database
        self.db.reset_all()

        # Reload mock data if path provided
        if mock_data_path:
            try:
                counts = self.db.load_from_json(mock_data_path)
                return f"Data reset for {self.service_name}. Loaded: {counts}"
            except Exception as e:
                return f"Data reset for {self.service_name}, but failed to load mock data: {str(e)}"

        return (
            f"Data reset successfully for {self.service_name}. Database is now empty."
        )

    def run(self):
        """Start the MCP server with HTTP transport."""
        print(f"Starting {self.service_name} on {self.mcp.settings.host}:{self.mcp.settings.port}")
        self.mcp.run(transport="streamable-http")


if __name__ == "__main__":
    service = BaseService("base_service")
    service.run()
