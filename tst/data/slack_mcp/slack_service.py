"""
Slack MCP Service - MCP server implementation for Slack API.

This module provides an MCP server that simulates the Slack API endpoints,
implementing tools as defined in the OpenAPI specification.
"""

import json
import sys
import importlib.util
from pathlib import Path
from typing import Annotated, Optional

from pydantic import Field


def _import_base_service():
    """Import BaseService handling hyphenated directory names."""
    # Get the path to the BaseService module
    current_dir = Path(__file__).parent
    base_service_path = current_dir.parent / "base_mcp" / "base_service.py"

    # Check if already imported
    if "slack_base_service" in sys.modules:
        return sys.modules["slack_base_service"].BaseService

    # Import using importlib
    spec = importlib.util.spec_from_file_location(
        "slack_base_service", base_service_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["slack_base_service"] = module
    spec.loader.exec_module(module)
    return module.BaseService


BaseService = _import_base_service()


def _import_local_modules():
    """Import local modules handling hyphenated directory names."""
    current_dir = Path(__file__).parent

    # Import models
    models_path = current_dir / "models.py"
    if "slack_models" not in sys.modules:
        spec = importlib.util.spec_from_file_location("slack_models", models_path)
        models = importlib.util.module_from_spec(spec)
        sys.modules["slack_models"] = models
        spec.loader.exec_module(models)
    else:
        models = sys.modules["slack_models"]

    # Import database
    db_path = current_dir / "database.py"
    if "slack_database" not in sys.modules:
        spec = importlib.util.spec_from_file_location("slack_database", db_path)
        db = importlib.util.module_from_spec(spec)
        sys.modules["slack_database"] = db
        spec.loader.exec_module(db)
    else:
        db = sys.modules["slack_database"]

    return models, db


# Try relative imports first, fall back to dynamic import
try:
    from .database import SlackDatabase
    from .models import (
        Channel,
        Message,
        User,
        ChannelType,
        PaginatedChannelResponse,
        PaginatedMessageResponse,
        SlackError,
        generate_slack_ts,
    )
except ImportError:
    _models, _db = _import_local_modules()

    SlackDatabase = _db.SlackDatabase
    Channel = _models.Channel
    Message = _models.Message
    User = _models.User
    ChannelType = _models.ChannelType
    PaginatedChannelResponse = _models.PaginatedChannelResponse
    PaginatedMessageResponse = _models.PaginatedMessageResponse
    SlackError = _models.SlackError
    generate_slack_ts = _models.generate_slack_ts


class SlackService(BaseService):
    """
    Slack MCP Service.

    Provides MCP tools for interacting with a simulated Slack workspace,
    including channels, messages, threads, and search.

    Requires PostgreSQL database via DATABASE_URL environment variable.

    Example usage:
        # Requires DATABASE_URL env var
        service = SlackService(connection_string=os.environ["DATABASE_URL"])
    """

    def __init__(self, connection_string: str):
        """
        Initialize the Slack MCP service.

        Args:
            connection_string: PostgreSQL database connection string.
                              e.g., "postgresql://user:pass@localhost:5432/slack"
        """
        # Initialize database with PostgreSQL backend
        self.db = SlackDatabase(connection_string=connection_string)

        # Initialize base service with database
        super().__init__("slack", db=self.db)

        # Register all Slack-specific tools
        self._register_channel_tools()
        self._register_conversation_tools()

    # =========================================================================
    # CHANNEL TOOLS
    # =========================================================================

    def _register_channel_tools(self):
        """Register channel-related MCP tools."""

        @self.mcp.tool(name="channels_list")
        def channels_list(
            channel_types: Annotated[str, Field(description="Comma-separated subset of public_channel, private_channel, im, mpim")],
            cursor: Annotated[Optional[str], Field(description="Returned next_cursor from prior page")] = None,
            limit: Annotated[int, Field(description="Number of items to return (1-999, default 100)")] = 100,
            sort: Annotated[Optional[str], Field(description='If set to "popularity", sorts by member/participant count (descending)')] = None,
        ) -> str:
            """
            List channels, DMs, and Group DMs.

            Lists channels based on types. Sorted by popularity (member count) if requested.
            If a requested type has no data, returns an empty page (no error).
            """
            # Parse channel types
            types_list = [t.strip() for t in channel_types.split(",")]
            valid_types = {"public_channel", "private_channel", "im", "mpim"}
            types_list = [t for t in types_list if t in valid_types]

            if not types_list:
                error = SlackError(error="invalid_types")
                return json.dumps(error.to_dict())

            # Validate limit
            limit = max(1, min(limit, 999))

            # Check for popularity sort
            sort_by_popularity = sort == "popularity"

            channels, next_cursor = self.db.list_channels(
                channel_types=types_list,
                cursor=cursor,
                limit=limit,
                sort_by_popularity=sort_by_popularity,
            )

            response = PaginatedChannelResponse(
                ok=True,
                channels=channels,
                next_cursor=next_cursor,
            )

            return json.dumps(response.to_dict())

    # =========================================================================
    # CONVERSATION TOOLS
    # =========================================================================

    def _register_conversation_tools(self):
        """Register conversation-related MCP tools."""

        @self.mcp.tool(name="conversations_history")
        def conversations_history(
            channel_id: Annotated[str, Field(description="Channel ID (C.../D.../G...) or Name (#general / @username)")],
            cursor: Annotated[Optional[str], Field(description="Paging cursor")] = None,
            include_activity_messages: Annotated[bool, Field(description="Include join/leave/topic messages")] = False,
            limit: Annotated[str, Field(description='Time window (1d, 1w, 30d, 90d) OR numeric count as string ("50"). Must be empty if cursor is provided.')] = "1d",
        ) -> str:
            """
            Get messages in a channel or DM.

            Retrieves message history.
            Order: ts desc (newest first).
            Constraints: limit and cursor are mutually exclusive.
            """
            # Resolve channel ID from name if needed
            resolved_channel_id = self._resolve_channel_id(channel_id)
            if not resolved_channel_id:
                error = SlackError(error="channel_not_found")
                return json.dumps(error.to_dict())

            # If cursor is provided, ignore limit
            if cursor:
                limit = "100"

            messages, next_cursor = self.db.get_conversation_history(
                channel_id=resolved_channel_id,
                cursor=cursor,
                limit=limit,
                include_activity_messages=include_activity_messages,
            )

            response = PaginatedMessageResponse(
                ok=True,
                messages=messages,
                next_cursor=next_cursor,
                has_more=next_cursor is not None,
            )

            return json.dumps(response.to_dict())

        @self.mcp.tool(name="conversations_replies")
        def conversations_replies(
            channel_id: Annotated[str, Field(description="Channel ID (C.../D.../G...) or Name (#general / @username)")],
            thread_ts: Annotated[str, Field(description="Slack style timestamp (e.g., 1234567890.123456)")],
            cursor: Annotated[Optional[str], Field(description="Paging cursor")] = None,
            include_activity_messages: Annotated[bool, Field(description="Include activity messages")] = False,
            limit: Annotated[str, Field(description="Time window or numeric count string. Empty if cursor provided.")] = "1d",
        ) -> str:
            """
            Get an entire message thread.

            Retrieves a thread by parent timestamp.
            Order: ts asc (oldest first), starting with the parent.
            Constraints: limit and cursor are mutually exclusive.
            """
            # Resolve channel ID from name if needed
            resolved_channel_id = self._resolve_channel_id(channel_id)
            if not resolved_channel_id:
                error = SlackError(error="channel_not_found")
                return json.dumps(error.to_dict())

            # If cursor is provided, ignore limit
            if cursor:
                limit = "100"

            messages, next_cursor = self.db.get_thread_replies(
                channel_id=resolved_channel_id,
                thread_ts=thread_ts,
                cursor=cursor,
                limit=limit,
                include_activity_messages=include_activity_messages,
            )

            response = PaginatedMessageResponse(
                ok=True,
                messages=messages,
                next_cursor=next_cursor,
                has_more=next_cursor is not None,
            )

            return json.dumps(response.to_dict())

        @self.mcp.tool(name="conversations_search_messages")
        def search_messages(
            search_query: Annotated[Optional[str], Field(description="Case-insensitive substring or Slack URL. Required if no filters.")] = None,
            cursor: Annotated[Optional[str], Field(description="Paging cursor")] = None,
            limit: Annotated[int, Field(description="Max results (1-100, default 20)")] = 20,
            filter_date_on: Annotated[Optional[str], Field(description="Exact date (YYYY-MM-DD) or natural language (Yesterday)")] = None,
            filter_date_during: Annotated[Optional[str], Field(description='Calendar period (e.g., "July")')] = None,
            filter_date_after: Annotated[Optional[str], Field(description="After date")] = None,
            filter_date_before: Annotated[Optional[str], Field(description="Before date")] = None,
            filter_in_channel: Annotated[Optional[str], Field(description="Channel ID or #name")] = None,
            filter_in_im_or_mpim: Annotated[Optional[str], Field(description="IM/MPIM ID or @username_dm")] = None,
            filter_threads_only: Annotated[bool, Field(description="Only return thread messages")] = False,
            filter_users_from: Annotated[Optional[str], Field(description="User ID or @display_name (sender)")] = None,
            filter_users_with: Annotated[Optional[str], Field(description="User ID or @display_name (participant)")] = None,
        ) -> str:
            """
            Search across messages.

            Search with filters or text query.
            Sort: ts desc.
            Special: If search_query contains a Slack URL, ignores all parameters
                     and returns that single message.
            Precedence: filter_date_on > during > after/before.
            """
            # Validate that we have at least a query or some filter
            has_filter = any(
                [
                    filter_date_on,
                    filter_date_during,
                    filter_date_after,
                    filter_date_before,
                    filter_in_channel,
                    filter_in_im_or_mpim,
                    filter_threads_only,
                    filter_users_from,
                    filter_users_with,
                ]
            )

            if not search_query and not has_filter:
                error = SlackError(error="missing_query_or_filter")
                return json.dumps(error.to_dict())

            messages, next_cursor = self.db.search_messages(
                search_query=search_query,
                cursor=cursor,
                limit=limit,
                filter_date_on=filter_date_on,
                filter_date_during=filter_date_during,
                filter_date_after=filter_date_after,
                filter_date_before=filter_date_before,
                filter_in_channel=filter_in_channel,
                filter_in_im_or_mpim=filter_in_im_or_mpim,
                filter_threads_only=filter_threads_only,
                filter_users_from=filter_users_from,
                filter_users_with=filter_users_with,
            )

            response = PaginatedMessageResponse(
                ok=True,
                messages=messages,
                next_cursor=next_cursor,
                has_more=next_cursor is not None,
            )

            return json.dumps(response.to_dict())

        @self.mcp.tool(name="conversations_add_message")
        def add_message(
            channel_id: Annotated[str, Field(description="Channel ID or name (#general) to post to")],
            payload: Annotated[str, Field(description="Message content (validates non-empty)")],
            content_type: Annotated[str, Field(description="Content type (text/markdown or text/plain)")] = "text/markdown",
            thread_ts: Annotated[Optional[str], Field(description="Optional thread timestamp to reply to")] = None,
        ) -> str:
            """
            Post a message (Deferred).

            Status: Not implemented in v0.
            Future endpoint to persist messages and update indices.
            """
            error = SlackError(error="not_implemented")
            return json.dumps(
                {
                    "ok": False,
                    "error": "not_implemented",
                    "status": 501,
                    "message": "This endpoint is not implemented in v0.",
                }
            )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _resolve_channel_id(self, identifier: str) -> Optional[str]:
        """
        Resolve a channel identifier to a channel ID.

        Args:
            identifier: Channel ID (C.../D.../G...) or name (#general / @username)

        Returns:
            Channel ID if found, None otherwise
        """
        # If it looks like an ID already, verify it exists
        if identifier.startswith(("C", "D", "G")):
            channel = self.db.get_channel(identifier)
            return identifier if channel else None

        # Try to resolve by name
        name = identifier.lstrip("#").lstrip("@")
        channel = self.db.get_channel_by_name(name)
        return channel.id if channel else None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import os

    connection_string = os.environ.get("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL environment variable is required")

    service = SlackService(connection_string=connection_string)
    service.run()
