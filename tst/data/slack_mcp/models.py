"""
Data models for Slack MCP Server.

This module defines all the data models representing Slack entities,
following the structure defined in the Slack API specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import time


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ChannelType(str, Enum):
    """Types of channels in Slack."""

    PUBLIC_CHANNEL = "public_channel"
    PRIVATE_CHANNEL = "private_channel"
    IM = "im"
    MPIM = "mpim"


class MessageType(str, Enum):
    """Types of messages in Slack."""

    MESSAGE = "message"
    CHANNEL_JOIN = "channel_join"
    CHANNEL_LEAVE = "channel_leave"
    CHANNEL_TOPIC = "channel_topic"
    CHANNEL_PURPOSE = "channel_purpose"
    CHANNEL_NAME = "channel_name"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_slack_ts() -> str:
    """Generate a Slack-style timestamp (e.g., 1234567890.123456)."""
    now = time.time()
    return f"{now:.6f}"


def ts_to_datetime(ts: str) -> datetime:
    """Convert Slack timestamp to datetime."""
    return datetime.fromtimestamp(float(ts))


def datetime_to_ts(dt: datetime) -> str:
    """Convert datetime to Slack timestamp."""
    return f"{dt.timestamp():.6f}"


# =============================================================================
# CORE MODELS
# =============================================================================


@dataclass
class User:
    """Slack user object."""

    id: str = field(default_factory=lambda: f"U{uuid.uuid4().hex[:10].upper()}")
    name: str = ""
    real_name: str = ""
    display_name: str = ""
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    is_admin: bool = False
    deleted: bool = False
    timezone: str = "America/Los_Angeles"
    status: str = ""
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": self.id,
            "name": self.name,
            "real_name": self.real_name,
            "display_name": self.display_name,
            "profile": {
                "email": self.email,
                "image_72": self.avatar_url,
                "display_name": self.display_name,
                "real_name": self.real_name,
            },
            "is_bot": self.is_bot,
            "is_admin": self.is_admin,
            "deleted": self.deleted,
            "tz": self.timezone,
        }


@dataclass
class Channel:
    """Slack channel object."""

    id: str = field(default_factory=lambda: f"C{uuid.uuid4().hex[:10].upper()}")
    name: str = ""
    is_channel: bool = True
    is_private: bool = False
    is_im: bool = False
    is_mpim: bool = False
    is_archived: bool = False
    is_general: bool = False
    num_members: int = 0
    topic: str = ""
    purpose: str = ""
    created: str = field(default_factory=generate_slack_ts)
    creator: Optional[str] = None  # User ID
    members: List[str] = field(default_factory=list)  # List of User IDs

    def get_channel_type(self) -> ChannelType:
        """Get the type of channel."""
        if self.is_im:
            return ChannelType.IM
        elif self.is_mpim:
            return ChannelType.MPIM
        elif self.is_private:
            return ChannelType.PRIVATE_CHANNEL
        else:
            return ChannelType.PUBLIC_CHANNEL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "id": self.id,
            "name": self.name,
            "is_channel": self.is_channel,
            "is_private": self.is_private,
            "is_im": self.is_im,
            "is_mpim": self.is_mpim,
            "is_archived": self.is_archived,
            "is_general": self.is_general,
            "num_members": self.num_members,
            "created": int(float(self.created)),
        }

        if self.topic:
            result["topic"] = {"value": self.topic}
        if self.purpose:
            result["purpose"] = {"value": self.purpose}
        if self.creator:
            result["creator"] = self.creator

        return result


@dataclass
class Message:
    """Slack message object."""

    ts: str = field(default_factory=generate_slack_ts)
    type: str = "message"
    user: Optional[str] = None  # User ID
    text: str = ""
    channel: Optional[str] = None  # Channel ID
    thread_ts: Optional[str] = None  # Parent thread timestamp
    reply_count: int = 0
    reply_users_count: int = 0
    latest_reply: Optional[str] = None
    is_activity_message: bool = False
    subtype: Optional[str] = None  # For activity messages like channel_join
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)

    def is_thread_parent(self) -> bool:
        """Check if this message is a thread parent."""
        return self.reply_count > 0

    def is_thread_reply(self) -> bool:
        """Check if this message is a thread reply."""
        return self.thread_ts is not None and self.thread_ts != self.ts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "ts": self.ts,
            "type": self.type,
            "text": self.text,
        }

        if self.user:
            result["user"] = self.user

        if self.thread_ts:
            result["thread_ts"] = self.thread_ts

        if self.reply_count > 0:
            result["reply_count"] = self.reply_count
            result["reply_users_count"] = self.reply_users_count
            if self.latest_reply:
                result["latest_reply"] = self.latest_reply

        if self.subtype:
            result["subtype"] = self.subtype

        if self.reactions:
            result["reactions"] = self.reactions

        if self.files:
            result["files"] = self.files

        if self.attachments:
            result["attachments"] = self.attachments

        return result


# =============================================================================
# PAGINATION MODELS
# =============================================================================


@dataclass
class PaginatedChannelResponse:
    """Paginated response for channel list endpoints."""

    ok: bool = True
    channels: List[Channel] = field(default_factory=list)
    next_cursor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "ok": self.ok,
            "channels": [c.to_dict() for c in self.channels],
        }
        if self.next_cursor:
            result["response_metadata"] = {"next_cursor": self.next_cursor}
        else:
            result["response_metadata"] = {"next_cursor": ""}
        return result


@dataclass
class PaginatedMessageResponse:
    """Paginated response for message list endpoints."""

    ok: bool = True
    messages: List[Message] = field(default_factory=list)
    next_cursor: Optional[str] = None
    has_more: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "ok": self.ok,
            "messages": [m.to_dict() for m in self.messages],
            "has_more": self.has_more,
        }
        if self.next_cursor:
            result["response_metadata"] = {"next_cursor": self.next_cursor}
        return result


# =============================================================================
# ERROR MODELS
# =============================================================================


@dataclass
class SlackError:
    """Error response from Slack API."""

    ok: bool = False
    error: str = "unknown_error"
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "ok": self.ok,
            "error": self.error,
        }
        if self.warning:
            result["warning"] = self.warning
        return result


# =============================================================================
# MOCK DATA SCHEMA (for LLM data generation)
# =============================================================================


@dataclass
class MockDataSchema:
    """
    Schema for mock data JSON files used to populate the Slack database.
    
    This class documents the expected JSON format for data generation.
    LLMs should generate data in this format for loading via load_from_json().
    
    Expected JSON structure:
    ```json
    {
        "users": [
            {
                "id": "U0123456789",
                "name": "johndoe",
                "real_name": "John Doe",
                "display_name": "John",
                "email": "john@example.com",
                "avatar_url": "https://...",
                "is_bot": false,
                "is_admin": false,
                "deleted": false,
                "timezone": "America/New_York"
            }
        ],
        "channels": [
            {
                "id": "C0123456789",
                "name": "general",
                "is_channel": true,
                "is_private": false,
                "is_im": false,
                "is_mpim": false,
                "is_archived": false,
                "is_general": true,
                "num_members": 150,
                "topic": "Channel topic",
                "purpose": "Channel purpose",
                "created": "1609459200.000000",
                "creator": "U0123456789",
                "members": ["U0123456789", "U1234567890"]
            }
        ],
        "messages": [
            {
                "ts": "1704067200.000001",
                "type": "message",
                "user": "U0123456789",
                "text": "Hello world!",
                "channel": "C0123456789",
                "thread_ts": null,
                "reply_count": 0,
                "reply_users_count": 0,
                "latest_reply": null,
                "is_activity_message": false,
                "subtype": null,
                "reactions": [{"name": "thumbsup", "count": 2}],
                "files": [],
                "attachments": []
            }
        ]
    }
    ```
    
    Notes:
    - User IDs start with "U", Channel IDs with "C", "D" (DM), or "G" (group)
    - Timestamps are Slack-style: "seconds.microseconds" format
    - thread_ts references another message's ts for thread replies
    - is_activity_message=true for system messages (joins, leaves, etc.)
    """
    
    users: List[User] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        return {
            "users": [u.to_dict() for u in self.users],
            "channels": [c.to_dict() for c in self.channels],
            "messages": [m.to_dict() for m in self.messages],
        }
