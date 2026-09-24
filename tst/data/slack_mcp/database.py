"""
Database layer for Slack MCP Server using SQLAlchemy.

This module provides database operations for the Slack MCP server,
extending the BaseDatabase class for consistent database management.

Features:
1. **Backend Agnostic**: Supports SQLite (memory/file) and PostgreSQL via BaseDatabase
2. **Type Safety**: Excellent typing support with modern Python
3. **Relationship Handling**: Easy to define and navigate entity relationships
4. **Session Management**: Built-in connection pooling and transaction handling
5. **Query Builder**: Flexible ORM and Core query APIs
"""

import json
import sys
import importlib.util
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    desc,
    asc,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# =============================================================================
# IMPORT BASE DATABASE
# =============================================================================


def _import_base_database():
    """Import BaseDatabase handling hyphenated directory names."""
    current_dir = Path(__file__).parent
    base_db_path = current_dir.parent / "base_mcp" / "base_database.py"

    # Check if already imported
    if "slack_base_database_module" in sys.modules:
        return (
            sys.modules["slack_base_database_module"].BaseDatabase,
            sys.modules["slack_base_database_module"].DatabaseBackend,
        )

    # Import using importlib
    spec = importlib.util.spec_from_file_location("slack_base_database_module", base_db_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["slack_base_database_module"] = module
    spec.loader.exec_module(module)
    return module.BaseDatabase, module.DatabaseBackend


# Try relative imports first, fall back to dynamic import for BaseDatabase
try:
    from ..base_mcp.base_database import BaseDatabase, DatabaseBackend
except ImportError:
    BaseDatabase, DatabaseBackend = _import_base_database()


# Handle imports from models module
try:
    from .models import (
        User,
        Channel,
        Message,
        ChannelType,
        MessageType,
        generate_slack_ts,
        ts_to_datetime,
        datetime_to_ts,
    )
except ImportError:
    # Direct import when not running as package
    from pathlib import Path as _Path

    _models_path = _Path(__file__).parent / "models.py"
    if "slack_models" not in sys.modules:
        _spec = importlib.util.spec_from_file_location("slack_models", _models_path)
        _models = importlib.util.module_from_spec(_spec)
        sys.modules["slack_models"] = _models
        _spec.loader.exec_module(_models)
    else:
        _models = sys.modules["slack_models"]

    User = _models.User
    Channel = _models.Channel
    Message = _models.Message
    ChannelType = _models.ChannelType
    MessageType = _models.MessageType
    generate_slack_ts = _models.generate_slack_ts
    ts_to_datetime = _models.ts_to_datetime
    datetime_to_ts = _models.datetime_to_ts


# =============================================================================
# SQLALCHEMY BASE FOR SLACK TABLES
# =============================================================================


class SlackBase(DeclarativeBase):
    """Base class for Slack-specific SQLAlchemy models."""

    pass


# =============================================================================
# DATABASE TABLES
# =============================================================================


class UserTable(SlackBase):
    """SQLAlchemy model for Slack users."""

    __tablename__ = "slack_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    real_name: Mapped[str] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    timezone: Mapped[str] = mapped_column(String(100), default="America/Los_Angeles")
    status: Mapped[str] = mapped_column(String(255), default="")
    title: Mapped[str] = mapped_column(String(255), default="")

    # Relationships
    messages: Mapped[List["MessageTable"]] = relationship(
        "MessageTable",
        back_populates="author",
        foreign_keys="MessageTable.user_id",
    )

    def to_model(self) -> User:
        """Convert to domain model."""
        return User(
            id=self.id,
            name=self.name or "",
            real_name=self.real_name or "",
            display_name=self.display_name or "",
            email=self.email,
            avatar_url=self.avatar_url,
            is_bot=self.is_bot,
            is_admin=self.is_admin,
            deleted=self.deleted,
            timezone=self.timezone,
            status=self.status or "",
            title=self.title or "",
        )

    @classmethod
    def from_model(cls, user: User) -> "UserTable":
        """Create from domain model."""
        return cls(
            id=user.id,
            name=user.name,
            real_name=user.real_name,
            display_name=user.display_name,
            email=user.email,
            avatar_url=user.avatar_url,
            is_bot=user.is_bot,
            is_admin=user.is_admin,
            deleted=user.deleted,
            timezone=user.timezone,
            status=user.status,
            title=user.title,
        )


class ChannelTable(SlackBase):
    """SQLAlchemy model for Slack channels."""

    __tablename__ = "slack_channels"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    is_channel: Mapped[bool] = mapped_column(Boolean, default=True)
    is_private: Mapped[bool] = mapped_column(Boolean, default=False)
    is_im: Mapped[bool] = mapped_column(Boolean, default=False)
    is_mpim: Mapped[bool] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    is_general: Mapped[bool] = mapped_column(Boolean, default=False)
    num_members: Mapped[int] = mapped_column(Integer, default=0)
    topic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_ts: Mapped[float] = mapped_column(Float, default=lambda: datetime.now().timestamp())
    creator_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("slack_users.id"), nullable=True
    )
    members_json: Mapped[str] = mapped_column(Text, default="[]")

    # Relationships
    messages: Mapped[List["MessageTable"]] = relationship(
        "MessageTable",
        back_populates="channel_obj",
        foreign_keys="MessageTable.channel_id",
    )

    def to_model(self) -> Channel:
        """Convert to domain model."""
        return Channel(
            id=self.id,
            name=self.name or "",
            is_channel=self.is_channel,
            is_private=self.is_private,
            is_im=self.is_im,
            is_mpim=self.is_mpim,
            is_archived=self.is_archived,
            is_general=self.is_general,
            num_members=self.num_members,
            topic=self.topic or "",
            purpose=self.purpose or "",
            created=f"{self.created_ts:.6f}",
            creator=self.creator_id,
            members=json.loads(self.members_json),
        )

    @classmethod
    def from_model(cls, channel: Channel) -> "ChannelTable":
        """Create from domain model."""
        return cls(
            id=channel.id,
            name=channel.name,
            is_channel=channel.is_channel,
            is_private=channel.is_private,
            is_im=channel.is_im,
            is_mpim=channel.is_mpim,
            is_archived=channel.is_archived,
            is_general=channel.is_general,
            num_members=channel.num_members,
            topic=channel.topic,
            purpose=channel.purpose,
            created_ts=float(channel.created),
            creator_id=channel.creator,
            members_json=json.dumps(channel.members),
        )


class MessageTable(SlackBase):
    """SQLAlchemy model for Slack messages."""

    __tablename__ = "slack_messages"

    # Use composite primary key: ts + channel_id
    ts: Mapped[str] = mapped_column(String(50), primary_key=True)
    channel_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("slack_channels.id"), primary_key=True
    )
    type: Mapped[str] = mapped_column(String(50), default="message")
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("slack_users.id"), nullable=True
    )
    text: Mapped[str] = mapped_column(Text, default="")
    thread_ts: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_users_count: Mapped[int] = mapped_column(Integer, default=0)
    latest_reply: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_activity_message: Mapped[bool] = mapped_column(Boolean, default=False)
    subtype: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    reactions_json: Mapped[str] = mapped_column(Text, default="[]")
    files_json: Mapped[str] = mapped_column(Text, default="[]")
    attachments_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # Relationships
    channel_obj: Mapped["ChannelTable"] = relationship(
        "ChannelTable",
        back_populates="messages",
        foreign_keys=[channel_id],
    )
    author: Mapped[Optional["UserTable"]] = relationship(
        "UserTable",
        back_populates="messages",
        foreign_keys=[user_id],
    )

    def to_model(self) -> Message:
        """Convert to domain model."""
        return Message(
            ts=self.ts,
            type=self.type,
            user=self.user_id,
            text=self.text,
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            reply_count=self.reply_count,
            reply_users_count=self.reply_users_count,
            latest_reply=self.latest_reply,
            is_activity_message=self.is_activity_message,
            subtype=self.subtype,
            reactions=json.loads(self.reactions_json),
            files=json.loads(self.files_json),
            attachments=json.loads(self.attachments_json),
        )

    @classmethod
    def from_model(cls, message: Message, channel_id: str) -> "MessageTable":
        """Create from domain model."""
        return cls(
            ts=message.ts,
            channel_id=channel_id,
            type=message.type,
            user_id=message.user,
            text=message.text,
            thread_ts=message.thread_ts,
            reply_count=message.reply_count,
            reply_users_count=message.reply_users_count,
            latest_reply=message.latest_reply,
            is_activity_message=message.is_activity_message,
            subtype=message.subtype,
            reactions_json=json.dumps(message.reactions),
            files_json=json.dumps(message.files),
            attachments_json=json.dumps(message.attachments),
            created_at=ts_to_datetime(message.ts),
        )


# =============================================================================
# DATABASE MANAGER
# =============================================================================


class SlackDatabase(BaseDatabase):
    """
    Database manager for Slack MCP Server.

    Extends BaseDatabase to provide CRUD operations for all Slack entities.
    Requires PostgreSQL database via connection_string.

    Example usage:
        db = SlackDatabase(connection_string="postgresql://user:pass@localhost:5432/slack")
    """

    def __init__(self, connection_string: str, **engine_kwargs: Any):
        """
        Initialize the Slack database.

        Args:
            connection_string: PostgreSQL connection string.
            **engine_kwargs: Additional arguments to pass to SQLAlchemy's create_engine.
        """
        super().__init__(
            backend=DatabaseBackend.POSTGRESQL,
            connection_string=connection_string,
            **engine_kwargs,
        )

    def _create_tables(self) -> None:
        """Create all Slack-specific tables."""
        SlackBase.metadata.create_all(self.engine)

    # =========================================================================
    # USER OPERATIONS
    # =========================================================================

    def create_user(self, user: User) -> User:
        """Create a new user."""
        with self.get_session() as session:
            user_table = UserTable.from_model(user)
            session.add(user_table)
            session.flush()
            return user_table.to_model()

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        with self.get_session() as session:
            user_table = (
                session.query(UserTable).filter(UserTable.id == user_id).first()
            )
            return user_table.to_model() if user_table else None

    def get_user_by_name(self, name: str) -> Optional[User]:
        """Get a user by name or display_name."""
        # Remove @ prefix if present
        name = name.lstrip("@")
        with self.get_session() as session:
            user_table = (
                session.query(UserTable)
                .filter(
                    (UserTable.name == name)
                    | (UserTable.display_name == name)
                )
                .first()
            )
            return user_table.to_model() if user_table else None

    def list_users(self) -> List[User]:
        """List all users."""
        with self.get_session() as session:
            users = session.query(UserTable).all()
            return [u.to_model() for u in users]

    # =========================================================================
    # CHANNEL OPERATIONS
    # =========================================================================

    def create_channel(self, channel: Channel) -> Channel:
        """Create a new channel."""
        with self.get_session() as session:
            channel_table = ChannelTable.from_model(channel)
            session.add(channel_table)
            session.flush()
            return channel_table.to_model()

    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a channel by ID."""
        with self.get_session() as session:
            channel_table = (
                session.query(ChannelTable).filter(ChannelTable.id == channel_id).first()
            )
            return channel_table.to_model() if channel_table else None

    def get_channel_by_name(self, name: str) -> Optional[Channel]:
        """Get a channel by name."""
        # Remove # prefix if present
        name = name.lstrip("#")
        with self.get_session() as session:
            channel_table = (
                session.query(ChannelTable).filter(ChannelTable.name == name).first()
            )
            return channel_table.to_model() if channel_table else None

    def list_channels(
        self,
        channel_types: List[str],
        cursor: Optional[str] = None,
        limit: int = 100,
        sort_by_popularity: bool = False,
    ) -> Tuple[List[Channel], Optional[str]]:
        """
        List channels filtered by type.

        Args:
            channel_types: List of channel types to include
            cursor: Pagination cursor
            limit: Maximum number of results
            sort_by_popularity: If True, sort by num_members descending

        Returns:
            Tuple of (channels, next_cursor)
        """
        with self.get_session() as session:
            query = session.query(ChannelTable)

            # Build type filter
            type_filters = []
            for ct in channel_types:
                if ct == "public_channel":
                    type_filters.append(
                        (ChannelTable.is_channel == True)
                        & (ChannelTable.is_private == False)
                        & (ChannelTable.is_im == False)
                        & (ChannelTable.is_mpim == False)
                    )
                elif ct == "private_channel":
                    type_filters.append(ChannelTable.is_private == True)
                elif ct == "im":
                    type_filters.append(ChannelTable.is_im == True)
                elif ct == "mpim":
                    type_filters.append(ChannelTable.is_mpim == True)

            if type_filters:
                from sqlalchemy import or_
                query = query.filter(or_(*type_filters))

            # Apply sorting
            if sort_by_popularity:
                query = query.order_by(desc(ChannelTable.num_members), ChannelTable.id)
            else:
                query = query.order_by(ChannelTable.id)

            # Apply cursor pagination
            if cursor:
                query = query.filter(ChannelTable.id > cursor)

            # Get one extra to check for more
            channels = query.limit(limit + 1).all()

            has_more = len(channels) > limit
            if has_more:
                channels = channels[:limit]

            next_cursor = channels[-1].id if has_more and channels else None

            return [c.to_model() for c in channels], next_cursor

    # =========================================================================
    # MESSAGE OPERATIONS
    # =========================================================================

    def create_message(self, message: Message, channel_id: str) -> Message:
        """Create a new message."""
        with self.get_session() as session:
            message_table = MessageTable.from_model(message, channel_id)
            session.add(message_table)
            session.flush()
            return message_table.to_model()

    def get_message(self, channel_id: str, ts: str) -> Optional[Message]:
        """Get a message by channel and timestamp."""
        with self.get_session() as session:
            message_table = (
                session.query(MessageTable)
                .filter(
                    MessageTable.channel_id == channel_id,
                    MessageTable.ts == ts,
                )
                .first()
            )
            return message_table.to_model() if message_table else None

    def get_conversation_history(
        self,
        channel_id: str,
        cursor: Optional[str] = None,
        limit: str = "1d",
        include_activity_messages: bool = False,
    ) -> Tuple[List[Message], Optional[str]]:
        """
        Get message history for a channel.

        Args:
            channel_id: Channel ID
            cursor: Pagination cursor (message ts)
            limit: Time window (1d, 1w, 30d, 90d) or numeric count as string
            include_activity_messages: Include join/leave/topic messages

        Returns:
            Tuple of (messages, next_cursor)
        """
        with self.get_session() as session:
            query = session.query(MessageTable).filter(
                MessageTable.channel_id == channel_id
            )

            # Filter activity messages unless requested
            if not include_activity_messages:
                query = query.filter(MessageTable.is_activity_message == False)

            # Apply time-based or count-based limit
            now = datetime.now()
            count_limit = 100  # Default

            if limit.endswith("d"):
                days = int(limit[:-1])
                start_time = now - timedelta(days=days)
                query = query.filter(MessageTable.created_at >= start_time)
            elif limit.endswith("w"):
                weeks = int(limit[:-1])
                start_time = now - timedelta(weeks=weeks)
                query = query.filter(MessageTable.created_at >= start_time)
            elif limit.isdigit():
                count_limit = int(limit)

            # Sort by ts descending (newest first)
            query = query.order_by(desc(MessageTable.ts))

            # Apply cursor pagination
            if cursor:
                query = query.filter(MessageTable.ts < cursor)

            # Get messages
            messages = query.limit(count_limit + 1).all()

            has_more = len(messages) > count_limit
            if has_more:
                messages = messages[:count_limit]

            next_cursor = messages[-1].ts if has_more and messages else None

            return [m.to_model() for m in messages], next_cursor

    def get_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
        cursor: Optional[str] = None,
        limit: str = "1d",
        include_activity_messages: bool = False,
    ) -> Tuple[List[Message], Optional[str]]:
        """
        Get thread replies.

        Args:
            channel_id: Channel ID
            thread_ts: Parent thread timestamp
            cursor: Pagination cursor
            limit: Time window or count
            include_activity_messages: Include activity messages

        Returns:
            Tuple of (messages, next_cursor)
        """
        with self.get_session() as session:
            # Get parent message first
            parent = (
                session.query(MessageTable)
                .filter(
                    MessageTable.channel_id == channel_id,
                    MessageTable.ts == thread_ts,
                )
                .first()
            )

            # Get replies
            query = session.query(MessageTable).filter(
                MessageTable.channel_id == channel_id,
                MessageTable.thread_ts == thread_ts,
            )

            if not include_activity_messages:
                query = query.filter(MessageTable.is_activity_message == False)

            # Sort by ts ascending (oldest first for threads)
            query = query.order_by(asc(MessageTable.ts))

            # Apply cursor
            if cursor:
                query = query.filter(MessageTable.ts > cursor)

            # Apply limit
            count_limit = 100
            if limit.isdigit():
                count_limit = int(limit)

            messages = query.limit(count_limit + 1).all()

            has_more = len(messages) > count_limit
            if has_more:
                messages = messages[:count_limit]

            next_cursor = messages[-1].ts if has_more and messages else None

            # Prepend parent if no cursor (first page)
            result = []
            if parent and not cursor:
                result.append(parent.to_model())
            result.extend([m.to_model() for m in messages])

            return result, next_cursor

    def search_messages(
        self,
        search_query: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 20,
        filter_date_on: Optional[str] = None,
        filter_date_during: Optional[str] = None,
        filter_date_after: Optional[str] = None,
        filter_date_before: Optional[str] = None,
        filter_in_channel: Optional[str] = None,
        filter_in_im_or_mpim: Optional[str] = None,
        filter_threads_only: bool = False,
        filter_users_from: Optional[str] = None,
        filter_users_with: Optional[str] = None,
    ) -> Tuple[List[Message], Optional[str]]:
        """
        Search messages with various filters.

        Args:
            search_query: Text to search for (or Slack URL)
            cursor: Pagination cursor
            limit: Max results (1-100, default 20)
            filter_date_on: Exact date (YYYY-MM-DD)
            filter_date_during: Calendar period (e.g., "July")
            filter_date_after: After date
            filter_date_before: Before date
            filter_in_channel: Channel ID or #name
            filter_in_im_or_mpim: IM/MPIM ID or @username_dm
            filter_threads_only: Only return thread messages
            filter_users_from: Filter by sender
            filter_users_with: Filter by participant

        Returns:
            Tuple of (messages, next_cursor)
        """
        with self.get_session() as session:
            # Check if search_query is a Slack URL
            if search_query and self._is_slack_url(search_query):
                message = self._get_message_from_url(session, search_query)
                if message:
                    return [message.to_model()], None
                return [], None

            query = session.query(MessageTable)

            # Text search
            if search_query:
                query = query.filter(
                    MessageTable.text.ilike(f"%{search_query}%")
                )

            # Date filters (precedence: on > during > after/before)
            if filter_date_on:
                date = self._parse_date(filter_date_on)
                if date:
                    next_day = date + timedelta(days=1)
                    query = query.filter(
                        MessageTable.created_at >= date,
                        MessageTable.created_at < next_day,
                    )
            elif filter_date_during:
                start, end = self._parse_date_period(filter_date_during)
                if start and end:
                    query = query.filter(
                        MessageTable.created_at >= start,
                        MessageTable.created_at < end,
                    )
            else:
                if filter_date_after:
                    date = self._parse_date(filter_date_after)
                    if date:
                        query = query.filter(MessageTable.created_at > date)
                if filter_date_before:
                    date = self._parse_date(filter_date_before)
                    if date:
                        query = query.filter(MessageTable.created_at < date)

            # Channel filter
            if filter_in_channel:
                channel_id = self._resolve_channel_id(session, filter_in_channel)
                if channel_id:
                    query = query.filter(MessageTable.channel_id == channel_id)

            # IM/MPIM filter
            if filter_in_im_or_mpim:
                channel_id = self._resolve_channel_id(session, filter_in_im_or_mpim)
                if channel_id:
                    query = query.filter(MessageTable.channel_id == channel_id)

            # Thread filter
            if filter_threads_only:
                query = query.filter(MessageTable.thread_ts.isnot(None))

            # User from filter
            if filter_users_from:
                user_id = self._resolve_user_id(session, filter_users_from)
                if user_id:
                    query = query.filter(MessageTable.user_id == user_id)

            # Sort by ts descending
            query = query.order_by(desc(MessageTable.ts))

            # Cursor pagination
            if cursor:
                query = query.filter(MessageTable.ts < cursor)

            # Limit
            limit = min(max(limit, 1), 100)
            messages = query.limit(limit + 1).all()

            has_more = len(messages) > limit
            if has_more:
                messages = messages[:limit]

            next_cursor = messages[-1].ts if has_more and messages else None

            return [m.to_model() for m in messages], next_cursor

    def _is_slack_url(self, text: str) -> bool:
        """Check if text is a Slack message URL."""
        return bool(re.match(r"https://[^/]+\.slack\.com/archives/", text))

    def _get_message_from_url(self, session, url: str) -> Optional[MessageTable]:
        """Extract and fetch message from Slack URL."""
        # URL format: https://workspace.slack.com/archives/C12345/p1234567890123456
        match = re.search(r"/archives/([^/]+)/p(\d+)", url)
        if match:
            channel_id = match.group(1)
            ts_raw = match.group(2)
            # Convert p format to ts format (insert decimal)
            ts = f"{ts_raw[:10]}.{ts_raw[10:]}"
            return (
                session.query(MessageTable)
                .filter(
                    MessageTable.channel_id == channel_id,
                    MessageTable.ts == ts,
                )
                .first()
            )
        return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        date_str = date_str.lower().strip()

        # Handle natural language
        if date_str == "today":
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == "yesterday":
            return (datetime.now() - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        # Try YYYY-MM-DD format
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass

        return None

    def _parse_date_period(self, period: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Parse a calendar period string to start/end datetimes."""
        period = period.lower().strip()
        now = datetime.now()

        # Month names
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }

        if period in months:
            month = months[period]
            year = now.year
            # If the month is in the future, use last year
            if month > now.month:
                year -= 1
            start = datetime(year, month, 1)
            # Get end of month
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
            return start, end

        return None, None

    def _resolve_channel_id(self, session, identifier: str) -> Optional[str]:
        """Resolve channel name or ID to channel ID."""
        # If it looks like an ID, return it
        if identifier.startswith(("C", "D", "G")):
            return identifier

        # Otherwise, search by name
        name = identifier.lstrip("#")
        channel = session.query(ChannelTable).filter(ChannelTable.name == name).first()
        return channel.id if channel else None

    def _resolve_user_id(self, session, identifier: str) -> Optional[str]:
        """Resolve username or ID to user ID."""
        # If it looks like an ID, return it
        if identifier.startswith("U"):
            return identifier

        # Otherwise, search by name
        name = identifier.lstrip("@")
        user = (
            session.query(UserTable)
            .filter((UserTable.name == name) | (UserTable.display_name == name))
            .first()
        )
        return user.id if user else None

    def update_thread_parent(self, channel_id: str, thread_ts: str) -> None:
        """Update thread parent's reply_count and latest_reply."""
        with self.get_session() as session:
            parent = (
                session.query(MessageTable)
                .filter(
                    MessageTable.channel_id == channel_id,
                    MessageTable.ts == thread_ts,
                )
                .first()
            )
            if parent:
                # Count replies
                reply_count = (
                    session.query(MessageTable)
                    .filter(
                        MessageTable.channel_id == channel_id,
                        MessageTable.thread_ts == thread_ts,
                        MessageTable.ts != thread_ts,
                    )
                    .count()
                )

                # Get latest reply
                latest = (
                    session.query(MessageTable)
                    .filter(
                        MessageTable.channel_id == channel_id,
                        MessageTable.thread_ts == thread_ts,
                        MessageTable.ts != thread_ts,
                    )
                    .order_by(desc(MessageTable.ts))
                    .first()
                )

                parent.reply_count = reply_count
                parent.latest_reply = latest.ts if latest else None

                # Count unique reply users
                reply_users = (
                    session.query(MessageTable.user_id)
                    .filter(
                        MessageTable.channel_id == channel_id,
                        MessageTable.thread_ts == thread_ts,
                        MessageTable.ts != thread_ts,
                        MessageTable.user_id.isnot(None),
                    )
                    .distinct()
                    .count()
                )
                parent.reply_users_count = reply_users

                session.flush()

    # =========================================================================
    # RESET & DATA LOADING (BaseDatabase implementation)
    # =========================================================================

    def reset_all(self) -> None:
        """Drop all tables and recreate them."""
        SlackBase.metadata.drop_all(self.engine)
        SlackBase.metadata.create_all(self.engine)

    def load_from_json(self, json_path: str) -> Dict[str, int]:
        """
        Load mock data from a JSON file.

        Expected JSON structure:
        {
            "users": [...],
            "channels": [...],
            "messages": [...]
        }

        Returns:
            Dict with counts of loaded entities.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        counts = {"users": 0, "channels": 0, "messages": 0}

        with self.get_session() as session:
            # Load users first (channels and messages reference them)
            for user_data in data.get("users", []):
                user = User(
                    id=user_data.get("id", f"U{generate_slack_ts().replace('.', '')[:10].upper()}"),
                    name=user_data.get("name", ""),
                    real_name=user_data.get("real_name", ""),
                    display_name=user_data.get("display_name", ""),
                    email=user_data.get("email"),
                    avatar_url=user_data.get("avatar_url"),
                    is_bot=user_data.get("is_bot", False),
                    is_admin=user_data.get("is_admin", False),
                    deleted=user_data.get("deleted", False),
                    timezone=user_data.get("timezone", "America/Los_Angeles"),
                    status=user_data.get("status", ""),
                    title=user_data.get("title", ""),
                )
                session.add(UserTable.from_model(user))
                counts["users"] += 1

            # Flush users before loading channels (FK constraint)
            session.flush()

            # Load channels (messages reference them)
            for channel_data in data.get("channels", []):
                channel = Channel(
                    id=channel_data.get("id", f"C{generate_slack_ts().replace('.', '')[:10].upper()}"),
                    name=channel_data.get("name", ""),
                    is_channel=channel_data.get("is_channel", True),
                    is_private=channel_data.get("is_private", False),
                    is_im=channel_data.get("is_im", False),
                    is_mpim=channel_data.get("is_mpim", False),
                    is_archived=channel_data.get("is_archived", False),
                    is_general=channel_data.get("is_general", False),
                    num_members=channel_data.get("num_members", 0),
                    topic=channel_data.get("topic", ""),
                    purpose=channel_data.get("purpose", ""),
                    created=channel_data.get("created", generate_slack_ts()),
                    creator=channel_data.get("creator"),
                    members=channel_data.get("members", []),
                )
                session.add(ChannelTable.from_model(channel))
                counts["channels"] += 1

            # Flush channels before loading messages (FK constraint)
            session.flush()

            # Load messages
            for message_data in data.get("messages", []):
                channel_id = message_data.get("channel") or message_data.get("channel_id")
                if not channel_id:
                    continue

                message = Message(
                    ts=message_data.get("ts", generate_slack_ts()),
                    type=message_data.get("type", "message"),
                    user=message_data.get("user") or message_data.get("user_id"),
                    text=message_data.get("text", ""),
                    channel=channel_id,
                    thread_ts=message_data.get("thread_ts"),
                    reply_count=message_data.get("reply_count", 0),
                    reply_users_count=message_data.get("reply_users_count", 0),
                    latest_reply=message_data.get("latest_reply"),
                    is_activity_message=message_data.get("is_activity_message", False),
                    subtype=message_data.get("subtype"),
                    reactions=message_data.get("reactions", []),
                    files=message_data.get("files", []),
                    attachments=message_data.get("attachments", []),
                )
                session.add(MessageTable.from_model(message, channel_id))
                counts["messages"] += 1

        return counts

    def export_state(self) -> Dict[str, Any]:
        """Export in the same format load_from_json expects (not API response format)."""
        with self.get_session() as session:
            users = []
            for row in session.query(UserTable).all():
                m = row.to_model()
                users.append({"id": m.id, "name": m.name, "real_name": m.real_name, "display_name": m.display_name, "email": m.email, "avatar_url": m.avatar_url, "is_bot": m.is_bot, "is_admin": m.is_admin, "deleted": m.deleted, "timezone": m.timezone, "status": m.status, "title": m.title})
            channels = []
            for row in session.query(ChannelTable).all():
                m = row.to_model()
                channels.append({"id": m.id, "name": m.name, "is_channel": m.is_channel, "is_private": m.is_private, "is_im": m.is_im, "is_mpim": m.is_mpim, "is_archived": m.is_archived, "is_general": m.is_general, "num_members": m.num_members, "topic": m.topic, "purpose": m.purpose, "created": m.created, "creator": m.creator, "members": m.members})
            messages = []
            for row in session.query(MessageTable).all():
                m = row.to_model()
                messages.append({"ts": m.ts, "type": m.type, "user": m.user, "text": m.text, "channel": m.channel, "thread_ts": m.thread_ts, "reply_count": m.reply_count, "reply_users_count": m.reply_users_count, "latest_reply": m.latest_reply, "is_activity_message": m.is_activity_message, "subtype": m.subtype, "reactions": m.reactions, "files": m.files, "attachments": m.attachments})
        return {"users": users, "channels": channels, "messages": messages}

