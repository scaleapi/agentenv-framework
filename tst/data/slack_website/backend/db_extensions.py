"""
Extended database operations for Slack Workspace Manager.

This module extends the MCP server's SlackDatabase class without modifying
the original source code. All additional operations needed by the web GUI
should be added here.
"""

import sys
import os
import json
from typing import Optional, List, Tuple, Any

# Set up MCP server path for imports
mcp_path = os.getenv("MCP_SERVER_PATH", "/mcp")
if mcp_path not in sys.path:
    sys.path.insert(0, mcp_path)

# Ensure BaseService is also in path
base_service_path = os.path.abspath(os.path.join(mcp_path, "..", "BaseService"))
if os.path.exists(base_service_path) and base_service_path not in sys.path:
    sys.path.insert(0, base_service_path)

# Import from MCP server (DO NOT MODIFY these source files)
try:
    from database import SlackDatabase, UserTable, ChannelTable, MessageTable
    from models import (
        User,
        Channel,
        Message,
        ChannelType,
        MessageType,
        generate_slack_ts,
        ts_to_datetime,
        datetime_to_ts,
    )
    IMPORTS_AVAILABLE = True
    TABLES_AVAILABLE = True
    print("Successfully imported MCP server components")
except ImportError as e:
    print(f"Warning: Failed to import MCP server components: {e}")
    print(f"MCP_SERVER_PATH: {mcp_path}")
    print(f"sys.path: {sys.path}")
    
    # Create fallback classes for testing or when MCP is not available
    class MockSlackDatabase:
        def __init__(self, connection_string: str, **kwargs):
            self.connection_string = connection_string
            print(f"MockSlackDatabase initialized with connection_string={connection_string}")
        
        def list_users(self):
            return []
        
        def get_user(self, user_id):
            return None
        
        def list_channels(self, **kwargs):
            return [], None
        
        def get_channel(self, channel_id):
            return None
        
        def get_conversation_history(self, **kwargs):
            return [], None
        
        def get_thread_replies(self, **kwargs):
            return [], None
        
        def search_messages(self, **kwargs):
            return [], None
        
        def create_message(self, message, channel_id):
            return message
        
        def load_from_json(self, path):
            return {"users": 0, "channels": 0, "messages": 0}
        
        def reset_all(self):
            pass
    
    class MockUser:
        def __init__(self, id, name="Mock User", **kwargs):
            self.id = id
            self.name = name
            self.real_name = name
            self.display_name = name
            self.email = None
            self.avatar_url = None
            self.is_bot = False
            self.is_admin = False
            self.deleted = False
            self.timezone = "UTC"
    
    class MockChannel:
        def __init__(self, id, name="Mock Channel", **kwargs):
            self.id = id
            self.name = name
            self.is_channel = True
            self.is_private = False
            self.is_im = False
            self.is_mpim = False
            self.is_archived = False
            self.is_general = False
            self.num_members = 0
            self.topic = ""
            self.purpose = ""
            self.created = "1609459200"  # Jan 1, 2021
            self.creator = None
            self.members = []
    
    class MockMessage:
        def __init__(self, **kwargs):
            self.ts = kwargs.get('ts', '1609459200.000000')
            self.type = kwargs.get('type', 'message')
            self.user = kwargs.get('user')
            self.text = kwargs.get('text', '')
            self.channel = kwargs.get('channel')
            self.thread_ts = kwargs.get('thread_ts')
            self.reply_count = 0
            self.reply_users_count = 0
            self.latest_reply = None
            self.is_activity_message = False
            self.subtype = None
            self.reactions = []
            self.files = []
            self.attachments = []
    
    def generate_slack_ts():
        import time
        return f"{time.time():.6f}"
    
    # Use mock classes
    SlackDatabase = MockSlackDatabase
    User = MockUser
    Channel = MockChannel
    Message = MockMessage
    IMPORTS_AVAILABLE = False
    TABLES_AVAILABLE = False

# Constants for current user (John Doe)
CURRENT_USER_ID = "U1234567890"


class ExtendedDatabase(SlackDatabase):
    """
    Extends SlackDatabase with additional operations needed by the web GUI.
    
    The parent class SlackDatabase provides:
    - User operations: create_user, get_user, get_user_by_name, list_users
    - Channel operations: create_channel, get_channel, get_channel_by_name, list_channels
    - Message operations: create_message, get_message, get_conversation_history, 
                          get_thread_replies, search_messages
    - Admin operations: reset_all, load_from_json
    
    This class adds additional methods without modifying the MCP server code.
    """

    def __init__(self, connection_string: str, **kwargs):
        """
        CRITICAL FIX: Initialize with PostgreSQL connection string.
        
        Args:
            connection_string: PostgreSQL connection string (NOT db_path)
            **kwargs: Additional arguments for parent class
        """
        print(f"ExtendedDatabase.__init__ called with connection_string={connection_string}")
        
        try:
            # CRITICAL: Pass connection_string to parent class (not db_path)
            super().__init__(connection_string=connection_string, **kwargs)
        except Exception as e:
            print(f"Error initializing database: {e}")
            if not IMPORTS_AVAILABLE:
                # Use mock database if real imports failed
                pass
            else:
                raise

    def update_thread_parent(self, channel_id: str, thread_ts: str) -> None:
        """
        Update the parent message of a thread when a reply is added.
        
        This is a best-effort operation - if the parent class doesn't support
        this, we silently skip it as thread metadata is not critical.
        
        Args:
            channel_id: Channel containing the thread
            thread_ts: Timestamp of the parent message
        """
        # Check if parent class has this method
        if hasattr(super(), 'update_thread_parent'):
            try:
                super().update_thread_parent(channel_id, thread_ts)
            except Exception:
                pass  # Ignore errors in thread metadata updates

    def create_new_message(
        self,
        channel_id: str,
        text: str,
        user_id: str = CURRENT_USER_ID,
        thread_ts: Optional[str] = None
    ) -> Any:  # Return type is Message but use Any for compatibility
        """
        Create a new message in a channel with auto-generated timestamp.
        
        Args:
            channel_id: Channel to post to
            text: Message content
            user_id: User ID of sender (defaults to John Doe)
            thread_ts: Optional parent thread timestamp for replies
            
        Returns:
            Created Message object
        """
        # Generate Slack-style timestamp
        ts = generate_slack_ts()
        
        # Create the message object
        message = Message(
            ts=ts,
            type="message",
            user=user_id,
            text=text,
            channel=channel_id,
            thread_ts=thread_ts,
        )
        
        # Use parent class method to persist
        try:
            created_message = self.create_message(message, channel_id)
        except Exception as e:
            print(f"Error creating message: {e}")
            # Return the original message if creation fails
            created_message = message
        
        # If this is a thread reply, update the parent message
        if thread_ts:
            self.update_thread_parent(channel_id, thread_ts)
        
        return created_message

    def list_channels_for_user(
        self,
        user_id: str = CURRENT_USER_ID,
        channel_types: Optional[List[str]] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
        sort_by_popularity: bool = False
    ) -> Tuple[List[Any], Optional[str]]:  # Use Any for Channel compatibility
        """
        List channels where a specific user is a member.
        
        Args:
            user_id: User ID to filter by membership
            channel_types: List of channel types to include
            cursor: Pagination cursor
            limit: Maximum number of results
            sort_by_popularity: Sort by member count
            
        Returns:
            Tuple of (channels, next_cursor)
        """
        if channel_types is None:
            channel_types = ["public_channel", "private_channel", "im", "mpim"]
        
        # Get all channels of requested types
        try:
            channels, next_cursor = self.list_channels(
                channel_types=channel_types,
                cursor=cursor,
                limit=limit,
                sort_by_popularity=sort_by_popularity
            )
        except Exception as e:
            print(f"Error listing channels: {e}")
            return [], None
        
        # Filter to only channels where user is a member
        filtered_channels = [
            ch for ch in channels
            if hasattr(ch, 'members') and user_id in ch.members
        ]
        
        return filtered_channels, next_cursor

    def get_current_user(self) -> Optional[Any]:  # Use Any for User compatibility
        """
        Get the current user (John Doe).
        
        Returns:
            User object for John Doe, or None if not found
        """
        try:
            user = self.get_user(CURRENT_USER_ID)
            if user:
                return user
            # Fall back to first user in DB (for generated data that doesn't include the hardcoded ID)
            users = self.list_users()
            return users[0] if users else None
        except Exception:
            return None

    def get_stats(self) -> dict:
        """
        Get database statistics for the health dashboard.
        
        Returns:
            Dict with counts of users, channels, and messages.
        """
        try:
            users = self.list_users()
            channels, _ = self.list_channels(
                channel_types=["public_channel", "private_channel", "im", "mpim"],
                cursor=None,
                limit=999,
                sort_by_popularity=False
            )
            # Get message count by searching with no filters
            messages, _ = self.search_messages(
                search_query=None,
                cursor=None,
                limit=100,
                filter_date_on=None,
                filter_date_during=None,
                filter_date_after=None,
                filter_date_before=None,
                filter_in_channel=None,
                filter_in_im_or_mpim=None,
                filter_threads_only=False,
                filter_users_from=None,
                filter_users_with=None
            )
            
            return {
                "users": len(users) if users else 0,
                "channels": len(channels) if channels else 0,
                "messages": len(messages) if messages else 0
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"users": 0, "channels": 0, "messages": 0}

    def get_user_channels(self, user_id: str) -> List[Any]:  # Use Any for Channel compatibility
        """
        Get all channels where a user is a member.
        
        Args:
            user_id: The user's ID
            
        Returns:
            List of Channel objects where user is a member
        """
        try:
            all_channels, _ = self.list_channels(
                channel_types=["public_channel", "private_channel", "im", "mpim"],
                cursor=None,
                limit=999,
                sort_by_popularity=False
            )
            return [ch for ch in all_channels if hasattr(ch, 'members') and user_id in ch.members]
        except Exception:
            return []

    def get_dm_for_user(self, user_id: str) -> List[Any]:  # Use Any for Channel compatibility
        """
        Get DM channels involving a specific user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            List of DM Channel objects involving the user
        """
        try:
            dm_channels, _ = self.list_channels(
                channel_types=["im", "mpim"],
                cursor=None,
                limit=999,
                sort_by_popularity=False
            )
            return [ch for ch in dm_channels if hasattr(ch, 'members') and user_id in ch.members]
        except Exception:
            return []

    def get_channel_members_with_details(self, channel_id: str) -> List[Any]:  # Use Any for User compatibility
        """
        Get the members of a channel with their full user details.
        
        Args:
            channel_id: The channel ID
            
        Returns:
            List of User objects for each member
        """
        try:
            channel = self.get_channel(channel_id)
            if not channel or not hasattr(channel, 'members'):
                return []
            
            members = []
            for member_id in channel.members:
                user = self.get_user(member_id)
                if user:
                    members.append(user)
            return members
        except Exception:
            return []

    def update_current_user_profile(
        self,
        full_name: Optional[str] = None,
        display_name: Optional[str] = None,
        username: Optional[str] = None,
        status: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Update editable current-user profile fields.

        Note: The underlying user model has no dedicated status field. We persist
        status in timezone as a best-effort placeholder so status can round-trip.
        """
        if not TABLES_AVAILABLE:
            return self.get_current_user()

        with self.get_session() as session:
            user_row = (
                session.query(UserTable)
                .filter(UserTable.id == CURRENT_USER_ID)
                .first()
            )
            if not user_row:
                return None

            if full_name is not None:
                user_row.real_name = full_name
            if display_name is not None:
                user_row.display_name = display_name
            if username is not None:
                user_row.name = username
            if status is not None:
                user_row.timezone = status
            if avatar_url is not None:
                user_row.avatar_url = avatar_url

            session.flush()
            return user_row.to_model()

    def open_or_create_dm(self, user_ids: List[str]) -> Optional[Any]:
        """Open an existing IM/MPIM or create one for the provided recipients."""
        if not user_ids:
            return None

        participant_ids = sorted(set([CURRENT_USER_ID, *user_ids]))
        is_group = len(participant_ids) > 2

        # Reuse an existing DM/group DM with exactly the same members.
        channels, _ = self.list_channels(
            channel_types=["im", "mpim"],
            cursor=None,
            limit=999,
            sort_by_popularity=False,
        )
        for channel in channels:
            if sorted(channel.members) == participant_ids:
                return channel

        # Build a deterministic display name from recipients (excluding current user).
        recipient_ids = [uid for uid in participant_ids if uid != CURRENT_USER_ID]
        recipient_names: List[str] = []
        for recipient_id in recipient_ids:
            user = self.get_user(recipient_id)
            if user:
                recipient_names.append(user.display_name or user.real_name or user.name or recipient_id)
            else:
                recipient_names.append(recipient_id)

        channel_id_prefix = "G" if is_group else "D"
        channel_id = f"{channel_id_prefix}{generate_slack_ts().replace('.', '')[:10].upper()}"
        channel = Channel(
            id=channel_id,
            name=", ".join(recipient_names),
            is_channel=False,
            is_private=True,
            is_im=not is_group,
            is_mpim=is_group,
            is_archived=False,
            is_general=False,
            num_members=len(participant_ids),
            topic="",
            purpose="",
            created=generate_slack_ts(),
            creator=CURRENT_USER_ID,
            members=participant_ids,
        )
        return self.create_channel(channel)

    def update_message_text(self, channel_id: str, message_ts: str, new_text: str) -> Optional[Any]:
        """Edit a message authored by the current user."""
        if not TABLES_AVAILABLE:
            return None

        with self.get_session() as session:
            row = (
                session.query(MessageTable)
                .filter(
                    MessageTable.channel_id == channel_id,
                    MessageTable.ts == message_ts,
                )
                .first()
            )
            if not row or row.user_id != CURRENT_USER_ID:
                return None

            row.text = new_text
            # Reuse subtype for edited indicator.
            row.subtype = "edited"
            session.flush()
            return row.to_model()

    def delete_message_by_ts(self, channel_id: str, message_ts: str) -> bool:
        """Delete a message authored by the current user."""
        if not TABLES_AVAILABLE:
            return False

        with self.get_session() as session:
            row = (
                session.query(MessageTable)
                .filter(
                    MessageTable.channel_id == channel_id,
                    MessageTable.ts == message_ts,
                )
                .first()
            )
            if not row or row.user_id != CURRENT_USER_ID:
                return False

            parent_thread_ts = row.thread_ts
            session.delete(row)
            session.flush()

            if parent_thread_ts and parent_thread_ts != message_ts:
                # Keep parent metadata accurate after reply deletion.
                self.update_thread_parent(channel_id, parent_thread_ts)

            return True

    def toggle_message_reaction(self, channel_id: str, message_ts: str, reaction_name: str) -> Optional[Any]:
        """Toggle current user's participation in a reaction on a message."""
        if not TABLES_AVAILABLE:
            return None

        with self.get_session() as session:
            row = (
                session.query(MessageTable)
                .filter(
                    MessageTable.channel_id == channel_id,
                    MessageTable.ts == message_ts,
                )
                .first()
            )
            if not row:
                return None

            reactions = json.loads(row.reactions_json or "[]")
            normalized_name = reaction_name.strip()
            if not normalized_name:
                return row.to_model()

            existing = None
            for reaction in reactions:
                if reaction.get("name") == normalized_name:
                    existing = reaction
                    break

            if existing is None:
                reactions.append(
                    {
                        "name": normalized_name,
                        "count": 1,
                        "users": [CURRENT_USER_ID],
                    }
                )
            else:
                users = existing.get("users") or []
                if CURRENT_USER_ID in users:
                    users = [uid for uid in users if uid != CURRENT_USER_ID]
                else:
                    users.append(CURRENT_USER_ID)

                if not users:
                    reactions = [reaction for reaction in reactions if reaction.get("name") != normalized_name]
                else:
                    existing["users"] = users
                    existing["count"] = len(users)

            row.reactions_json = json.dumps(reactions)
            session.flush()
            return row.to_model()

    def is_available(self) -> bool:
        """
        Check if the database is available and working.
        
        Returns:
            True if database is operational, False otherwise
        """
        try:
            # Try to list users as a simple health check
            self.list_users()
            return True
        except Exception as e:
            print(f"Database availability check failed: {e}")
            return False