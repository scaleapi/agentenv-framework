"""
API route definitions for Slack Workspace Manager.
"""

import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List

from schemas import (
    # User schemas
    UserResponse,
    CurrentUserResponse,
    UpdateCurrentUserRequest,
    UserListResponse,
    SingleUserResponse,
    # Channel schemas
    ChannelResponse,
    ChannelListResponse,
    SingleChannelResponse,
    ChannelMembersResponse,
    ResponseMetadata,
    # Message schemas
    MessageResponse,
    MessageListResponse,
    CreateMessageRequest,
    CreateMessageResponse,
    UpdateMessageRequest,
    ToggleReactionRequest,
    OpenDMRequest,
    OpenDMResponse,
    SearchMessagesResponse,
    # Data management schemas
    ResetRequest,
    ResetResponse,
    # Health schemas
    HealthResponse,
    StatsResponse,
    # Error schemas
    ErrorResponse,
)
from db_extensions import ExtendedDatabase, CURRENT_USER_ID

router = APIRouter()

# Global database instance (set by app.py during startup)
_db: Optional[ExtendedDatabase] = None


def get_db() -> ExtendedDatabase:
    """Get database instance."""
    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return _db


def set_db(db: ExtendedDatabase) -> None:
    """Set database instance (called during app startup)."""
    global _db
    _db = db


def slack_error_response(status_code: int, error_code: str, warning: Optional[str] = None):
    """Return Slack API-compatible error response."""
    content = {
        "ok": False,
        "error": error_code
    }
    if warning:
        content["warning"] = warning
    
    return JSONResponse(status_code=status_code, content=content)


# ============================================================================
# Health & System Endpoints
# ============================================================================

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    # Return healthy even if DB is not yet initialized - this allows container to start
    # The health check's purpose is to verify the server is responding
    if _db is None:
        return HealthResponse(status="ok", database="initializing")
    
    try:
        # Try to list users to verify database is working
        _db.list_users()
        return HealthResponse(status="ok", database="connected")
    except Exception as e:
        # Still return 200 but indicate degraded state
        return HealthResponse(status="ok", database=f"degraded: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get database statistics."""
    try:
        db = get_db()
        stats = db.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/add", response_model=ResetResponse)
def add_data(request: dict):
    """Load additional data from a JSON file into the database without resetting."""
    file_path = request.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    try:
        counts = get_db().load_from_json(file_path)
        return ResetResponse(ok=True, loaded=counts)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Invalid path: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/reset", response_model=ResetResponse)
def reset_database(request: ResetRequest = None):
    """Reset database and optionally reload mock data."""
    try:
        db = get_db()
        
        # Reset database
        db.reset_all()
        
        # Determine mock data path
        mock_path = None
        if request:
            if request.mock_data_path:
                # Explicit path provided
                mock_path = request.mock_data_path
            elif request.load_mock_data:
                # Use default mock data location
                mcp_path = os.getenv("MCP_SERVER_PATH", "/mcp")
                default_path = os.path.join(mcp_path, "sample_data.json")
                if os.path.exists(default_path):
                    mock_path = default_path
                else:
                    # Also try relative to current directory
                    alt_path = os.path.join(os.path.dirname(__file__), "..", "..", "sample_data.json")
                    if os.path.exists(alt_path):
                        mock_path = alt_path
        
        # Load mock data if path determined
        if mock_path:
            try:
                counts = db.load_from_json(mock_path)
                return ResetResponse(ok=True, loaded=counts)
            except FileNotFoundError:
                raise HTTPException(status_code=400, detail=f"Invalid path: {mock_path}")
        
        return ResetResponse(ok=True, loaded={"users": 0, "channels": 0, "messages": 0})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ============================================================================
# Current User Endpoint
# ============================================================================

@router.get("/me", response_model=CurrentUserResponse)
def get_current_user():
    """Get the current logged-in user (John Doe)."""
    try:
        db = get_db()
        user = db.get_current_user()
        if not user:
            return slack_error_response(404, "user_not_found")
        
        return CurrentUserResponse(
            ok=True,
            user=UserResponse(
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
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.patch("/me", response_model=CurrentUserResponse)
def update_current_user(request: UpdateCurrentUserRequest):
    """Update editable profile fields for the current user."""
    try:
        db = get_db()
        user = db.update_current_user_profile(
            full_name=request.full_name,
            display_name=request.display_name,
            username=request.username,
            status=request.status,
            avatar_url=request.avatar_url,
        )
        if not user:
            return slack_error_response(404, "user_not_found")

        return CurrentUserResponse(
            ok=True,
            user=UserResponse(
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
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ============================================================================
# User Endpoints
# ============================================================================

@router.get("/users", response_model=UserListResponse)
def list_users(
    include_bots: bool = Query(True, description="Include bot users"),
    include_deleted: bool = Query(False, description="Include deleted users"),
    admins_only: bool = Query(False, description="Include only admin users"),
    search: Optional[str] = Query(None, description="Search by name/display name/real name"),
):
    """List users with optional filtering."""
    try:
        db = get_db()
        users = db.list_users()
        if not include_bots:
            users = [user for user in users if not user.is_bot]
        if not include_deleted:
            users = [user for user in users if not user.deleted]
        if admins_only:
            users = [user for user in users if user.is_admin]
        if search:
            needle = search.lower()
            users = [
                user
                for user in users
                if needle in (user.name or "").lower()
                or needle in (user.display_name or "").lower()
                or needle in (user.real_name or "").lower()
            ]

        return UserListResponse(
            ok=True,
            users=[
                UserResponse(
                    id=u.id,
                    name=u.name,
                    real_name=u.real_name,
                    display_name=u.display_name,
                    email=u.email,
                    avatar_url=u.avatar_url,
                    is_bot=u.is_bot,
                    is_admin=u.is_admin,
                    deleted=u.deleted,
                    timezone=u.timezone,
                )
                for u in users
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/channels/dm", response_model=OpenDMResponse)
def open_or_create_dm(request: OpenDMRequest):
    """Open an existing DM/MPIM or create one with selected recipients."""
    try:
        db = get_db()
        channel = db.open_or_create_dm(request.user_ids)
        if not channel:
            return slack_error_response(400, "invalid_user_ids")

        return OpenDMResponse(
            ok=True,
            channel=ChannelResponse(
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
                created=channel.created,
                creator=channel.creator,
                members=channel.members,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/users/{user_id}", response_model=SingleUserResponse)
def get_user(user_id: str):
    """Get user by ID."""
    try:
        db = get_db()
        user = db.get_user(user_id)
        if not user:
            return slack_error_response(404, "user_not_found")
        
        return SingleUserResponse(
            ok=True,
            user=UserResponse(
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
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/users/by-name/{name}", response_model=SingleUserResponse)
def get_user_by_name(name: str):
    """Get user by username or display name."""
    try:
        db = get_db()
        user = db.get_user_by_name(name)
        if not user:
            return slack_error_response(404, "user_not_found")
        
        return SingleUserResponse(
            ok=True,
            user=UserResponse(
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
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/users/{user_id}/channels", response_model=ChannelListResponse)
def get_user_channels(user_id: str):
    """Get channels where user is a member."""
    try:
        db = get_db()
        
        # Verify user exists
        user = db.get_user(user_id)
        if not user:
            return slack_error_response(404, "user_not_found")
        
        channels = db.get_user_channels(user_id)
        return ChannelListResponse(
            ok=True,
            channels=[
                ChannelResponse(
                    id=ch.id,
                    name=ch.name,
                    is_channel=ch.is_channel,
                    is_private=ch.is_private,
                    is_im=ch.is_im,
                    is_mpim=ch.is_mpim,
                    is_archived=ch.is_archived,
                    is_general=ch.is_general,
                    num_members=ch.num_members,
                    topic=ch.topic,
                    purpose=ch.purpose,
                    created=ch.created,
                    creator=ch.creator,
                    members=ch.members,
                )
                for ch in channels
            ],
            response_metadata=ResponseMetadata(next_cursor=None)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ============================================================================
# Search Endpoints (placed before channel routes to avoid route conflicts)
# ============================================================================

def _search_messages_impl(
    query: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = 20,
    date_on: Optional[str] = None,
    date_during: Optional[str] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None,
    in_channel: Optional[str] = None,
    in_im_or_mpim: Optional[str] = None,
    threads_only: bool = False,
    from_user: Optional[str] = None,
    with_user: Optional[str] = None,
) -> SearchMessagesResponse:
    """Shared implementation for search endpoints."""
    db = get_db()
    
    # Require at least one filter
    has_filter = any([
        query, date_on, date_during, date_after, date_before,
        in_channel, in_im_or_mpim, threads_only, from_user, with_user
    ])
    
    if not has_filter:
        return slack_error_response(400, "missing_query_or_filter")
    
    messages, next_cursor = db.search_messages(
        search_query=query,
        cursor=cursor,
        limit=limit,
        filter_date_on=date_on,
        filter_date_during=date_during,
        filter_date_after=date_after,
        filter_date_before=date_before,
        filter_in_channel=in_channel,
        filter_in_im_or_mpim=in_im_or_mpim,
        filter_threads_only=threads_only,
        filter_users_from=from_user,
        filter_users_with=with_user,
    )
    
    return SearchMessagesResponse(
        ok=True,
        messages=[
            MessageResponse(
                ts=m.ts,
                type=m.type,
                user=m.user,
                text=m.text,
                channel=m.channel,
                thread_ts=m.thread_ts,
                reply_count=m.reply_count,
                reply_users_count=m.reply_users_count,
                latest_reply=m.latest_reply,
                is_activity_message=m.is_activity_message,
                subtype=m.subtype,
                reactions=m.reactions,
                files=m.files,
                attachments=m.attachments,
            )
            for m in messages
        ],
        response_metadata=ResponseMetadata(next_cursor=next_cursor)
    )


@router.get("/search/messages", response_model=SearchMessagesResponse)
def search_messages_alt(
    q: Optional[str] = Query(None, description="Search query or Slack URL"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results (1-100)"),
    date_on: Optional[str] = Query(None, description="Exact date (YYYY-MM-DD)"),
    date_during: Optional[str] = Query(None, description="Calendar period"),
    date_after: Optional[str] = Query(None, description="After date"),
    date_before: Optional[str] = Query(None, description="Before date"),
    in_channel: Optional[str] = Query(None, description="Channel ID or #name"),
    in_im_or_mpim: Optional[str] = Query(None, description="IM/MPIM ID or @username"),
    threads_only: bool = Query(False, description="Only return thread messages"),
    from_user: Optional[str] = Query(None, description="Filter by sender"),
    with_user: Optional[str] = Query(None, description="Filter by participant"),
):
    """Search messages with filters (alternative route: /search/messages)."""
    try:
        return _search_messages_impl(
            query=q,
            cursor=cursor,
            limit=limit,
            date_on=date_on,
            date_during=date_during,
            date_after=date_after,
            date_before=date_before,
            in_channel=in_channel,
            in_im_or_mpim=in_im_or_mpim,
            threads_only=threads_only,
            from_user=from_user,
            with_user=with_user,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/messages/search", response_model=SearchMessagesResponse)
def search_messages(
    query: Optional[str] = Query(None, alias="q", description="Search query or Slack URL"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results (1-100)"),
    date_on: Optional[str] = Query(None, description="Exact date (YYYY-MM-DD)"),
    date_during: Optional[str] = Query(None, description="Calendar period"),
    date_after: Optional[str] = Query(None, description="After date"),
    date_before: Optional[str] = Query(None, description="Before date"),
    in_channel: Optional[str] = Query(None, description="Channel ID or #name"),
    in_im_or_mpim: Optional[str] = Query(None, description="IM/MPIM ID or @username"),
    threads_only: bool = Query(False, description="Only return thread messages"),
    from_user: Optional[str] = Query(None, description="Filter by sender"),
    with_user: Optional[str] = Query(None, description="Filter by participant"),
):
    """Search messages with filters."""
    try:
        return _search_messages_impl(
            query=query,
            cursor=cursor,
            limit=limit,
            date_on=date_on,
            date_during=date_during,
            date_after=date_after,
            date_before=date_before,
            in_channel=in_channel,
            in_im_or_mpim=in_im_or_mpim,
            threads_only=threads_only,
            from_user=from_user,
            with_user=with_user,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ============================================================================
# Channel Endpoints
# ============================================================================

@router.get("/channels", response_model=ChannelListResponse)
def list_channels(
    member_only: bool = Query(False, description="Only show channels where current user is a member"),
    types: str = Query("public_channel,private_channel,im,mpim", description="Comma-separated channel types"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(100, ge=1, le=999, description="Maximum number of results"),
    sort: Optional[str] = Query(None, description="Sort by 'popularity' for member count sort"),
):
    """List channels with filtering and pagination."""
    try:
        db = get_db()
        
        # Parse channel types
        channel_types = [t.strip() for t in types.split(",")]
        valid_types = {"public_channel", "private_channel", "im", "mpim"}
        for ct in channel_types:
            if ct not in valid_types:
                raise HTTPException(status_code=400, detail=f"invalid_types: {ct}")
        
        sort_by_popularity = sort == "popularity"
        
        if member_only:
            current_user = db.get_current_user()
            if not current_user:
                return ChannelListResponse(
                    ok=True, channels=[], response_metadata=ResponseMetadata(next_cursor=None)
                )
            channels, next_cursor = db.list_channels_for_user(
                user_id=current_user.id,
                channel_types=channel_types,
                cursor=cursor,
                limit=limit,
                sort_by_popularity=sort_by_popularity,
            )
        else:
            channels, next_cursor = db.list_channels(
                channel_types=channel_types,
                cursor=cursor,
                limit=limit,
                sort_by_popularity=sort_by_popularity,
            )
        
        return ChannelListResponse(
            ok=True,
            channels=[
                ChannelResponse(
                    id=ch.id,
                    name=ch.name,
                    is_channel=ch.is_channel,
                    is_private=ch.is_private,
                    is_im=ch.is_im,
                    is_mpim=ch.is_mpim,
                    is_archived=ch.is_archived,
                    is_general=ch.is_general,
                    num_members=ch.num_members,
                    topic=ch.topic,
                    purpose=ch.purpose,
                    created=ch.created,
                    creator=ch.creator,
                    members=ch.members,
                )
                for ch in channels
            ],
            response_metadata=ResponseMetadata(next_cursor=next_cursor)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/channels/{channel_id}", response_model=SingleChannelResponse)
def get_channel(channel_id: str):
    """Get channel details."""
    try:
        db = get_db()
        channel = db.get_channel(channel_id)
        if not channel:
            return slack_error_response(404, "channel_not_found")
        
        return SingleChannelResponse(
            ok=True,
            channel=ChannelResponse(
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
                created=channel.created,
                creator=channel.creator,
                members=channel.members,
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/channels/{channel_id}/members", response_model=ChannelMembersResponse)
def get_channel_members(channel_id: str):
    """Get channel members with their full details."""
    try:
        db = get_db()
        
        # Verify channel exists
        channel = db.get_channel(channel_id)
        if not channel:
            return slack_error_response(404, "channel_not_found")
        
        members = db.get_channel_members_with_details(channel_id)
        return ChannelMembersResponse(
            ok=True,
            members=[
                UserResponse(
                    id=u.id,
                    name=u.name,
                    real_name=u.real_name,
                    display_name=u.display_name,
                    email=u.email,
                    avatar_url=u.avatar_url,
                    is_bot=u.is_bot,
                    is_admin=u.is_admin,
                    deleted=u.deleted,
                    timezone=u.timezone,
                )
                for u in members
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ============================================================================
# Message Endpoints
# ============================================================================

@router.get("/channels/{channel_id}/messages", response_model=MessageListResponse)
def get_channel_messages(
    channel_id: str,
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    include_activity: bool = Query(False, description="Include activity messages"),
):
    """
    Get channel conversation history.
    
    CRITICAL: This returns ALL messages (parents + replies).
    The frontend MUST filter to show only parent messages in the main channel view.
    """
    try:
        db = get_db()
        
        # Verify channel exists
        channel = db.get_channel(channel_id)
        if not channel:
            return slack_error_response(404, "channel_not_found")
        
        # CRITICAL FIX: Use correct parameter name 'include_activity_messages'
        messages, next_cursor = db.get_conversation_history(
            channel_id=channel_id,
            cursor=cursor,
            limit=str(limit),  # MCP method expects string
            include_activity_messages=include_activity,
        )
        
        return MessageListResponse(
            ok=True,
            messages=[
                MessageResponse(
                    ts=m.ts,
                    type=m.type,
                    user=m.user,
                    text=m.text,
                    channel=m.channel,
                    thread_ts=m.thread_ts,
                    reply_count=m.reply_count,
                    reply_users_count=m.reply_users_count,
                    latest_reply=m.latest_reply,
                    is_activity_message=m.is_activity_message,
                    subtype=m.subtype,
                    reactions=m.reactions,
                    files=m.files,
                    attachments=m.attachments,
                )
                for m in messages
            ],
            has_more=next_cursor is not None,
            response_metadata=ResponseMetadata(next_cursor=next_cursor)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/channels/{channel_id}/threads/{thread_ts}", response_model=MessageListResponse)
def get_thread_messages(
    channel_id: str,
    thread_ts: str,
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    include_activity: bool = Query(False, description="Include activity messages"),
):
    """
    Get thread replies.
    
    Returns the parent message and all replies in the thread.
    The frontend should display all returned messages (no filtering needed).
    """
    try:
        db = get_db()
        
        # Verify channel exists
        channel = db.get_channel(channel_id)
        if not channel:
            return slack_error_response(404, "channel_not_found")
        
        # CRITICAL FIX: Use correct parameter name 'include_activity_messages'
        messages, next_cursor = db.get_thread_replies(
            channel_id=channel_id,
            thread_ts=thread_ts,
            cursor=cursor,
            limit=str(limit),  # MCP method expects string
            include_activity_messages=include_activity,
        )
        
        return MessageListResponse(
            ok=True,
            messages=[
                MessageResponse(
                    ts=m.ts,
                    type=m.type,
                    user=m.user,
                    text=m.text,
                    channel=m.channel,
                    thread_ts=m.thread_ts,
                    reply_count=m.reply_count,
                    reply_users_count=m.reply_users_count,
                    latest_reply=m.latest_reply,
                    is_activity_message=m.is_activity_message,
                    subtype=m.subtype,
                    reactions=m.reactions,
                    files=m.files,
                    attachments=m.attachments,
                )
                for m in messages
            ],
            has_more=next_cursor is not None,
            response_metadata=ResponseMetadata(next_cursor=next_cursor)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/messages", response_model=CreateMessageResponse)
def create_message(request: CreateMessageRequest):
    """Create a new message or thread reply."""
    try:
        db = get_db()
        
        # Verify channel exists
        channel = db.get_channel(request.channel_id)
        if not channel:
            return slack_error_response(404, "channel_not_found")
        
        # Create the message using the extended database method
        created_message = db.create_new_message(
            channel_id=request.channel_id,
            text=request.text,
            user_id=CURRENT_USER_ID,
            thread_ts=request.thread_ts,
        )
        
        return CreateMessageResponse(
            ok=True,
            message=MessageResponse(
                ts=created_message.ts,
                type=created_message.type,
                user=created_message.user,
                text=created_message.text,
                channel=created_message.channel,
                thread_ts=created_message.thread_ts,
                reply_count=created_message.reply_count,
                reply_users_count=created_message.reply_users_count,
                latest_reply=created_message.latest_reply,
                is_activity_message=created_message.is_activity_message,
                subtype=created_message.subtype,
                reactions=created_message.reactions,
                files=created_message.files,
                attachments=created_message.attachments,
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.patch("/channels/{channel_id}/messages/{message_ts}", response_model=CreateMessageResponse)
def edit_message(channel_id: str, message_ts: str, request: UpdateMessageRequest):
    """Edit the current user's own message."""
    try:
        db = get_db()
        updated_message = db.update_message_text(channel_id, message_ts, request.text)
        if not updated_message:
            return slack_error_response(404, "message_not_found_or_forbidden")

        return CreateMessageResponse(
            ok=True,
            message=MessageResponse(
                ts=updated_message.ts,
                type=updated_message.type,
                user=updated_message.user,
                text=updated_message.text,
                channel=updated_message.channel,
                thread_ts=updated_message.thread_ts,
                reply_count=updated_message.reply_count,
                reply_users_count=updated_message.reply_users_count,
                latest_reply=updated_message.latest_reply,
                is_activity_message=updated_message.is_activity_message,
                subtype=updated_message.subtype,
                reactions=updated_message.reactions,
                files=updated_message.files,
                attachments=updated_message.attachments,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.delete("/channels/{channel_id}/messages/{message_ts}")
def delete_message(channel_id: str, message_ts: str):
    """Delete the current user's own message."""
    try:
        db = get_db()
        deleted = db.delete_message_by_ts(channel_id, message_ts)
        if not deleted:
            return slack_error_response(404, "message_not_found_or_forbidden")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/channels/{channel_id}/messages/{message_ts}/reactions", response_model=CreateMessageResponse)
def toggle_reaction(channel_id: str, message_ts: str, request: ToggleReactionRequest):
    """Toggle reaction participation for the current user."""
    try:
        db = get_db()
        message = db.toggle_message_reaction(channel_id, message_ts, request.name)
        if not message:
            return slack_error_response(404, "message_not_found")
        return CreateMessageResponse(
            ok=True,
            message=MessageResponse(
                ts=message.ts,
                type=message.type,
                user=message.user,
                text=message.text,
                channel=message.channel,
                thread_ts=message.thread_ts,
                reply_count=message.reply_count,
                reply_users_count=message.reply_users_count,
                latest_reply=message.latest_reply,
                is_activity_message=message.is_activity_message,
                subtype=message.subtype,
                reactions=message.reactions,
                files=message.files,
                attachments=message.attachments,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")