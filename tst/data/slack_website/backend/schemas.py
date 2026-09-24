"""
Pydantic schemas for API request/response validation.
These schemas map to MCP models for type safety and validation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class ChannelTypeEnum(str, Enum):
    PUBLIC_CHANNEL = "public_channel"
    PRIVATE_CHANNEL = "private_channel"
    IM = "im"
    MPIM = "mpim"


class SortEnum(str, Enum):
    POPULARITY = "popularity"
    NAME = "name"


# ============================================================================
# User Schemas
# ============================================================================

class UserResponse(BaseModel):
    """User response schema matching MCP User model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    name: str
    real_name: str
    display_name: str
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    is_admin: bool = False
    deleted: bool = False
    timezone: str = "UTC"


class CurrentUserResponse(BaseModel):
    """Response for /api/me endpoint."""
    ok: bool = True
    user: UserResponse


class UpdateCurrentUserRequest(BaseModel):
    """Request body for PATCH /api/me."""
    full_name: Optional[str] = None
    display_name: Optional[str] = None
    username: Optional[str] = None
    status: Optional[str] = None
    avatar_url: Optional[str] = None


class UserListResponse(BaseModel):
    """Response for /api/users endpoint."""
    ok: bool = True
    users: List[UserResponse]


class SingleUserResponse(BaseModel):
    """Response for /api/users/{id} endpoint."""
    ok: bool = True
    user: UserResponse


class UserDetailResponse(BaseModel):
    """Response for user detail."""
    ok: bool = True
    user: UserResponse


# ============================================================================
# Channel Schemas
# ============================================================================

class ChannelResponse(BaseModel):
    """Channel response schema matching MCP Channel model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    name: str
    is_channel: bool = True
    is_private: bool = False
    is_im: bool = False
    is_mpim: bool = False
    is_archived: bool = False
    is_general: bool = False
    num_members: int = 0
    topic: str = ""
    purpose: str = ""
    created: str = ""
    creator: Optional[str] = None
    members: List[str] = []


class ResponseMetadata(BaseModel):
    """Pagination metadata."""
    next_cursor: Optional[str] = None


class ChannelListResponse(BaseModel):
    """Response for /api/channels endpoint."""
    ok: bool = True
    channels: List[ChannelResponse]
    response_metadata: ResponseMetadata = ResponseMetadata()


class SingleChannelResponse(BaseModel):
    """Response for /api/channels/{id} endpoint."""
    ok: bool = True
    channel: ChannelResponse


class ChannelDetailResponse(BaseModel):
    """Response for channel detail."""
    ok: bool = True
    channel: ChannelResponse


class ChannelMembersResponse(BaseModel):
    """Response for /api/channels/{id}/members endpoint."""
    ok: bool = True
    members: List[UserResponse]


# ============================================================================
# Message Schemas
# ============================================================================

class MessageResponse(BaseModel):
    """Message response schema matching MCP Message model."""
    model_config = ConfigDict(from_attributes=True)
    
    ts: str
    type: str = "message"
    user: Optional[str] = None
    text: str = ""
    channel: Optional[str] = None
    thread_ts: Optional[str] = None
    reply_count: int = 0
    reply_users_count: int = 0
    latest_reply: Optional[str] = None
    is_activity_message: bool = False
    subtype: Optional[str] = None
    reactions: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    attachments: List[Dict[str, Any]] = []


class MessageListResponse(BaseModel):
    """Response for message list endpoints."""
    ok: bool = True
    messages: List[MessageResponse]
    has_more: bool = False
    response_metadata: ResponseMetadata = ResponseMetadata()


class MessageDetailResponse(BaseModel):
    """Response for single message."""
    ok: bool = True
    message: MessageResponse


class CreateMessageRequest(BaseModel):
    """Request body for POST /api/messages."""
    channel_id: str = Field(..., description="Channel ID to post to")
    text: str = Field(..., min_length=1, description="Message content")
    thread_ts: Optional[str] = Field(None, description="Parent thread timestamp for replies")


class CreateMessageResponse(BaseModel):
    """Response for POST /api/messages."""
    ok: bool = True
    message: MessageResponse


class UpdateMessageRequest(BaseModel):
    """Request body for PATCH message endpoint."""
    text: str = Field(..., min_length=1, description="Updated message content")


class ToggleReactionRequest(BaseModel):
    """Request body for toggling a reaction."""
    name: str = Field(..., min_length=1, description="Reaction emoji or name")


class OpenDMRequest(BaseModel):
    """Request body for opening or creating a DM/MPIM."""
    user_ids: List[str] = Field(..., min_length=1, description="Recipient user IDs")


class OpenDMResponse(BaseModel):
    """Response for opening/creating a DM."""
    ok: bool = True
    channel: ChannelResponse


# ============================================================================
# Search Schemas
# ============================================================================

class SearchMessagesResponse(BaseModel):
    """Response for /api/messages/search or /api/search/messages endpoint."""
    ok: bool = True
    messages: List[MessageResponse]
    response_metadata: ResponseMetadata = ResponseMetadata()


class SearchResponse(BaseModel):
    """Response for general search."""
    ok: bool = True
    messages: List[MessageResponse]
    total_count: Optional[int] = None
    response_metadata: ResponseMetadata = ResponseMetadata()


# ============================================================================
# Data Management Schemas
# ============================================================================

class ResetRequest(BaseModel):
    """Request body for POST /api/reset."""
    load_mock_data: bool = Field(
        default=False,
        description="Whether to load mock data from default location"
    )
    mock_data_path: Optional[str] = Field(
        default=None,
        description="Path to mock data JSON file (overrides default location)"
    )


class ResetResponse(BaseModel):
    """Response for POST /api/reset."""
    ok: bool = True
    loaded: Optional[Dict[str, int]] = None
    message: Optional[str] = None


# ============================================================================
# Health Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Response for GET /api/health."""
    status: str = "ok"
    database: str = "connected"


class StatsResponse(BaseModel):
    """Database statistics response."""
    users: int
    channels: int
    messages: int


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    ok: bool = False
    error: str
    warning: Optional[str] = None