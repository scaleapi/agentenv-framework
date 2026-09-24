"""
Data models for Email MCP Server.

This module defines all the data models representing Email entities,
following the structure defined in the OpenAPI specification.
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


class EmailFolderName(str, Enum):
    """Valid folder names for organizing emails."""

    INBOX = "INBOX"
    SENT = "SENT"
    DRAFT = "DRAFT"
    TRASH = "TRASH"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_email_id() -> str:
    """Generate a unique email ID (hex UUID)."""
    return uuid.uuid4().hex


def validate_email(email: str) -> bool:
    """Validate email address format (simple check)."""
    return "@" in email and "." in email


# =============================================================================
# CORE MODELS
# =============================================================================


@dataclass
class Email:
    """Email object.
    
    Attributes:
        email_id: Unique identifier for the email.
        folder: The folder containing this email (INBOX, SENT, DRAFT, TRASH).
                Required for data loading/generation.
        sender: Email address of the sender.
        recipients: List of recipient email addresses.
        subject: Email subject line.
        content: Email body content.
        cc: List of CC email addresses.
        parent_id: ID of parent email for replies/forwards.
        attachments: Dict mapping filename to base64 content.
        timestamp: Unix timestamp of when email was sent/received.
        is_read: Whether the email has been read.
    """

    email_id: str = field(default_factory=generate_email_id)
    folder: str = "INBOX"  # One of: INBOX, SENT, DRAFT, TRASH
    sender: str = ""
    recipients: List[str] = field(default_factory=list)
    subject: str = ""
    content: str = ""
    cc: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None  # For replies/forwards
    attachments: Dict[str, str] = field(default_factory=dict)  # filename -> base64 content
    timestamp: float = field(default_factory=time.time)
    is_read: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "email_id": self.email_id,
            "folder": self.folder,
            "sender": self.sender,
            "recipients": self.recipients,
            "subject": self.subject,
            "content": self.content,
            "cc": self.cc,
            "parent_id": self.parent_id,
            "attachments": self.attachments,
            "timestamp": self.timestamp,
            "is_read": self.is_read,
        }


# =============================================================================
# RESPONSE MODELS
# =============================================================================


@dataclass
class EmailListResponse:
    """Response for listing emails with pagination info."""

    emails: List[Email] = field(default_factory=list)
    emails_range: tuple = field(default_factory=lambda: (0, 0))
    total_returned_emails: int = 0
    total_emails: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "emails": [e.to_dict() for e in self.emails],
            "emails_range": list(self.emails_range),
            "total_returned_emails": self.total_returned_emails,
            "total_emails": self.total_emails,
        }


@dataclass
class EmailActionResponse:
    """Response for email actions (send, move, delete)."""

    email_id: str
    moved_to: Optional[str] = None
    permanently_deleted: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {"email_id": self.email_id}
        if self.moved_to is not None:
            result["moved_to"] = self.moved_to
        if self.permanently_deleted is not None:
            result["permanently_deleted"] = self.permanently_deleted
        return result


@dataclass
class ShowDataResponse:
    """Response for show_data endpoint."""

    folders: Dict[str, List[Email]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "folders": {
                folder_name: [e.to_dict() for e in emails]
                for folder_name, emails in self.folders.items()
            }
        }


# =============================================================================
# ERROR MODELS
# =============================================================================


@dataclass
class EmailError:
    """Error response from Email API."""

    error: str = "unknown_error"
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {"error": self.error}
        if self.message:
            result["message"] = self.message
        return result


# =============================================================================
# MOCK DATA SCHEMA (for LLM data generation)
# =============================================================================


@dataclass
class MockDataSchema:
    """
    Schema for mock data JSON files used to populate the Email database.
    
    This class documents the expected JSON format for data generation.
    LLMs should generate data in this format for loading via load_from_json().
    
    Expected JSON structure:
    ```json
    {
        "emails": [
            {
                "email_id": "unique_hex_string",
                "folder": "INBOX",  // One of: INBOX, SENT, DRAFT, TRASH
                "sender": "sender@example.com",
                "recipients": ["recipient@example.com"],
                "subject": "Email Subject",
                "content": "Email body content",
                "cc": [],
                "parent_id": null,  // or email_id of parent for replies
                "attachments": {},
                "timestamp": 1702900000.0,  // Unix timestamp
                "is_read": false
            }
        ],
        "user_email": "currentuser@example.com"  // Optional: the logged-in user's email
    }
    ```
    
    Notes:
    - folder must be one of the values in EmailFolderName enum
    - email_id should be a unique hex string (32 chars recommended)
    - timestamp should be a Unix timestamp (float)
    - parent_id references another email_id for reply chains
    """
    
    emails: List[Email] = field(default_factory=list)
    user_email: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        return {
            "emails": [e.to_dict() for e in self.emails],
            "user_email": self.user_email,
        }

