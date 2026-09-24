"""
Database layer for Email MCP Server using SQLAlchemy.

This module provides database operations for the Email MCP server,
extending the BaseDatabase class for consistent database management.

Features:
1. **Backend Agnostic**: Supports SQLite (memory/file) and PostgreSQL via BaseDatabase
2. **Type Safety**: Excellent typing support with modern Python
3. **Folder-based Organization**: Emails organized in INBOX, SENT, DRAFT, TRASH folders
4. **Session Management**: Built-in connection pooling and transaction handling
"""

import json
import sys
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    Boolean,
    Float,
    String,
    Text,
    Enum as SQLEnum,
    desc,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)


# =============================================================================
# IMPORT BASE DATABASE
# =============================================================================


def _import_base_database():
    """Import BaseDatabase handling hyphenated directory names."""
    current_dir = Path(__file__).parent
    base_db_path = current_dir.parent / "base_mcp" / "base_database.py"

    # Check if already imported
    if "email_base_database_module" in sys.modules:
        return (
            sys.modules["email_base_database_module"].BaseDatabase,
            sys.modules["email_base_database_module"].DatabaseBackend,
        )

    # Import using importlib
    spec = importlib.util.spec_from_file_location("email_base_database_module", base_db_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["email_base_database_module"] = module
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
        Email,
        EmailFolderName,
        EmailListResponse,
        generate_email_id,
    )
except ImportError:
    # Direct import when not running as package
    from pathlib import Path as _Path

    _models_path = _Path(__file__).parent / "models.py"
    if "email_models" not in sys.modules:
        _spec = importlib.util.spec_from_file_location("email_models", _models_path)
        _models = importlib.util.module_from_spec(_spec)
        sys.modules["email_models"] = _models
        _spec.loader.exec_module(_models)
    else:
        _models = sys.modules["email_models"]

    Email = _models.Email
    EmailFolderName = _models.EmailFolderName
    EmailListResponse = _models.EmailListResponse
    generate_email_id = _models.generate_email_id


# =============================================================================
# SQLALCHEMY BASE FOR EMAIL TABLES
# =============================================================================


class EmailBase(DeclarativeBase):
    """Base class for Email-specific SQLAlchemy models."""

    pass


# =============================================================================
# DATABASE TABLES
# =============================================================================


class EmailTable(EmailBase):
    """SQLAlchemy model for emails."""

    __tablename__ = "emails"

    email_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    folder: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    sender: Mapped[str] = mapped_column(String(255), nullable=False)
    recipients_json: Mapped[str] = mapped_column(Text, default="[]")
    subject: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text, default="")
    cc_json: Mapped[str] = mapped_column(Text, default="[]")
    parent_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    attachments_json: Mapped[str] = mapped_column(Text, default="{}")
    timestamp: Mapped[float] = mapped_column(Float, default=lambda: datetime.now().timestamp())
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)

    def to_model(self) -> Email:
        """Convert to domain model."""
        return Email(
            email_id=self.email_id,
            folder=self.folder,
            sender=self.sender,
            recipients=json.loads(self.recipients_json),
            subject=self.subject,
            content=self.content,
            cc=json.loads(self.cc_json),
            parent_id=self.parent_id,
            attachments=json.loads(self.attachments_json),
            timestamp=self.timestamp,
            is_read=self.is_read,
        )

    @classmethod
    def from_model(cls, email: Email, folder: str) -> "EmailTable":
        """Create from domain model."""
        return cls(
            email_id=email.email_id,
            folder=folder,
            sender=email.sender,
            recipients_json=json.dumps(email.recipients),
            subject=email.subject,
            content=email.content,
            cc_json=json.dumps(email.cc),
            parent_id=email.parent_id,
            attachments_json=json.dumps(email.attachments),
            timestamp=email.timestamp,
            is_read=email.is_read,
        )


# =============================================================================
# DATABASE MANAGER
# =============================================================================


class EmailDatabase(BaseDatabase):
    """
    Database manager for Email MCP Server.

    Extends BaseDatabase to provide CRUD operations for all Email entities.
    Requires PostgreSQL database via connection_string.

    Example usage:
        db = EmailDatabase(connection_string="postgresql://user:pass@localhost:5432/email")
    """

    def __init__(self, connection_string: str, **engine_kwargs: Any):
        """
        Initialize the Email database.

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
        """Create all Email-specific tables."""
        EmailBase.metadata.create_all(self.engine)

    # =========================================================================
    # EMAIL OPERATIONS
    # =========================================================================

    def create_email(self, email: Email, folder: str) -> Email:
        """Create a new email in a specific folder."""
        with self.get_session() as session:
            email_table = EmailTable.from_model(email, folder)
            session.add(email_table)
            session.flush()
            return email_table.to_model()

    def get_email_by_id(self, email_id: str, folder: Optional[str] = None) -> Optional[Email]:
        """
        Get an email by ID.

        Args:
            email_id: The unique email ID.
            folder: Optional folder to search in. If None, searches all folders.

        Returns:
            Email if found, None otherwise.
        """
        with self.get_session() as session:
            query = session.query(EmailTable).filter(EmailTable.email_id == email_id)
            if folder:
                query = query.filter(EmailTable.folder == folder)
            email_table = query.first()
            return email_table.to_model() if email_table else None

    def get_email_by_index(self, idx: int, folder: str) -> Optional[Email]:
        """
        Get an email by index position in a folder.

        Args:
            idx: The 0-based index.
            folder: The folder to search in.

        Returns:
            Email if found, None otherwise.
        """
        with self.get_session() as session:
            emails = (
                session.query(EmailTable)
                .filter(EmailTable.folder == folder)
                .order_by(desc(EmailTable.timestamp))
                .all()
            )
            if 0 <= idx < len(emails):
                return emails[idx].to_model()
            return None

    def list_emails(
        self,
        folder: str,
        offset: int = 0,
        limit: int = 10,
    ) -> EmailListResponse:
        """
        List emails in a folder with pagination.

        Args:
            folder: The folder to list emails from.
            offset: Number of emails to skip.
            limit: Maximum number of emails to return.

        Returns:
            EmailListResponse with emails and pagination info.
        """
        with self.get_session() as session:
            # Get total count
            total_count = (
                session.query(EmailTable)
                .filter(EmailTable.folder == folder)
                .count()
            )

            # Get paginated emails
            emails = (
                session.query(EmailTable)
                .filter(EmailTable.folder == folder)
                .order_by(desc(EmailTable.timestamp))
                .offset(offset)
                .limit(limit)
                .all()
            )

            end_idx = min(offset + limit, total_count)

            return EmailListResponse(
                emails=[e.to_model() for e in emails],
                emails_range=(offset, end_idx),
                total_returned_emails=len(emails),
                total_emails=total_count,
            )

    def search_emails(
        self,
        query: str,
        folder: str,
    ) -> List[Email]:
        """
        Search for emails in a folder.

        Args:
            query: Case-insensitive search query.
            folder: The folder to search in.

        Returns:
            List of matching emails.
        """
        query_lower = f"%{query.lower()}%"
        with self.get_session() as session:
            emails = (
                session.query(EmailTable)
                .filter(EmailTable.folder == folder)
                .filter(
                    (EmailTable.sender.ilike(query_lower))
                    | (EmailTable.recipients_json.ilike(query_lower))
                    | (EmailTable.subject.ilike(query_lower))
                    | (EmailTable.content.ilike(query_lower))
                )
                .order_by(desc(EmailTable.timestamp))
                .all()
            )
            return [e.to_model() for e in emails]

    def move_email(self, email_id: str, from_folder: str, to_folder: str) -> Optional[Email]:
        """
        Move an email from one folder to another.

        Args:
            email_id: The email ID to move.
            from_folder: Source folder.
            to_folder: Destination folder.

        Returns:
            The moved email, or None if not found.
        """
        with self.get_session() as session:
            email_table = (
                session.query(EmailTable)
                .filter(
                    EmailTable.email_id == email_id,
                    EmailTable.folder == from_folder,
                )
                .first()
            )
            if email_table:
                email_table.folder = to_folder
                session.flush()
                return email_table.to_model()
            return None

    def delete_email(self, email_id: str, folder: str) -> bool:
        """
        Permanently delete an email.

        Args:
            email_id: The email ID to delete.
            folder: The folder containing the email.

        Returns:
            True if deleted, False if not found.
        """
        with self.get_session() as session:
            email_table = (
                session.query(EmailTable)
                .filter(
                    EmailTable.email_id == email_id,
                    EmailTable.folder == folder,
                )
                .first()
            )
            if email_table:
                session.delete(email_table)
                return True
            return False

    def mark_as_read(self, email_id: str, folder: str) -> Optional[Email]:
        """
        Mark an email as read.

        Args:
            email_id: The email ID.
            folder: The folder containing the email.

        Returns:
            The updated email, or None if not found.
        """
        with self.get_session() as session:
            email_table = (
                session.query(EmailTable)
                .filter(
                    EmailTable.email_id == email_id,
                    EmailTable.folder == folder,
                )
                .first()
            )
            if email_table:
                email_table.is_read = True
                session.flush()
                return email_table.to_model()
            return None

    def get_all_emails_by_folder(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, List[Email]]:
        """
        Get all emails organized by folder.

        Args:
            offset: Number of emails to skip per folder.
            limit: Maximum number of emails per folder.

        Returns:
            Dict mapping folder names to lists of emails.
        """
        result = {}
        with self.get_session() as session:
            for folder_name in EmailFolderName:
                emails = (
                    session.query(EmailTable)
                    .filter(EmailTable.folder == folder_name.value)
                    .order_by(desc(EmailTable.timestamp))
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                result[folder_name.value] = [e.to_model() for e in emails]
        return result

    def get_folder_email_count(self, folder: str) -> int:
        """Get the count of emails in a folder."""
        with self.get_session() as session:
            return (
                session.query(EmailTable)
                .filter(EmailTable.folder == folder)
                .count()
            )

    # =========================================================================
    # RESET & DATA LOADING (BaseDatabase implementation)
    # =========================================================================

    def reset_all(self) -> None:
        """Drop all tables and recreate them."""
        EmailBase.metadata.drop_all(self.engine)
        EmailBase.metadata.create_all(self.engine)

    def load_from_json(self, json_path: str) -> Dict[str, int]:
        """
        Load mock data from a JSON file.

        Expected JSON structure:
        {
            "emails": [
                {
                    "email_id": "...",
                    "folder": "INBOX",  // Required: INBOX, SENT, DRAFT, or TRASH
                    "sender": "...",
                    "recipients": ["..."],
                    "subject": "...",
                    "content": "...",
                    "cc": [],
                    "parent_id": null,
                    "attachments": {},
                    "timestamp": 1702900000.0,
                    "is_read": false
                }
            ],
            "user_email": "user@example.com"
        }

        Returns:
            Dict with counts of loaded entities.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        counts = {"emails": 0}

        with self.get_session() as session:
            for email_data in data.get("emails", []):
                folder_name = email_data.get("folder", "INBOX")
                # Validate folder name
                if folder_name not in [f.value for f in EmailFolderName]:
                    folder_name = "INBOX"
                
                email = Email(
                    email_id=email_data.get("email_id", generate_email_id()),
                    folder=folder_name,
                    sender=email_data.get("sender", ""),
                    recipients=email_data.get("recipients", []),
                    subject=email_data.get("subject", ""),
                    content=email_data.get("content", ""),
                    cc=email_data.get("cc", []),
                    parent_id=email_data.get("parent_id"),
                    attachments=email_data.get("attachments", {}),
                    timestamp=email_data.get("timestamp", datetime.now().timestamp()),
                    is_read=email_data.get("is_read", False),
                )
                session.add(EmailTable.from_model(email, folder_name))
                counts["emails"] += 1

        return counts

    def export_state(self) -> Dict[str, Any]:
        with self.get_session() as session:
            emails = [row.to_model().to_dict() for row in session.query(EmailTable).all()]
        return {"emails": emails}

