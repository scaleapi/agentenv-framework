"""
Email MCP Service - MCP server implementation for Email API.

This module provides an MCP server that simulates the Email API endpoints,
implementing tools as defined in the OpenAPI specification.
"""

import json
import os
import sys
import importlib.util
import time
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

from pydantic import Field


def _import_base_service():
    """Import BaseService handling hyphenated directory names."""
    # Get the path to the BaseService module
    current_dir = Path(__file__).parent
    base_service_path = current_dir.parent / "base_mcp" / "base_service.py"

    # Check if already imported
    if "email_base_service" in sys.modules:
        return sys.modules["email_base_service"].BaseService

    # Import using importlib
    spec = importlib.util.spec_from_file_location(
        "email_base_service", base_service_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["email_base_service"] = module
    spec.loader.exec_module(module)
    return module.BaseService


BaseService = _import_base_service()


def _import_local_modules():
    """Import local modules handling hyphenated directory names."""
    current_dir = Path(__file__).parent

    # Import models
    models_path = current_dir / "models.py"
    if "email_models" not in sys.modules:
        spec = importlib.util.spec_from_file_location("email_models", models_path)
        models = importlib.util.module_from_spec(spec)
        sys.modules["email_models"] = models
        spec.loader.exec_module(models)
    else:
        models = sys.modules["email_models"]

    # Import database
    db_path = current_dir / "database.py"
    if "email_database" not in sys.modules:
        spec = importlib.util.spec_from_file_location("email_database", db_path)
        db = importlib.util.module_from_spec(spec)
        sys.modules["email_database"] = db
        spec.loader.exec_module(db)
    else:
        db = sys.modules["email_database"]

    return models, db


# Try relative imports first, fall back to dynamic import
try:
    from .database import EmailDatabase
    from .models import (
        Email,
        EmailFolderName,
        EmailListResponse,
        EmailActionResponse,
        ShowDataResponse,
        EmailError,
        generate_email_id,
    )
except ImportError:
    _models, _db = _import_local_modules()

    EmailDatabase = _db.EmailDatabase
    Email = _models.Email
    EmailFolderName = _models.EmailFolderName
    EmailListResponse = _models.EmailListResponse
    EmailActionResponse = _models.EmailActionResponse
    ShowDataResponse = _models.ShowDataResponse
    EmailError = _models.EmailError
    generate_email_id = _models.generate_email_id


# Valid folder names for validation
VALID_FOLDERS = {f.value for f in EmailFolderName}
# CLI Interface Manifest projection (the shape shipped as cli.json).
INTERFACE_MANIFEST = {
    "service": "email",
    "manifest_version": "0.1.0",
    "interface": "cli",
    "entities": [{
        "entity": "Email",
        "commands": {
            "list": {
                "tool": "list_emails",
                "params": [{
                    "name": "folder_name",
                    "type": "string",
                    "required": False,
                    "enum": sorted(VALID_FOLDERS),
                }],
            },
        },
    }],
    "actions": [{
        "name": "show_data",
        "tool": "show_data",
        "params": [],
        "description": "Show raw email data",
    }],
}


class EmailService(BaseService):
    """
    Email MCP Service.

    Provides MCP tools for interacting with a simulated email system,
    including sending, receiving, organizing emails in folders,
    searching, replying, forwarding, and managing attachments.

    Requires PostgreSQL database via DATABASE_URL environment variable.

    Example usage:
        # Requires DATABASE_URL env var
        service = EmailService(
            connection_string=os.environ["DATABASE_URL"],
            user_email="user@example.com"
        )
    """

    def __init__(self, connection_string: str, user_email: str = "user@example.com"):
        """
        Initialize the Email MCP service.

        Args:
            connection_string: PostgreSQL database connection string.
                              e.g., "postgresql://user:pass@localhost:5432/email"
            user_email: The email address of the current user.
        """
        # Store user email
        self.user_email = user_email

        # Initialize database with PostgreSQL backend
        self.db = EmailDatabase(connection_string=connection_string)

        # Initialize base service with database
        super().__init__("email", db=self.db)

        # Register REST /api/reset endpoint (replaces MCP tool for reset)
        import json as _json
        from starlette.requests import Request as _Request
        from starlette.responses import Response as _Response

        @self.mcp.custom_route("/api/reset", methods=["POST"])
        async def reset_endpoint(request: _Request) -> _Response:
            body = await request.body()
            mock_data_path = None
            if body:
                data = _json.loads(body)
                mock_data_path = data.get("mock_data_path")
            result = self.reset_data(mock_data_path or self._mock_data_path)
            return _Response(
                content=_json.dumps({"ok": True, "message": result}),
                media_type="application/json",
            )

        @self.mcp.custom_route("/agentenv/interface-manifest", methods=["GET"])
        async def interface_manifest_index(request: _Request) -> _Response:
            return _Response(
                content=_json.dumps([INTERFACE_MANIFEST["interface"]]),
                media_type="application/json",
            )

        @self.mcp.custom_route("/agentenv/interface-manifest/{interface}", methods=["GET"])
        async def interface_manifest_endpoint(request: _Request) -> _Response:
            if request.path_params["interface"] != INTERFACE_MANIFEST["interface"]:
                return _Response(status_code=404)
            return _Response(
                content=_json.dumps(INTERFACE_MANIFEST),
                media_type="application/json",
            )

        # Register all Email-specific tools
        self._register_email_tools()

    # =========================================================================
    # EMAIL TOOLS
    # =========================================================================

    def _register_email_tools(self):
        """Register email-related MCP tools."""

        @self.mcp.tool(name="list_emails")
        def list_emails(
            folder_name: Annotated[str, Field(description="The folder to list emails from (INBOX, SENT, DRAFT, TRASH)")] = "INBOX",
            offset: Annotated[int, Field(description="The offset of the first email to return (0-based)")] = 0,
            limit: Annotated[int, Field(description="The maximum number of emails to return")] = 10,
        ) -> str:
            """List emails in a folder with pagination support."""
            folder_name = folder_name.upper()
            if folder_name not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid folder: {folder_name}")
                return json.dumps(error.to_dict())

            result = self.db.list_emails(folder_name, offset, limit)
            return json.dumps(result.to_dict())

        @self.mcp.tool(name="get_email_by_id")
        def get_email_by_id(
            email_id: Annotated[str, Field(description="The unique ID of the email to retrieve")],
            folder_name: Annotated[str, Field(description="The folder to search in (INBOX, SENT, DRAFT, TRASH)")] = "INBOX",
        ) -> str:
            """Get an email by its unique ID. Marks the email as read upon retrieval."""
            folder_name = folder_name.upper()
            if folder_name not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid folder: {folder_name}")
                return json.dumps(error.to_dict())

            email = self.db.get_email_by_id(email_id, folder_name)
            if not email:
                error = EmailError(error="email_not_found", message=f"Email {email_id} not found in {folder_name}")
                return json.dumps(error.to_dict())

            # Mark as read
            self.db.mark_as_read(email_id, folder_name)
            email.is_read = True

            return json.dumps(email.to_dict())

        @self.mcp.tool(name="get_email_by_index")
        def get_email_by_index(
            idx: Annotated[int, Field(description="The index of the email to retrieve (0-based)")],
            folder_name: Annotated[str, Field(description="The folder to search in (INBOX, SENT, DRAFT, TRASH)")] = "INBOX",
        ) -> str:
            """Get an email by its index position in a folder. Marks the email as read upon retrieval."""
            folder_name = folder_name.upper()
            if folder_name not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid folder: {folder_name}")
                return json.dumps(error.to_dict())

            if idx < 0:
                error = EmailError(error="invalid_index", message="Index must be non-negative")
                return json.dumps(error.to_dict())

            email = self.db.get_email_by_index(idx, folder_name)
            if not email:
                error = EmailError(error="index_out_of_range", message=f"Index {idx} out of range for {folder_name}")
                return json.dumps(error.to_dict())

            # Mark as read
            self.db.mark_as_read(email.email_id, folder_name)
            email.is_read = True

            return json.dumps(email.to_dict())

        @self.mcp.tool(name="send_email")
        def send_email(
            recipients: Annotated[Optional[List[str]], Field(description="List of recipient email addresses")] = None,
            subject: Annotated[str, Field(description="The subject of the email")] = "",
            content: Annotated[str, Field(description="The body content of the email")] = "",
            cc: Annotated[Optional[List[str]], Field(description="List of CC recipient email addresses")] = None,
            attachment_paths: Annotated[Optional[List[str]], Field(description="List of file paths for attachments")] = None,
        ) -> str:
            """Send an email to the specified recipients. The email is added to the SENT folder."""
            if recipients is None:
                recipients = []
            if cc is None:
                cc = []
            if attachment_paths is None:
                attachment_paths = []

            # Validate email addresses
            for email_addr in recipients + cc:
                if "@" not in email_addr:
                    error = EmailError(error="invalid_email", message=f"Invalid email address: {email_addr}")
                    return json.dumps(error.to_dict())

            # Create attachments dict (simplified - just store filenames)
            attachments = {}
            for path in attachment_paths:
                filename = os.path.basename(path)
                attachments[filename] = f"[Attachment: {filename}]"

            email = Email(
                email_id=generate_email_id(),
                sender=self.user_email,
                recipients=recipients,
                subject=subject,
                content=content,
                cc=cc,
                attachments=attachments,
                timestamp=time.time(),
                is_read=True,  # Sent emails are considered read
            )

            self.db.create_email(email, EmailFolderName.SENT.value)
            return json.dumps({"email_id": email.email_id})

        @self.mcp.tool(name="reply_to_email")
        def reply_to_email(
            email_id: Annotated[str, Field(description="The ID of the email to reply to")],
            content: Annotated[str, Field(description="The content of the reply")],
            folder_name: Annotated[str, Field(description="The folder containing the original email (INBOX, SENT, DRAFT, TRASH)")] = "INBOX",
            cc: Annotated[Optional[List[str]], Field(description="Additional CC recipients for the reply")] = None,
        ) -> str:
            """Reply to an existing email. The reply is sent to the original sender and added to the SENT folder."""
            folder_name = folder_name.upper()
            if folder_name not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid folder: {folder_name}")
                return json.dumps(error.to_dict())

            if cc is None:
                cc = []

            # Find original email
            original_email = self.db.get_email_by_id(email_id, folder_name)
            if not original_email:
                error = EmailError(error="email_not_found", message=f"Email {email_id} not found in {folder_name}")
                return json.dumps(error.to_dict())

            # Create reply subject
            reply_subject = original_email.subject
            if not reply_subject.startswith("Re: "):
                reply_subject = f"Re: {reply_subject}"

            reply_email = Email(
                email_id=generate_email_id(),
                sender=self.user_email,
                recipients=[original_email.sender],
                subject=reply_subject,
                content=content,
                cc=cc,
                parent_id=original_email.email_id,
                timestamp=time.time(),
                is_read=True,
            )

            self.db.create_email(reply_email, EmailFolderName.SENT.value)
            return json.dumps({"email_id": reply_email.email_id})

        @self.mcp.tool(name="forward_email")
        def forward_email(
            email_id: Annotated[str, Field(description="The ID of the email to forward")],
            recipients: Annotated[List[str], Field(description="List of recipients to forward the email to")],
            content: Annotated[str, Field(description="Additional content to add before the forwarded message")] = "",
            folder_name: Annotated[str, Field(description="The folder containing the original email (INBOX, SENT, DRAFT, TRASH)")] = "INBOX",
        ) -> str:
            """Forward an existing email to new recipients. Includes original content and attachments."""
            folder_name = folder_name.upper()
            if folder_name not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid folder: {folder_name}")
                return json.dumps(error.to_dict())

            # Find original email
            original_email = self.db.get_email_by_id(email_id, folder_name)
            if not original_email:
                error = EmailError(error="email_not_found", message=f"Email {email_id} not found in {folder_name}")
                return json.dumps(error.to_dict())

            # Validate email addresses
            for email_addr in recipients:
                if "@" not in email_addr:
                    error = EmailError(error="invalid_email", message=f"Invalid email address: {email_addr}")
                    return json.dumps(error.to_dict())

            # Create forward subject
            forward_subject = original_email.subject
            if not forward_subject.startswith("Fwd: "):
                forward_subject = f"Fwd: {forward_subject}"

            # Create forward content
            original_content = (
                f"\n\n--- Forwarded message ---\n"
                f"From: {original_email.sender}\n"
                f"To: {', '.join(original_email.recipients)}\n"
                f"CC: {', '.join(original_email.cc)}\n"
                f"Subject: {original_email.subject}\n"
                f"Content: {original_email.content}"
            )
            forward_content = content + original_content

            forward_email_obj = Email(
                email_id=generate_email_id(),
                sender=self.user_email,
                recipients=recipients,
                subject=forward_subject,
                content=forward_content,
                parent_id=original_email.email_id,
                attachments=original_email.attachments.copy(),  # Forward attachments
                timestamp=time.time(),
                is_read=True,
            )

            self.db.create_email(forward_email_obj, EmailFolderName.SENT.value)
            return json.dumps({"email_id": forward_email_obj.email_id})

        @self.mcp.tool(name="move_email")
        def move_email(
            email_id: Annotated[str, Field(description="The ID of the email to move")],
            from_folder: Annotated[str, Field(description="The source folder (INBOX, SENT, DRAFT, TRASH)")],
            to_folder: Annotated[str, Field(description="The destination folder (INBOX, SENT, DRAFT, TRASH)")],
        ) -> str:
            """Move an email from one folder to another."""
            from_folder = from_folder.upper()
            to_folder = to_folder.upper()

            if from_folder not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid source folder: {from_folder}")
                return json.dumps(error.to_dict())

            if to_folder not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid destination folder: {to_folder}")
                return json.dumps(error.to_dict())

            email = self.db.move_email(email_id, from_folder, to_folder)
            if not email:
                error = EmailError(error="email_not_found", message=f"Email {email_id} not found in {from_folder}")
                return json.dumps(error.to_dict())

            response = EmailActionResponse(email_id=email.email_id, moved_to=to_folder)
            return json.dumps(response.to_dict())

        @self.mcp.tool(name="delete_email")
        def delete_email(
            email_id: Annotated[str, Field(description="The ID of the email to delete")],
            folder_name: Annotated[str, Field(description="The folder containing the email (INBOX, SENT, DRAFT, TRASH)")] = "INBOX",
        ) -> str:
            """Delete an email by moving it to TRASH. If already in TRASH, permanently deletes it."""
            folder_name = folder_name.upper()
            if folder_name not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid folder: {folder_name}")
                return json.dumps(error.to_dict())

            if folder_name == EmailFolderName.TRASH.value:
                # Permanently delete from trash
                if self.db.delete_email(email_id, folder_name):
                    response = EmailActionResponse(email_id=email_id, permanently_deleted=True)
                    return json.dumps(response.to_dict())
                else:
                    error = EmailError(error="email_not_found", message=f"Email {email_id} not found in {folder_name}")
                    return json.dumps(error.to_dict())
            else:
                # Move to trash
                email = self.db.move_email(email_id, folder_name, EmailFolderName.TRASH.value)
                if not email:
                    error = EmailError(error="email_not_found", message=f"Email {email_id} not found in {folder_name}")
                    return json.dumps(error.to_dict())

                response = EmailActionResponse(email_id=email.email_id, moved_to=EmailFolderName.TRASH.value)
                return json.dumps(response.to_dict())

        @self.mcp.tool(name="search_emails")
        def search_emails(
            query: Annotated[str, Field(description="The search query string (case-insensitive)")],
            folder_name: Annotated[str, Field(description="The folder to search in (INBOX, SENT, DRAFT, TRASH)")] = "INBOX",
        ) -> str:
            """Search for emails in a folder. Matches sender, recipients, subject, and content."""
            folder_name = folder_name.upper()
            if folder_name not in VALID_FOLDERS:
                error = EmailError(error="invalid_folder", message=f"Invalid folder: {folder_name}")
                return json.dumps(error.to_dict())

            emails = self.db.search_emails(query, folder_name)
            return json.dumps([e.to_dict() for e in emails])

        @self.mcp.tool(name="show_data")
        def show_data(
            offset: Annotated[int, Field(description="Number of items to skip per folder")] = 0,
            limit: Annotated[int, Field(description="Maximum number of items per folder")] = 100,
        ) -> str:
            """Show raw email data as JSON. Returns {folders: {folder_name: [emails]}}."""
            folders = self.db.get_all_emails_by_folder(offset, limit)
            response = ShowDataResponse(folders=folders)
            return json.dumps(response.to_dict())


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import os

    connection_string = os.environ.get("DATABASE_URL")
    if not connection_string:
        raise ValueError("DATABASE_URL environment variable is required")

    user_email = os.environ.get("USER_EMAIL", "user@example.com")

    service = EmailService(connection_string=connection_string, user_email=user_email)
    service.run()

