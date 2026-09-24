"""
BaseService package for Synthetic MCP Servers.

Provides base classes for creating MCP services with database support.
"""

from .base_service import BaseService
from .base_database import BaseDatabase, DatabaseBackend, BaseSQLAlchemyModel

__all__ = ["BaseService", "BaseDatabase", "DatabaseBackend", "BaseSQLAlchemyModel"]

