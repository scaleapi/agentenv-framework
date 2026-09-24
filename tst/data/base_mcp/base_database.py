"""
Base Database abstraction for Synthetic MCP Servers.

This module provides an abstract base class for database implementations,
allowing services to use different database backends (SQLite, PostgreSQL, etc.)
through a consistent interface.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Optional, TypeVar

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class DatabaseBackend(Enum):
    """Supported database backends."""

    SQLITE_MEMORY = "sqlite_memory"
    SQLITE_FILE = "sqlite_file"
    POSTGRESQL = "postgresql"


class BaseSQLAlchemyModel(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


T = TypeVar("T", bound=BaseSQLAlchemyModel)


class BaseDatabase(ABC):
    """
    Abstract base class for database implementations.

    Provides common functionality for database operations including:
    - Connection management with swappable backends
    - Session handling
    - Reset and data loading capabilities

    Subclasses must implement:
    - _create_tables(): Define and create the specific tables for the service
    - reset_all(): Clear all data from the database
    - load_from_json(): Load mock data from a JSON file

    Example usage:
        class MyServiceDatabase(BaseDatabase):
            def _create_tables(self):
                # Define your SQLAlchemy models and create tables
                self.Base.metadata.create_all(self.engine)

            def reset_all(self):
                self.Base.metadata.drop_all(self.engine)
                self._create_tables()

            def load_from_json(self, json_path: str) -> Dict[str, int]:
                # Load your specific data format
                ...
    """

    def __init__(
        self,
        backend: DatabaseBackend = DatabaseBackend.SQLITE_MEMORY,
        connection_string: Optional[str] = None,
        db_path: Optional[str] = None,
        **engine_kwargs: Any,
    ):
        """
        Initialize the database connection.

        Args:
            backend: The database backend to use.
            connection_string: Full connection string (overrides backend/db_path).
                              e.g., "postgresql://user:pass@localhost:5432/dbname"
            db_path: Path to database file (for SQLite file backend).
            **engine_kwargs: Additional arguments to pass to create_engine.
        """
        self.backend = backend
        self._engine_kwargs = engine_kwargs
        self._connection_string = connection_string or self._build_connection_string(
            backend, db_path
        )

        # Create engine
        self.engine = self._create_engine()

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables for this database
        self._create_tables()

    def _build_connection_string(
        self, backend: DatabaseBackend, db_path: Optional[str] = None
    ) -> str:
        """
        Build a connection string based on the backend type.

        Args:
            backend: The database backend to use.
            db_path: Path to database file (for SQLite file backend).

        Returns:
            A SQLAlchemy connection string.
        """
        if backend == DatabaseBackend.SQLITE_MEMORY:
            return "sqlite:///:memory:"
        elif backend == DatabaseBackend.SQLITE_FILE:
            if not db_path:
                raise ValueError("db_path is required for SQLite file backend")
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path}"
        elif backend == DatabaseBackend.POSTGRESQL:
            raise ValueError(
                "PostgreSQL requires a full connection_string. "
                "Use connection_string='postgresql://user:pass@host:port/dbname'"
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _create_engine(self) -> Engine:
        """
        Create and configure the SQLAlchemy engine.

        Returns:
            Configured SQLAlchemy Engine.
        """
        # Default engine settings
        engine_settings = {"echo": False}
        engine_settings.update(self._engine_kwargs)

        engine = create_engine(
            self._connection_string, query_cache_size=0, **engine_settings
        )

        # Apply backend-specific configurations
        self._configure_engine(engine)

        return engine

    def _configure_engine(self, engine: Engine) -> None:
        """
        Apply backend-specific engine configurations.

        Args:
            engine: The SQLAlchemy engine to configure.
        """
        # SQLite-specific: Enable foreign keys
        if self.backend in (DatabaseBackend.SQLITE_MEMORY, DatabaseBackend.SQLITE_FILE):

            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.

        Yields:
            A SQLAlchemy Session that will be committed on success
            or rolled back on exception.
        """
        session = self.SessionLocal()
        session.expire_all()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @abstractmethod
    def _create_tables(self) -> None:
        """
        Create all tables for this database.

        Subclasses must implement this to define their specific schema.
        """
        pass

    @abstractmethod
    def reset_all(self) -> None:
        """
        Drop all tables and recreate them, clearing all data.

        Subclasses must implement this to handle their specific tables.
        """
        pass

    @abstractmethod
    def load_from_json(self, json_path: str) -> Dict[str, int]:
        """
        Load mock data from a JSON file.

        Args:
            json_path: Path to the JSON file containing mock data.

        Returns:
            Dict with counts of loaded entities by type.
        """
        pass

    @abstractmethod
    def export_state(self) -> Dict[str, Any]:
        """
        Export the current database state as a dict.

        The returned dict must use the same keys/format that load_from_json() expects,
        so the data can be round-tripped: export_state() -> JSON file -> load_from_json().
        """
        pass

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about the current database connection.

        Returns:
            Dict containing connection details (backend type, connection status, etc.)
        """
        return {
            "backend": self.backend.value,
            "connection_string": self._sanitize_connection_string(),
            "is_connected": self._check_connection(),
        }

    def _sanitize_connection_string(self) -> str:
        """
        Return a sanitized version of the connection string (hides passwords).

        Returns:
            Connection string with sensitive data masked.
        """
        conn_str = self._connection_string
        # Mask password in connection strings like postgresql://user:password@host
        import re

        return re.sub(r":([^:@]+)@", ":****@", conn_str)

    def _check_connection(self) -> bool:
        """
        Check if the database connection is working.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
