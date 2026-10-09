"""Path helpers shared by the artifact loaders, bundle folders and build contexts."""

from __future__ import annotations

import os

# What the OS or Python writes into a folder on its own. In an artifact or skill folder, everything
# else is content, dot files included.
LEAVINGS = frozenset({".DS_Store", "Thumbs.db", "desktop.ini", "__pycache__"})


def validate_relative_filename(filename: str) -> None:
    """Reject absolute paths and ``..`` traversal.

    Universe filenames are staged untrusted — one bad entry otherwise escapes
    the destination directory via ``posixpath.join``.
    """
    if not filename:
        raise ValueError("universe contains an empty filename")
    if os.path.isabs(filename) or filename.startswith("/"):
        raise ValueError(f"Filename {filename!r} must be relative")
    parts = filename.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        raise ValueError(f"Filename {filename!r} must not contain '..' segments")
