"""Path helpers shared by the artifact loaders."""

from __future__ import annotations

import os


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
