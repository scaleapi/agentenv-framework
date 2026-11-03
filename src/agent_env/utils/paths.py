"""Path helpers shared by the artifact loaders, bundle folders and build contexts."""

from __future__ import annotations

import errno
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


def validate_file_path(path: str) -> None:
    """Reject anything but plain file names joined by ``/``: an env stages each file of a loaded environment at host and
    container paths built from its path."""
    for name in path.split("/"):
        if name in ("", ".", "..") or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
            raise ValueError(f"Artifact file path {path!r} must be plain file names joined by '/', with no '.', '..', "
                             "'\\' or control characters")


def validate_file_name(filename: str) -> None:
    """Reject anything but one plain file name: an env stages a loaded artifact at host and container paths built from it."""
    if filename in ("", ".", "..") or "/" in filename or "\\" in filename or any(ord(c) < 32 or ord(c) == 127 for c in filename):
        raise ValueError(f"Artifact filename {filename!r} must be a single file name, with no '/', '\\' or control characters")


def os_reason(e: OSError) -> str:
    """What an OSError says went wrong, as a problem message names it."""
    if e.errno == errno.ELOOP:
        return "a symlink loop, or too many links"
    if isinstance(e, PermissionError):
        return "permission denied"
    return e.strerror or str(e)
