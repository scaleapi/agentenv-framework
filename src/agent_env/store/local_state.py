"""On-disk state directories for the local (no-infra) stores."""

from __future__ import annotations

from pathlib import Path


def ensure_state_dir(path: Path) -> None:
    """Create ``path`` if missing. A directory this creates gets a ``.gitignore`` of ``*``
    so generated state is never committed; a directory that already existed is left as is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        return
    (path / ".gitignore").write_text("*\n")
