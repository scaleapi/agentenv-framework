"""What an entity's ``to_toml`` writes itself out through: names for what it references, its images, its files.

``to_toml`` is the inverse of ``from_toml``: an entity written out with it and read back with ``from_toml`` is the same
entity, on every field its toml carries. A type without it, or whose ``to_toml`` raises ``Unwritable``, can't be
written into a bundle folder, so ``agent-env bundle init`` leaves it in the store.
"""

from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_env.config import get_config

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact


class Unwritable(Exception):
    """Why an entity can't be written into a bundle folder."""


class ExportContext(ABC):
    """One entity being written out: its folder's files, and the toml values naming what it references."""

    @abstractmethod
    def ref(self, entity: Any) -> str | dict:
        """The toml value naming ``entity``: its name in the bundle, which writes it there too, or a pinned table."""

    @abstractmethod
    def image(self, artifact: DockerImageArtifact, key: str = "image") -> str | dict:
        """The toml value of the image key ``key``: built from this folder, a registry ref, or a pinned table."""

    @abstractmethod
    def file(self, path: str, object_url: str) -> None:
        """Write the object at ``object_url`` to ``path`` in this folder; raises Unwritable when it would leave it."""

    def left_out(self, field: str, why: str) -> None:
        """Note that the entity's ``field`` isn't written, since its toml carries no such field."""

    def tree(self, prefix_url: str) -> None:
        """Write every object under ``prefix_url`` to its path below the prefix, in this folder."""
        prefix = prefix_url if prefix_url.endswith("/") else f"{prefix_url}/"
        urls = sorted(url for url in get_config().get_object_store_at(prefix).list_at(prefix)
                      if url.startswith(prefix) and len(url) > len(prefix) and not url.endswith("/"))
        if not urls:
            raise Unwritable(f"nothing is stored under {prefix_url}")
        for url in urls:
            self.file(url[len(prefix):], url)


class FolderExport(ExportContext):
    """Writes an entity's files straight into ``folder``, as ``get --output-dir`` does; it takes no refs or images."""

    def __init__(self, folder: Path, on_file: Callable[[str, Path], None] | None = None):
        self.folder = folder
        self.on_file = on_file

    def ref(self, entity: Any) -> str | dict:
        raise Unwritable(f"it references {entity.type} {entity.id}, which a folder of its own files can't hold")

    def image(self, artifact: DockerImageArtifact, key: str = "image") -> str | dict:
        raise Unwritable(f"its {key} is an image, which a folder of its own files can't hold")

    def file(self, path: str, object_url: str) -> None:
        dest = self.folder / checked_path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        get_config().get_object_store_at(object_url).download_to_file(object_url, str(dest))
        if self.on_file is not None:
            self.on_file(path, dest)


def checked_path(path: str) -> str:
    """``path`` normalized as a POSIX path inside a folder; raises Unwritable when it's absolute, climbs out or is empty."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise Unwritable(f"{path!r} isn't a file's path")
    posix = path.replace("\\", "/")
    if posix.startswith("/") or ".." in posix.split("/"):
        raise Unwritable(f"{path!r} would be written outside its folder")
    normal = posixpath.normpath(posix)
    if normal == ".":
        raise Unwritable(f"{path!r} names a folder, not a file")
    return normal
