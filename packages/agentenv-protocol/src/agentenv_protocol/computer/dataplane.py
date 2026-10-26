"""The data plane on a computer: ``data/add`` writes files under the session user's home, ``data/get`` returns the
env's ``data_paths``. Each file part names its path, relative to the home, under the ``data-objects/v1`` marker."""
from __future__ import annotations

import base64
import binascii
import mimetypes
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx
from pydantic import ValidationError

from ..agent_env_environment import add_data, get_data
from ..transfers import NamespaceUploader, ReadObject, WriteNamespaceGrant, download
from ..types import DATA_OBJECTS_EXTENSION_URI, FilePart, FileWithBytes, FileWithUri, uploaded_file_part, uploaded_object_path

if TYPE_CHECKING:
    from .runtime import ComputerRuntime
    from .settings import ComputerSettings

#: The most an https URL part may fetch; a read grant carries its own limit.
MAX_URL_BYTES = 1 << 30


class DataPlane:
    """The data plane handlers, mixed into ComputerDriverEnvironment."""

    _computer: ComputerRuntime
    settings: ComputerSettings

    @add_data
    async def _computer_data_add(self, parts: list) -> None:
        files = {}
        for part in parts:
            try:
                path = uploaded_object_path(part) if isinstance(part, FilePart) else None
            except ValueError:
                marker = part.metadata[DATA_OBJECTS_EXTENSION_URI]
                raise ValueError(f"the part's path {marker.get('path')!r} is not a normalized path relative to "
                                 "the home") from None
            if path is None:
                raise ValueError(f"data/add on a computer takes file parts naming their path under "
                                 f"{DATA_OBJECTS_EXTENSION_URI}, got a {part.kind} part")
            files[path] = await _fetch(part)
        await self._computer.call("write_files", files)

    @get_data
    async def _computer_data_get(self, write_namespace: Optional[WriteNamespaceGrant] = None) -> list:
        files = _data_files(Path.home(), self.settings.data_paths)
        if write_namespace is not None:
            uploader = NamespaceUploader(write_namespace)
            parts = []
            for relative, source in files:
                await uploader.upload(relative, source)
                parts.append(uploaded_file_part(relative, name=source.name, mime_type=_mime(source)))
            return parts
        return [
            FilePart(file=FileWithBytes(bytes=base64.b64encode(source.read_bytes()).decode("ascii"),
                                        name=source.name, mimeType=_mime(source)),
                     metadata={DATA_OBJECTS_EXTENSION_URI: {"path": relative}})
            for relative, source in files
        ]


async def _fetch(part: FilePart) -> bytes:
    """A part's bytes: inline, from a read grant under the marker, or from an https URL."""
    marker = part.metadata[DATA_OBJECTS_EXTENSION_URI]
    if "read" in marker:
        try:
            grant = ReadObject.model_validate(marker["read"])
        except ValidationError:
            raise ValueError("the part's read grant is not a valid ReadObject") from None
        with tempfile.TemporaryDirectory() as scratch:
            destination = Path(scratch) / "object"
            await download(grant, destination)
            return destination.read_bytes()
    if isinstance(part.file, FileWithBytes):
        try:
            return base64.b64decode(part.file.bytes, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("the part's bytes are not base64") from None
    return await _get_url(part.file)


async def _get_url(file: FileWithUri) -> bytes:
    if not file.uri.startswith("https://"):
        raise ValueError("a file part's uri is an https URL")
    data = bytearray()
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        async with client.stream("GET", file.uri) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                data += chunk
                if len(data) > MAX_URL_BYTES:
                    raise ValueError(f"the file at the part's uri is over {MAX_URL_BYTES} bytes")
    return bytes(data)


def _data_files(home: Path, data_paths: list[str]) -> list[tuple[str, Path]]:
    """Every file under the data paths, by its path relative to the home; a path not there yet is skipped."""
    root = home.resolve()
    found: dict[str, Path] = {}
    for relative in data_paths:
        target = home / relative
        if not target.exists() or not target.resolve().is_relative_to(root):
            continue
        for path in sorted(target.rglob("*")) if target.is_dir() else [target]:
            if path.is_file() and path.resolve().is_relative_to(root):
                found[path.relative_to(home).as_posix()] = path
    return list(found.items())


def _mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
