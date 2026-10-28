"""A computer-recording/v1 manifest, as ``end`` returns it and a recording's artifact keeps it. A manifest rebuilt from
what a recording's namespace holds, for a driver that forgot the recording, knows only each segment's index and path,
so the rest is optional here; a driver's own manifest has all of it."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

EndedBy = Literal["end", "reset", "teardown", "limit", "error"]


class ManifestSegment(BaseModel):
    """One segment: ``started_at`` and ``ended_at`` are its first and last frames' capture times."""

    index: int
    path: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frames: Optional[int] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    uploaded: bool
    upload_error: Optional[str] = None

    @property
    def measured(self) -> bool:
        """Whether its times, size, frame count and digest are all known."""
        return None not in (self.started_at, self.ended_at, self.width, self.height, self.frames, self.bytes,
                            self.sha256)


class RecordingManifest(BaseModel):
    recording_id: Optional[str]
    scope: Optional[str] = None
    format: Optional[str] = None
    fps: Optional[int] = None
    cursor: Optional[bool] = None
    source: Optional[Literal["stream", "capture"]] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    ended_by: EndedBy
    segments: list[ManifestSegment]
    error: Optional[str] = None

    def wire(self) -> dict:
        """As JSON goes on the wire, without the fields it doesn't know."""
        return self.model_dump(mode="json", exclude_none=True)
