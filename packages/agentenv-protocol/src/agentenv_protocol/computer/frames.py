"""Frames: every screenshot, kept by id so any caller can point into it later, and its mapping to screen pixels."""
from __future__ import annotations

import base64
import io
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from PIL import Image as PILImage

from .errors import Internal, StaleFrame
from .operations import Display, Image
from .results import DisplayData, FrameData

KEEP = 64
#: The JPEG quality a screenshot answers with when it names none.
JPEG_QUALITY = 85


@dataclass(frozen=True)
class Frame:
    """A capture, or a crop of one: ``origin`` is where this frame's pixels start in the root capture, and the root
    capture's ``root_size`` image covers the whole ``display``. The pixels are kept as the PNG every screenshot
    answers with, a tenth or less of them raw, and decoded only to crop or to answer in JPEG."""

    id: str
    png: bytes = field(repr=False)
    size: tuple[int, int]
    display: Display
    origin: tuple[int, int]
    root_size: tuple[int, int]
    parent_id: str | None
    captured_at: datetime
    inputs: tuple = field(repr=False)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def image(self) -> PILImage.Image:
        image = PILImage.open(io.BytesIO(self.png))
        image.load()
        return image

    def to_screen(self, fx: float, fy: float) -> tuple[int, int]:
        """The screen pixel under frame pixel (fx, fy); exact at identity scale."""
        rx, ry = self.origin[0] + fx, self.origin[1] + fy
        return (int(math.floor(rx * self.display.width / self.root_size[0])),
                int(math.floor(ry * self.display.height / self.root_size[1])))

    def from_screen(self, sx: float, sy: float) -> tuple[int, int]:
        rx = sx * self.root_size[0] / self.display.width
        ry = sy * self.root_size[1] / self.display.height
        return int(math.floor(rx - self.origin[0])), int(math.floor(ry - self.origin[1]))

    def contains(self, fx: float, fy: float) -> bool:
        return 0 <= fx < self.width and 0 <= fy < self.height

    def result(self) -> FrameData:
        return FrameData(
            frame_id=self.id,
            width=self.width,
            height=self.height,
            display=DisplayData(width=self.display.width, height=self.display.height, scale=self.display.scale),
            parent_frame_id=self.parent_id,
            captured_at=self.captured_at.isoformat().replace("+00:00", "Z"),
        )

    def encode(self, fmt: str = "png", quality: int | None = None) -> tuple[str, str]:
        """The frame as base64 and its MIME type."""
        if fmt != "jpeg":
            return base64.b64encode(self.png).decode("ascii"), "image/png"
        buffer = io.BytesIO()
        self.image.convert("RGB").save(buffer, format="JPEG", quality=quality or JPEG_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/jpeg"


class FrameStore:
    """The last ``KEEP`` frames by id, ``f<n>``, and each caller's latest; a frame no longer kept is unknown."""

    def __init__(self, keep: int = KEEP) -> None:
        self._keep = keep
        self._frames: OrderedDict[str, Frame] = OrderedDict()
        self._next = 1
        self._latest: dict[str, str] = {}

    def add(self, image: PILImage.Image, display: Display, inputs: tuple, *, identity: str,
            parent: Frame | None = None, origin: tuple[int, int] = (0, 0)) -> Frame:
        frame = Frame(
            id=f"f{self._next}",
            png=_png(image),
            size=image.size,
            display=parent.display if parent is not None else display,
            origin=(parent.origin[0] + origin[0], parent.origin[1] + origin[1]) if parent is not None else (0, 0),
            root_size=parent.root_size if parent is not None else image.size,
            parent_id=parent.id if parent is not None else None,
            captured_at=parent.captured_at if parent is not None else datetime.now(timezone.utc),
            inputs=parent.inputs if parent is not None else inputs,
        )
        self._next += 1
        self._frames[frame.id] = frame
        while len(self._frames) > self._keep:
            self._frames.popitem(last=False)
        self._latest[identity] = frame.id
        return frame

    def get(self, frame_id: str | None, identity: str) -> Frame:
        """The named frame, or the caller's latest when none is named."""
        if frame_id is None:
            frame_id = self._latest.get(identity)
            if frame_id is None:
                raise StaleFrame("there is no frame yet: take a screenshot first")
        frame = self._frames.get(frame_id)
        if frame is None:
            raise StaleFrame(f"frame {frame_id} is unknown or no longer kept: take a new screenshot")
        return frame

    def latest(self, identity: str) -> Frame | None:
        return self._frames.get(self._latest.get(identity, ""))

    def check_current(self, frame: Frame, now: Display) -> None:
        if (frame.display.width, frame.display.height, frame.display.scale) != (now.width, now.height, now.scale):
            raise StaleFrame(
                f"frame {frame.id} was captured at {_size(frame.display)} and the display is now {_size(now)}: "
                "take a new screenshot")


def to_pil(capture: object, display: Display) -> PILImage.Image:
    """A capture as a Pillow image, checked to be the display at native resolution."""
    if isinstance(capture, PILImage.Image):
        image = capture
    elif isinstance(capture, Image):
        if capture.mode in ("png", "jpeg"):
            image = PILImage.open(io.BytesIO(capture.data))
            image.load()
        elif capture.mode in ("BGRA", "BGRX"):
            image = PILImage.frombytes("RGB", (capture.width, capture.height), capture.data, "raw", "BGRX")
        else:
            image = PILImage.frombytes(capture.mode, (capture.width, capture.height), capture.data)
    else:
        raise Internal(f"@capture returned {type(capture).__name__}, not an Image")
    expected = (round(display.width * display.scale), round(display.height * display.scale))
    if image.size != expected:
        raise Internal(f"@capture returned {image.size[0]}×{image.size[1]}, but the display at scale "
                       f"{display.scale:g} is {expected[0]}×{expected[1]}")
    return image


def _png(image: PILImage.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _size(display: Display) -> str:
    return f"{display.width}×{display.height} at scale {display.scale:g}"
