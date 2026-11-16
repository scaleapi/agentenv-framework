"""The computer_* tools' results: each tool's output schema, and the one way a result is sent, so what a tool
answers is what its schema says."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel

from .operations import Effect

#: The ``_meta`` key of a tool answer that carries a frame: when the driver captured it, in real time (RFC 3339, UTC).
CAPTURED_AT_META = "agentenv.io/capturedAt"


class DisplayData(BaseModel):
    width: int
    height: int
    scale: float


class DialogData(BaseModel):
    type: str
    message: str
    default_text: str = ""


class FileChooserData(BaseModel):
    multiple: bool


class PageData(BaseModel):
    """The page a frame shows, on a computer whose screen is a website browser's active tab."""
    tab: str
    url: str
    title: str
    dialog: Optional[DialogData] = None
    file_chooser: Optional[FileChooserData] = None


class FrameData(BaseModel):
    """A frame as an agent sees it. When it was captured isn't part of it: the driver keeps real time, which in a
    computer on the universe's clock would tell the agent the real date; the tool's answer carries it in its ``_meta``
    (``CAPTURED_AT_META``) for the callers that line frames up with a recording."""
    frame_id: str
    width: int
    height: int
    display: DisplayData
    parent_frame_id: Optional[str] = None
    page: Optional[PageData] = None


class ScreenshotResult(FrameData):
    summary: str


class EffectData(BaseModel):
    focus_changed: Optional[bool] = None
    pointer_moved: Optional[bool] = None
    clipboard_used: Optional[bool] = None
    detail: Optional[str] = None

    @classmethod
    def of(cls, effect: object) -> EffectData | None:
        """What an input operation returned, when it is an Effect that says something."""
        if not isinstance(effect, Effect):
            return None
        data = cls(**vars(effect))
        return data if data.model_dump(exclude_none=True) else None


class InputResult(BaseModel):
    summary: str
    applied: Literal["yes", "no", "unknown"]
    delivery: Literal["foreground", "background"]
    others_since_frame: int
    effect: Optional[EffectData] = None
    frame: Optional[FrameData] = None


class TabData(BaseModel):
    tab: str
    url: str
    title: str
    active: bool
    #: The label of the lease it's in, None for the browser's default context.
    context: Optional[str] = None


class TabsResult(InputResult):
    tabs: list[TabData]


class WaitResult(BaseModel):
    summary: str
    waited_ms: int
    frame: Optional[FrameData] = None


class Point(BaseModel):
    x: int
    y: int


class PointerPositionResult(BaseModel):
    summary: str
    screen: Point
    frame_id: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    inside_frame: Optional[bool] = None


class PointerData(BaseModel):
    id: str
    kind: Literal["foreground", "background"]
    label: str
    default: bool


class PointerListResult(BaseModel):
    summary: str
    pointers: list[PointerData]


def send(result: BaseModel, image: tuple[str, str] | None = None,
         captured_at: datetime | None = None) -> CallToolResult:
    """A result as a tool answers: the summary as the text block, then the image (base64, MIME type) when there is
    one, and the result as structured content with its unset fields left out; when its frame was ``captured_at``,
    that real time in its ``_meta``, which a client doesn't show its model."""
    content: list = [TextContent(type="text", text=result.summary)]
    if image is not None:
        content.append(ImageContent(type="image", data=image[0], mimeType=image[1]))
    meta = {CAPTURED_AT_META: captured_at.isoformat().replace("+00:00", "Z")} if captured_at is not None else None
    return CallToolResult(content=content, structuredContent=result.model_dump(mode="json", exclude_none=True),
                          _meta=meta)
