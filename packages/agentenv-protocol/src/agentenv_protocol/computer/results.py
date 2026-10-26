"""The computer_* tools' results: each tool's output schema, and the one way a result is sent, so what a tool
answers is what its schema says."""
from __future__ import annotations

from typing import Literal, Optional

from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel

from .operations import Effect


class DisplayData(BaseModel):
    width: int
    height: int
    scale: float


class FrameData(BaseModel):
    frame_id: str
    width: int
    height: int
    display: DisplayData
    parent_frame_id: Optional[str] = None
    captured_at: str


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


def send(result: BaseModel, image: tuple[str, str] | None = None) -> CallToolResult:
    """A result as a tool answers: the summary as the text block, then the image (base64, MIME type) when there is
    one, and the result as structured content with its unset fields left out."""
    content: list = [TextContent(type="text", text=result.summary)]
    if image is not None:
        content.append(ImageContent(type="image", data=image[0], mimeType=image[1]))
    return CallToolResult(content=content, structuredContent=result.model_dump(mode="json", exclude_none=True))
