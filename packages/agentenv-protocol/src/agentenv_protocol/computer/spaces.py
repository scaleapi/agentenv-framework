"""Coordinate spaces: points in a frame's pixels (``frame``) or on a 0..norm_scale grid over it (``norm``), mapped
to the frame's pixels and refused, never clamped, when outside it."""
from __future__ import annotations

import math

from .errors import InvalidArgument, OutOfBounds
from .frames import Frame
from .settings import Space

SPACES = ("frame", "norm")


def frame_point(frame: Frame, x: float, y: float, space: Space, norm_scale: int) -> tuple[float, float]:
    if space == "norm":
        if not (0 <= x <= norm_scale and 0 <= y <= norm_scale):
            raise OutOfBounds(f"({x:g}, {y:g}) is outside the norm range 0..{norm_scale} of frame {frame.id}")
        fx, fy = min(x * frame.width / norm_scale, frame.width - 1), min(y * frame.height / norm_scale, frame.height - 1)
        return fx, fy
    if not frame.contains(x, y):
        raise OutOfBounds(f"({x:g}, {y:g}) is outside frame {frame.id} ({frame.width}×{frame.height})")
    return x, y


def frame_region(frame: Frame, region: list, space: Space, norm_scale: int) -> tuple[int, int, int, int]:
    """``[x0, y0, x1, y1]`` as a pixel box within the frame, end exclusive."""
    if len(region) != 4:
        raise InvalidArgument(f"region is [x0, y0, x1, y1], got {len(region)} numbers")
    x0, y0, x1, y1 = region
    if space == "norm":
        if not all(0 <= v <= norm_scale for v in region):
            raise OutOfBounds(f"region {list(region)} is outside the norm range 0..{norm_scale} of frame {frame.id}")
        x0, x1 = x0 * frame.width / norm_scale, x1 * frame.width / norm_scale
        y0, y1 = y0 * frame.height / norm_scale, y1 * frame.height / norm_scale
    box = (int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1)))
    if not (0 <= box[0] < box[2] <= frame.width and 0 <= box[1] < box[3] <= frame.height):
        raise OutOfBounds(f"region {list(region)} is not a non-empty box within frame {frame.id} "
                          f"({frame.width}×{frame.height})")
    return box


def check_space(space: str | None, norm_scale: int | None, default_space: Space, default_scale: int) -> tuple:
    space = space or default_space
    if space not in SPACES:
        raise InvalidArgument(f"space is 'frame' or 'norm', got {space!r}")
    scale = default_scale if norm_scale is None else norm_scale
    if not (isinstance(scale, int) and 1 <= scale <= 1_000_000):
        raise InvalidArgument(f"norm_scale is an integer from 1 to 1000000, got {norm_scale!r}")
    return space, scale
