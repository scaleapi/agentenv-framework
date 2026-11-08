"""The settings a computer driver reads at start, from one JSON value in ``AGENTENV_COMPUTER_SETTINGS``."""
from __future__ import annotations

import json
import os
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


SETTINGS_ENV = "AGENTENV_COMPUTER_SETTINGS"
#: The prefix a deploy gives the names of the computer's tools, so several computers serve them side by side.
TOOL_PREFIX_ENV = "AGENTENV_TOOL_PREFIX"
_TOOL_PREFIX = re.compile(r"[A-Za-z0-9_-]*")
#: The longest a tool prefix may be, its separator included, so a prefixed name stays well within the 64 characters
#: model APIs take for a tool's name.
MAX_TOOL_PREFIX = 16


def tool_prefix_problem(prefix: str) -> str | None:
    """What's wrong with ``prefix`` as a tool prefix, or None: only what a tool's name may hold, at most
    ``MAX_TOOL_PREFIX`` characters."""
    if not _TOOL_PREFIX.fullmatch(prefix):
        return f"tool prefix {prefix!r} holds characters a tool's name can't"
    if len(prefix) > MAX_TOOL_PREFIX:
        return f"tool prefix {prefix!r} is {len(prefix)} characters, and a tool prefix is at most {MAX_TOOL_PREFIX}"
    return None

Space = Literal["frame", "norm"]


class ComputerSettings(BaseModel):
    """The defaults the card reports and the framework extensions change."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    default_space: Space = "frame"
    default_norm_scale: int = Field(default=1000, ge=1, le=1_000_000)
    queue_wait_s: float = Field(default=30.0, gt=0, le=3600)
    tap_hold_ms: int = Field(default=50, ge=0, le=1000)
    type_delay_ms: int = Field(default=10, ge=0, le=1000)
    max_hold_ms: int = Field(default=10_000, ge=0, le=600_000)
    layout: str = "us"
    drag_duration_ms: int = Field(default=500, ge=0)
    exec_output_bytes: int = Field(default=1 << 20, gt=0)
    exec_timeout_s: float = Field(default=60.0, gt=0, le=3600)


def load_settings(environ: dict | None = None, model: type[ComputerSettings] = ComputerSettings) -> ComputerSettings:
    """The settings in ``environ`` (this process's environment by default), read as ``model``: a driver's ``Settings``,
    the base ones and those it adds."""
    raw = (os.environ if environ is None else environ).get(SETTINGS_ENV)
    if not raw:
        return model()
    try:
        return model.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as e:
        raise ValueError(f"{SETTINGS_ENV} is not valid computer settings: {e}") from None
