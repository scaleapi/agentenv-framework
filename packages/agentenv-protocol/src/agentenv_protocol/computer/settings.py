"""The settings a computer driver reads at start, from one JSON value in ``AGENTENV_COMPUTER_SETTINGS``."""
from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..transfers import RelativePath

SETTINGS_ENV = "AGENTENV_COMPUTER_SETTINGS"

Space = Literal["frame", "norm"]


class ComputerSettings(BaseModel):
    """``data_paths`` are relative to the session user's home and are what ``data/get`` returns; the rest are the
    defaults the card reports and the framework extensions change."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    data_paths: list[RelativePath] = Field(default_factory=list)
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


def load_settings(environ: dict | None = None) -> ComputerSettings:
    raw = (os.environ if environ is None else environ).get(SETTINGS_ENV)
    if not raw:
        return ComputerSettings()
    try:
        return ComputerSettings.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as e:
        raise ValueError(f"{SETTINGS_ENV} is not valid computer settings: {e}") from None
