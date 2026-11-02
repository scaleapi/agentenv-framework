"""The computer SDK. A driver subclasses ComputerDriverEnvironment, marks the methods that capture and send input, and
serve()s it, which needs the ``computer`` extra: ``pip install 'agentenv-framework-protocol[computer]'``. An app
subclasses ComputerAppEnvironment, which needs only the plain package, so what a driver needs is imported on first
use."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .app import (
    APP_SETTINGS_ENV,
    AppDeclaration,
    AppOs,
    AptSource,
    ComputerAppEnvironment,
    Download,
    Launch,
    Linux,
    Packages,
    Ready,
    RunScript,
    Service,
    computer_app,
    install,
    install_step,
    linux,
    ready_check,
)
from .errors import (
    BackgroundUnavailable,
    ComputerError,
    DeliveryFailed,
    DeviceUnavailable,
    InputBusy,
    Internal,
    InvalidArgument,
    LimitExceeded,
    OutOfBounds,
    PermissionDenied,
    StaleFrame,
    TargetUnavailable,
    Timeout,
    Unsupported,
)
from .operations import (
    Display,
    Effect,
    ExecResult,
    Image,
    ReadyCheck,
    capture,
    click,
    display,
    drag,
    key_down,
    key_up,
    pointer_down,
    pointer_move,
    pointer_position,
    pointer_up,
    press,
    record,
    run_in_session,
    scroll,
    set_display,
    type_text,
    write_files,
)
from .dataplane import part_bytes
from .keys import PRINTABLE_KEYS
from .manifest import ManifestSegment, RecordingManifest
from .platform import Platform
from .settings import SETTINGS_ENV, ComputerSettings

if TYPE_CHECKING:
    from .card import (
        COMPUTER,
        COMPUTER_DISPLAY,
        COMPUTER_EXEC,
        COMPUTER_KEYBOARD,
        COMPUTER_POINTER,
        COMPUTER_READY,
        COMPUTER_RECORDING,
        COMPUTER_SPACES,
    )
    from .driver import DEFAULT_IDENTITY, ComputerDriverEnvironment, DriverDeclaration, computer_driver
    from .recording import RecordedFrame, RecordedSegment, RecordHints, Recording, frames_in_thread

# The names whose modules need the computer extra (mcp, av and Pillow), by module.
_ON_USE = {
    "driver": ("ComputerDriverEnvironment", "computer_driver", "DEFAULT_IDENTITY", "DriverDeclaration"),
    "card": ("COMPUTER", "COMPUTER_READY", "COMPUTER_DISPLAY", "COMPUTER_SPACES", "COMPUTER_POINTER",
             "COMPUTER_KEYBOARD", "COMPUTER_EXEC", "COMPUTER_RECORDING"),
    "recording": ("Recording", "RecordHints", "RecordedFrame", "RecordedSegment", "frames_in_thread"),
}


def __getattr__(name: str) -> Any:
    module = next((module for module, names in _ON_USE.items() if name in names), None)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


__all__ = [
    "ComputerDriverEnvironment", "computer_driver", "DEFAULT_IDENTITY", "DriverDeclaration",
    "ComputerAppEnvironment", "computer_app", "install", "linux", "install_step", "ready_check", "Linux", "Service",
    "Launch", "Ready", "Packages", "AptSource", "Download", "RunScript",
    "AppDeclaration", "AppOs",
    "APP_SETTINGS_ENV", "part_bytes",
    "Display", "Image", "Effect", "ExecResult", "ReadyCheck", "Platform", "ComputerSettings", "SETTINGS_ENV",
    "PRINTABLE_KEYS",
    "display", "set_display", "capture", "click", "drag", "scroll", "type_text", "press",
    "pointer_move", "pointer_down", "pointer_up", "pointer_position", "key_down", "key_up",
    "run_in_session", "write_files", "record", "Recording", "RecordHints", "RecordedFrame", "RecordedSegment",
    "frames_in_thread", "RecordingManifest", "ManifestSegment",
    "ComputerError", "InvalidArgument", "OutOfBounds", "StaleFrame", "InputBusy", "PermissionDenied", "Unsupported",
    "BackgroundUnavailable", "TargetUnavailable", "DeviceUnavailable", "LimitExceeded", "Timeout", "DeliveryFailed",
    "Internal",
    "COMPUTER", "COMPUTER_READY", "COMPUTER_DISPLAY", "COMPUTER_SPACES", "COMPUTER_POINTER", "COMPUTER_KEYBOARD",
    "COMPUTER_EXEC", "COMPUTER_RECORDING",
]
