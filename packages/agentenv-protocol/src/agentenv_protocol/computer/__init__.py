"""The computer driver SDK: subclass ComputerDriverEnvironment, mark the methods that capture and send input, and
serve() it. Needs the ``computer`` extra: ``pip install 'agentenv-framework-protocol[computer]'``."""

from .driver import DEFAULT_IDENTITY, ComputerDriverEnvironment, computer_driver
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
from .card import (
    COMPUTER,
    COMPUTER_DISPLAY,
    COMPUTER_EXEC,
    COMPUTER_KEYBOARD,
    COMPUTER_POINTER,
    COMPUTER_READY,
    COMPUTER_SPACES,
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
    run_in_session,
    scroll,
    set_display,
    type_text,
    write_files,
)
from .platform import Platform
from .settings import SETTINGS_ENV, ComputerSettings

__all__ = [
    "ComputerDriverEnvironment", "computer_driver", "DEFAULT_IDENTITY",
    "Display", "Image", "Effect", "ExecResult", "ReadyCheck", "Platform", "ComputerSettings", "SETTINGS_ENV",
    "display", "set_display", "capture", "click", "drag", "scroll", "type_text", "press",
    "pointer_move", "pointer_down", "pointer_up", "pointer_position", "key_down", "key_up",
    "run_in_session", "write_files",
    "ComputerError", "InvalidArgument", "OutOfBounds", "StaleFrame", "InputBusy", "PermissionDenied", "Unsupported",
    "BackgroundUnavailable", "TargetUnavailable", "DeviceUnavailable", "LimitExceeded", "Timeout", "DeliveryFailed",
    "Internal",
    "COMPUTER", "COMPUTER_READY", "COMPUTER_DISPLAY", "COMPUTER_SPACES", "COMPUTER_POINTER", "COMPUTER_KEYBOARD",
    "COMPUTER_EXEC",
]
