"""The computer contract's error codes as exceptions a driver raises: ``raise OutOfBounds("...")``."""
from __future__ import annotations

from typing import ClassVar

from ..errors import AgentEnvError, Applied


class ComputerError(AgentEnvError):
    """One row of the errors table; a subclass fixes its code, status, whether it may be retried and whether the
    action may have happened."""

    CODE: ClassVar[str] = "internal"
    STATUS: ClassVar[int] = 500
    RETRYABLE: ClassVar[bool] = False
    APPLIED: ClassVar[Applied] = "unknown"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(self.CODE, message, retryable=self.RETRYABLE, applied=self.APPLIED, hint=hint, status=self.STATUS)


class InvalidArgument(ComputerError):
    CODE, STATUS, APPLIED = "invalid_argument", 400, "no"


class OutOfBounds(ComputerError):
    CODE, STATUS, APPLIED = "out_of_bounds", 400, "no"


class StaleFrame(ComputerError):
    CODE, STATUS, APPLIED, RETRYABLE = "stale_frame", 409, "no", True


class InputBusy(ComputerError):
    CODE, STATUS, APPLIED, RETRYABLE = "input_busy", 409, "no", True


class PermissionDenied(ComputerError):
    CODE, STATUS, APPLIED = "permission_denied", 403, "no"


class Unsupported(ComputerError):
    CODE, STATUS, APPLIED = "unsupported", 501, "no"


class BackgroundUnavailable(ComputerError):
    CODE, STATUS, APPLIED = "background_unavailable", 501, "no"


class TargetUnavailable(ComputerError):
    CODE, STATUS, APPLIED = "target_unavailable", 404, "no"


class DeviceUnavailable(ComputerError):
    CODE, STATUS, APPLIED, RETRYABLE = "device_unavailable", 503, "no", True


class LimitExceeded(ComputerError):
    CODE, STATUS, APPLIED, RETRYABLE = "limit_exceeded", 429, "no", True


class Timeout(ComputerError):
    CODE, STATUS = "timeout", 504


class DeliveryFailed(ComputerError):
    CODE, STATUS = "delivery_failed", 502


class Internal(ComputerError):
    pass
