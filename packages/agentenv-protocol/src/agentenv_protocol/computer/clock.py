"""A computer's side of clock/v1. A base that follows the universe clock runs a follower (its image has ``FOLLOWER``),
which core configures through the sandbox's root channel (``CONFIG``, root's and the follower's alone, holding the
computer's read token) and which says how it's doing in ``STATUS``, holding no token. The driver serves clock/v1's
``sync_time``, the method every server that follows the clock serves, so the computer's card is what tells core and the
gateway that it follows: called when the clock is armed, it checks the task's zone against the one the computer booted
with and answers once the follower is connected, or says why it isn't."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from .errors import InvalidArgument, TargetUnavailable

CLOCK = "urn:agentenv:clock/v1"
FOLLOWER = "/usr/local/lib/agentenv-clock/clock-follower"
#: The follower's user, whose group may read ``CONFIG``.
FOLLOWER_USER = "agentenv-clock"
CONFIG = "/etc/agentenv/clock/follower.json"
STATUS = "/run/agentenv-clock/status.json"
#: What a disk form's guest has, made from the computer's container image: core's root channel reaches its launcher,
#: not the guest, so it can't configure a follower there, and a disk computer stays on real time for now.
DISK_GUEST = "/etc/agentenv/vm/image.env"
#: How long ``sync_time`` waits for the follower to be connected, as after a gateway restart it reconnects.
CONNECT_WAIT_S = 10.0


def follows() -> bool:
    """Whether this computer follows the universe clock: its base runs the follower, and it isn't a disk guest."""
    return os.path.exists(FOLLOWER) and not os.path.exists(DISK_GUEST)


def session_zone() -> str:
    """The zone the computer booted with, which every session program is in."""
    try:
        return os.environ.get("TZ") or Path("/etc/timezone").read_text().strip() or "UTC"
    except OSError:
        return "UTC"


def follower_status() -> dict:
    try:
        status = json.loads(Path(STATUS).read_text())
    except (OSError, ValueError):
        return {}
    return status if isinstance(status, dict) else {}


def _why(status: dict) -> str:
    if status.get("error"):
        return f"it can't reach the gateway at {status.get('url')}: {status['error']}"
    if not status.get("url"):
        return "core hasn't written its config, which it does when a multi env deploys this computer"
    return f"it hasn't heard from the gateway at {status['url']} yet"


async def sync_time(env_get_time_url: str, timezone: Optional[str] = None) -> dict:
    """``clock/v1``'s ``sync_time`` on a computer: its follower already follows the gateway core configured it with,
    so ``env_get_time_url`` is taken as every server takes it and not used; ``timezone``, the task's, must be the one
    the computer booted with, which a deploy sets (``computer_settings.timezone``)."""
    if not follows():
        raise TargetUnavailable("this computer's base doesn't follow the universe clock")
    zone = session_zone()
    if timezone is not None and timezone.strip() != zone:
        raise InvalidArgument(f"the task's clock is in {timezone.strip()!r} and this computer's session in {zone!r}; "
                              f"deploy it with computer_settings.timezone = {timezone.strip()!r}")
    deadline = time.monotonic() + CONNECT_WAIT_S
    while True:
        status = follower_status()
        if status.get("connected"):
            return {"synced": True, "timezone": zone, "env_get_time_url": env_get_time_url,
                    "follower": {key: status.get(key) for key in ("name", "url", "version", "armed")}}
        if time.monotonic() >= deadline:
            raise TargetUnavailable(f"this computer's clock follower isn't following the gateway: {_why(status)}")
        await asyncio.sleep(0.1)
