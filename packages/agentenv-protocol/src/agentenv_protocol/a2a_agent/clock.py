"""An agent's side of clock/v1. An agent image that follows the env's clock runs the computer base's follower, and
agent-env's libfaketime is preloaded for the agent's user, so every program the A2A server starts reads universe time
from the follower's page; the server itself starts with the real-time marker and stays on real time.

The framework serves clock/v1's ``sync_time``, the method every server and computer that follows the clock serves, on
the card of an agent in such an image (``follows``), so the card is what tells core the agent follows. Unlike a
computer's, whose config core writes through the sandbox's root channel, an agent's request carries the follower's
config itself (``watch_url``, ``name``, ``read_token``), and this server, running as the agent's user, writes it: the
model's own commands could rewrite it and fool their own view of the time, though never the env's clock, and the token
only reads. It answers once the follower shows the gateway's current version, or says why it doesn't. The task's zone
is kept in the config beside what the follower reads, so a server restarted mid-run (``restore_zone``) brings the zone
back as the follower brings back the clock."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..computer import clock as follower
from ..computer.errors import InvalidArgument, TargetUnavailable


def follows() -> bool:
    """Whether this agent's image runs the clock follower."""
    return follower.follows()


def write_config(watch_url: str, name: str, read_token: str, timezone: str | None = None) -> int:
    """The follower's config, with the task's ``timezone`` for ``restore_zone``, written beside its place and renamed
    into it, so the follower, which reads it every quarter second, never reads half of one; readable by this user and
    the follower's group alone. When it was
    written (``st_mtime_ns``), which a status must postdate to be of this config; 0 when the config was this one
    already, which the follower goes on following without writing its status anew."""
    path = Path(follower.CONFIG)
    raw = json.dumps({"watch_url": watch_url, "name": name, "token": read_token,
                      **({"timezone": timezone} if timezone else {})}).encode()
    try:
        if path.read_bytes() == raw:
            return 0
    except OSError:
        pass
    staged = path.with_name(f"{path.name}.{uuid.uuid4().hex}")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.chmod(staged, 0o640)
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return path.stat().st_mtime_ns


def set_zone(timezone: str) -> str:
    """The task's zone for every program this server starts from now on: ``TZ`` in its environment, which a started
    program inherits; this server's own records keep UTC timestamps either way."""
    zone = timezone.strip()
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise InvalidArgument(f"timezone must be an IANA name such as 'America/Chicago', got {timezone!r}") from e
    os.environ["TZ"] = zone
    time.tzset()
    return zone


def restore_zone() -> str | None:
    """The task's zone ``sync_time`` saved, set again for what this server starts: called when it starts, so one that
    restarts mid-run keeps the zone, as the follower keeps the clock. None when there's none, or it isn't a zone."""
    try:
        zone = json.loads(Path(follower.CONFIG).read_text()).get("timezone")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(zone, str) or not zone:
        return None
    try:
        return set_zone(zone)
    except InvalidArgument:
        return None


async def sync_time(env_get_time_url: str, timezone: str | None = None, watch_url: str | None = None,
                    name: str | None = None, read_token: str | None = None) -> dict:
    """``clock/v1``'s ``sync_time`` on an agent: the follower configured to follow the gateway at ``watch_url`` as
    ``name`` with ``read_token``, the task's ``timezone`` set for what this server starts, and an answer once the
    follower shows the gateway's version for that config, never what it showed of one before. ``env_get_time_url`` is taken as every server takes it and not used: the
    follower reads the gateway's watch, not its time."""
    if not follows():
        raise TargetUnavailable("this agent's image doesn't run the clock follower")
    if not (watch_url and name and read_token):
        raise InvalidArgument("an agent follows the clock with its watch_url, name and read_token, which "
                              "sync_env_clock sends; this request names "
                              + (", ".join(k for k, v in (("watch_url", watch_url), ("name", name),
                                                          ("read_token", read_token)) if v) or "none of them"))
    zone = set_zone(timezone) if timezone is not None else os.environ.get("TZ") or "UTC"
    written = write_config(watch_url, name, read_token, zone)
    shown = await follower.following("this agent's clock follower", follower.CONNECT_WAIT_S, url=watch_url, name=name, since_ns=written)
    return {"synced": True, "timezone": zone, "env_get_time_url": env_get_time_url, "follower": shown}
