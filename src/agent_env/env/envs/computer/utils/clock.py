"""A multi env's computer on its gateway's clock: the clock follower configured and started as root, through the
sandbox's root channel, never through the driver, which runs as the session user.

A computer follows when it composes the ``clock`` app (``agent_env/computers/envs/clock``): agent-env's libfaketime in
every process, reading the clock page, and the follower that writes it as ``agentenv-clock``. The app's card advertises
``clock/v1`` ``sync_time``, which makes the gateway count the computer as a follower (``follows_clock``) and gives it a
read token for its watch. ``sync_env_clock`` then calls ``start_follower`` for each such computer before it sets the
clock, and sets it waiting for their acknowledgements."""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from agentenv_protocol import client as protocol_v1

from agent_env.env.envs.computer.utils.sites import RootChannel
from agent_env.env.gateway.constants import EXT_CLOCK_URI

if TYPE_CHECKING:
    from agent_env.providers.sandbox_providers.sandbox import Sandbox

#: The follower's user, which owns the clock page and the clock file, and alone reads the follower's config.
FOLLOWER_USER = "agentenv-clock"
CONFIG_FOLDER = PurePosixPath("/etc/agentenv/clock-follower")
CONFIG = CONFIG_FOLDER / "follower.json"
#: The clock file's folder, where the follower also keeps its status.
RUN_FOLDER = PurePosixPath("/run/agentenv-clock")
#: What the follower last showed, which anyone may read.
STATUS = RUN_FOLDER / "status.json"
CLOCK = PurePosixPath("/opt/agentenv/clock")
LOG = PurePosixPath("/var/log/agentenv/clock-follower.log")
#: The ``sync_time`` param that carries the id of the config core wrote, which the clock app waits for its follower
#: to show.
CONFIG_PARAM = "follower_config"
#: How long a write of the clock, or a computer's follower started again, waits for the computer to show it.
FOLLOWER_WAIT_S = 10.0


def follows_clock(card: dict | None) -> bool:
    """Whether a computer with ``card`` follows the gateway's clock: it, or one of its apps, offers ``clock/v1``'s
    ``sync_time``."""
    if not isinstance(card, dict):
        return False
    return any(protocol_v1.find_extension_method(each, EXT_CLOCK_URI, "sync_time") is not None
               for each in [card, *(child.card for child in protocol_v1.children(card) if child.card)])


def takes_config(card: dict | None) -> bool:
    """Whether a server with ``card`` is a clock app's, whose ``sync_time`` takes the follower's config id."""
    method = protocol_v1.find_extension_method(card or {}, EXT_CLOCK_URI, "sync_time") or {}
    return CONFIG_PARAM in ((method.get("request") or {}).get("properties") or {})


def runs_follower(card: dict | None) -> bool:
    """Whether a computer with ``card`` runs the clock follower: one of its apps is the clock app (``takes_config``)."""
    return isinstance(card, dict) and any(takes_config(child.card) for child in protocol_v1.children(card))


async def start_follower(sandbox: Sandbox, *, watch_url: str, token: str, follower: str) -> str:
    """Configure the computer's clock follower to watch ``watch_url`` as ``follower`` with ``token``, its read token,
    and (re)start it, all as root; the id of the config written, which the clock app's ``sync_time`` waits for.

    The config folder is the follower's alone (0700), the config its alone to read (0600), so the session user never
    sees the token. The clock page is made as root, the follower's from its first byte, replacing any the session user
    made first; the library trusts no page another user owns (``clock.conf``'s ``AGENTENV_CLOCK_PAGE_UID``)."""
    config_id = secrets.token_hex(8)
    channel = RootChannel(sandbox)
    owner = f"{FOLLOWER_USER}:{FOLLOWER_USER}"
    await channel.run(["install", "-d", "-m", "0700", "-o", FOLLOWER_USER, "-g", FOLLOWER_USER, str(CONFIG_FOLDER)])
    await channel.write_secret(CONFIG, json.dumps({"watch_url": watch_url, "token": token, "follower": follower,
                                                    "config_id": config_id}))
    await channel.run(["chown", owner, str(CONFIG)])
    await channel.run(["chmod", "0600", str(CONFIG)])
    await channel.run(["install", "-d", "-m", "0755", "-o", FOLLOWER_USER, "-g", FOLLOWER_USER, str(RUN_FOLDER)])
    await channel.run(["python3", str(CLOCK / "clockpage.py"), "create", FOLLOWER_USER])
    # Detached from the exec, which returns at once; ``follow`` replaces a follower an earlier arm started.
    await channel.run(["sh", "-c", f"setsid {CLOCK / 'follow'} </dev/null >>{LOG} 2>&1 &"])
    return config_id


async def wait_shown(sandbox: Sandbox, config_id: str, timeout_s: float = FOLLOWER_WAIT_S) -> dict:
    """What the computer's follower showed once it has shown a version of the clock for ``config_id``, read as root;
    TimeoutError naming its last status when it hasn't within ``timeout_s``."""
    channel, deadline, status = RootChannel(sandbox), time.monotonic() + timeout_s, None
    while True:
        try:
            status = json.loads(await channel.run(["sh", "-c", f"cat {STATUS} 2>/dev/null || true"]))
        except json.JSONDecodeError:
            status = None
        if status and status.get("config_id") == config_id and status.get("version") and not status.get("error"):
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(f"the clock follower didn't show config {config_id!r} within {timeout_s:g} s; its "
                               f"status: {status}")
        await asyncio.sleep(0.2)
