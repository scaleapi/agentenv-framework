"""A computer's clock follower from core's side: its config, the gateway's watch URL as the computer reaches it, the
computer's name as the gateway knows it and its read token, written through the sandbox's root channel and never
through the driver, which runs as the session user, so whatever it could write the session user could too, and point
the follower at a clock of its own. Only root and the follower can read it; it's written beside its place and renamed
into it, so the follower, which reads it every quarter second, never reads half of one. Core then calls the computer's
own clock/v1 ``sync_time``, which answers once the follower shows the gateway's current version, so a deploy or a reset
returns with the computer on the universe's clock. One that can't follow is logged as a warning naming why, and the
deploy or reset goes on: a task that never arms the clock needn't care, and one that does is refused when it arms it,
as ``sync_env_clock`` calls the same ``sync_time``. The deploy record keeps the URL and name (``metadata.computer.clock``)
beside its control token, the read token for now, so a reset configures the fresh computer as the one it replaced."""
from __future__ import annotations

import json
import logging
import shlex
import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Optional

import httpx
from agentenv_protocol.client import ExtensionCallError
from agentenv_protocol.computer.clock import CONFIG, FOLLOWER_USER

from agent_env.env.envs.computer.utils.sites import RootChannel
from agent_env.env.gateway.constants import EXT_CLOCK_URI

if TYPE_CHECKING:
    from agent_env.env.env import DeployedEnv
    from agent_env.providers.sandbox_providers.sandbox_provider import Sandbox

logger = logging.getLogger(__name__)

WATCH_PATH = "/clock/watch"
TIME_PATH = "/clock/time"


def follows(record: DeployedEnv) -> bool:
    """Whether the computer ``record`` deployed follows the universe clock, as its card says."""
    return record.supports(EXT_CLOCK_URI, "sync_time")


def recorded(record: DeployedEnv) -> Optional[dict]:
    """The follower's config ``record`` keeps, its token aside: the watch URL and the computer's name."""
    clock = ((record.metadata or {}).get("computer") or {}).get("clock")
    return clock if isinstance(clock, dict) and clock.get("watch_url") and clock.get("name") else None


async def configure(sandbox: Sandbox, record: DeployedEnv, gateway_url: str, name: str) -> None:
    """Have the computer ``record`` deployed on ``sandbox`` follow the gateway at ``gateway_url`` (as this machine
    reaches it) as ``name``, with its control token, and keep the URL and name in ``record``."""
    watch_url = sandbox.url_from_sandbox(gateway_url.rstrip("/") + WATCH_PATH)
    await _write(sandbox, {"watch_url": watch_url, "name": name, "token": record.control_token})
    record.metadata.setdefault("computer", {})["clock"] = {"watch_url": watch_url, "name": name}
    await _following(record, watch_url)


async def reconfigure(sandbox: Sandbox, record: DeployedEnv) -> None:
    """The config ``record`` keeps, written again on the fresh computer a reset booted; nothing for one that has none."""
    if (kept := recorded(record)) is not None:
        await _write(sandbox, {**kept, "token": record.control_token})
        await _following(record, kept["watch_url"])


async def _following(record: DeployedEnv, watch_url: str) -> None:
    """Once the computer's follower shows the gateway's current version, as its own ``sync_time``, handed the gateway's
    time URL as the computer reaches it, says; a warning naming why when it doesn't."""
    try:
        await record.invoke(EXT_CLOCK_URI, "sync_time",
                            {"env_get_time_url": watch_url.removesuffix(WATCH_PATH) + TIME_PATH}, timeout=30)
    except ExtensionCallError as e:
        logger.warning(f"{record.env_id}: its clock follower doesn't follow the gateway at {watch_url}, so its "
                       f"session reads real time until it does: {str(e)[:600]}")
    except httpx.HTTPError as e:
        logger.warning(f"{record.env_id}: couldn't ask whether its clock follower follows the gateway at {watch_url}: "
                       f"{type(e).__name__}: {e}")


async def _write(sandbox: Sandbox, config: dict) -> None:
    channel = RootChannel(sandbox)
    staged = f"{CONFIG}.{uuid.uuid4().hex}"
    await channel.write_secret(PurePosixPath(staged), json.dumps(config))
    await channel.run(["sh", "-c", f"chown root:{shlex.quote(FOLLOWER_USER)} {shlex.quote(staged)} && "
                                   f"chmod 0640 {shlex.quote(staged)} && mv -f {shlex.quote(staged)} {shlex.quote(CONFIG)}"])
