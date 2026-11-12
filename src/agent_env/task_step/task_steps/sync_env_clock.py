"""Arm a deployed env gateway's virtual clock, then sync its backing servers to it.

Arms the clock (set_time) and reads the gateway's server-reachable env_get_time_url (state), both as the
stored env card declares clock/v1, then invokes sync_time on each backing server, MCP server or website, the
env card lists (``legacy_protocol.backing_servers``) whose own card advertises it, at the route that card gives.
Non-advertisers are skipped when tolerate_missing_sync_time (default True), and websites always. Place right
before prompt_agent so the virtual timeline starts at agent-start.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone as _utc_tz
from typing import TYPE_CHECKING, ClassVar, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import tzdata  # noqa: F401
from agentenv_protocol import client as protocol_v1

from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import StepContextError, TaskStepContext
from agent_env.entity.refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

if TYPE_CHECKING:
    from agent_env.env.legacy_protocol import BackingServer

logger = logging.getLogger(__name__)


def resolve_virtual_time(virtual_time: str, timezone: Optional[str]) -> str:
    """The instant to arm the gateway at. Without a zone, `virtual_time` unchanged; with one, a naive wall
    time is read in that zone and an explicit offset must agree with it. Returned as a UTC RFC3339 string."""
    if timezone is None:
        return virtual_time
    try:
        zone = ZoneInfo(timezone.strip()) if isinstance(timezone, str) and timezone.strip() else None
    except (ZoneInfoNotFoundError, ValueError, OSError):
        zone = None
    if zone is None:
        raise ValueError(f"timezone must be a valid IANA name such as 'America/Chicago', got {timezone!r}")
    try:
        parsed = datetime.fromisoformat(virtual_time)
    except ValueError as e:
        raise ValueError(f"virtual_time {virtual_time!r} is not an ISO 8601 / RFC3339 timestamp") from e
    if parsed.tzinfo is None:
        local = parsed.replace(tzinfo=zone)
        if local.astimezone(_utc_tz.utc).astimezone(zone).replace(tzinfo=None) != parsed:
            raise ValueError(f"virtual_time {virtual_time!r} does not exist in {zone.key} (clock skips it)")
        return local.astimezone(_utc_tz.utc).isoformat().replace("+00:00", "Z")
    if parsed.astimezone(zone).utcoffset() != parsed.utcoffset():
        raise ValueError(f"virtual_time {virtual_time!r} carries offset {parsed.utcoffset()} but {zone.key} is at "
                         f"{parsed.astimezone(zone).utcoffset()} at that instant; drop the offset or fix the timezone")
    return parsed.astimezone(_utc_tz.utc).isoformat().replace("+00:00", "Z")


class SyncEnvClockTaskStep(TaskStep):
    type: ClassVar[str] = "sync_env_clock"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        virtual_time: str,
        virtual_seconds_per_real_second: float = 1.0,
        tolerate_missing_sync_time: bool = True,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        timeout_seconds: int = 30,
        timezone: Optional[str] = None,
    ):
        """virtual_seconds_per_real_second = clock speed (1.0 real time, 0 frozen, 86400 cap =
        1 real s -> 1 virtual day). For ~V virtual seconds over an ~R-real-second task, pass V/R.

        timezone = optional IANA name. Unset keeps today's behaviour. Set, a wall-clock virtual_time is local time in
        that zone, and every synced server is told the zone (a server that cannot take it fails the step)."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(virtual_time, str):
            raise ValueError("virtual_time must be an RFC3339 string")
        self.env_id = env_id
        self.virtual_time = virtual_time
        self.virtual_seconds_per_real_second = virtual_seconds_per_real_second
        self.tolerate_missing_sync_time = tolerate_missing_sync_time
        self.timeout_seconds = timeout_seconds
        self.timezone = timezone.strip() if isinstance(timezone, str) else timezone
        self.armed_virtual_time = resolve_virtual_time(virtual_time, self.timezone)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["virtual_time"] = self.virtual_time
        base["virtual_seconds_per_real_second"] = self.virtual_seconds_per_real_second
        base["tolerate_missing_sync_time"] = self.tolerate_missing_sync_time
        base["timeout_seconds"] = self.timeout_seconds
        if self.timezone is not None:
            base["timezone"] = self.timezone
        return base

    @classmethod
    def from_dict(cls, data: dict) -> SyncEnvClockTaskStep:
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            virtual_time=data["virtual_time"],
            virtual_seconds_per_real_second=data.get("virtual_seconds_per_real_second", 1.0),
            tolerate_missing_sync_time=data.get("tolerate_missing_sync_time", True),
            timeout_seconds=data.get("timeout_seconds", 30),
            timezone=data.get("timezone"),
        )

    async def _backing_servers(self, deployed) -> list[BackingServer]:
        """The env's backing MCP servers and websites, as its card lists them (``legacy_protocol.backing_servers``):
        a website sharing its name with an MCP server is a server of its own, at its own route."""
        from agent_env.env import legacy_protocol
        from agent_env.env.env import Env, gateway_url_of
        return await legacy_protocol.backing_servers(deployed, gateway_url_of(deployed), Env.get(deployed.env_id, deployed.env_version),
                                                     timeout=self.timeout_seconds)

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.env.env import gateway_url_of

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise StepContextError(f"Env '{self.env_id}' not found in context.deployed_envs")
        try:
            await deployed.invoke(EXT_CLOCK_URI, "set_time",
                                  {"virtual_time": self.armed_virtual_time, "virtual_seconds_per_real_second": self.virtual_seconds_per_real_second},
                                  timeout=self.timeout_seconds)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"clock arm failed (HTTP {e.response.status_code}): {e.response.text}") from e
        state = await deployed.invoke(EXT_CLOCK_URI, "state", timeout=self.timeout_seconds)
        env_get_time_url = state.get("env_get_time_url")

        synced, skipped = [], []
        for server in await self._backing_servers(deployed):
            service = server.environment_name
            outcome = await self._sync_one(server.owner(deployed), gateway_url_of(deployed), service, env_get_time_url,
                                           mcp=server.mcp, tolerate=self.tolerate_missing_sync_time or not server.mcp,
                                           server=server)
            entry = {"service": service, "environment": service, **({} if server.mcp else {"website": True}), **outcome}
            (synced if outcome.get("synced") else skipped).append(entry)

        configuration = {
            "step_id": self.id, "env_id": self.env_id,
            "virtual_time": self.virtual_time, "virtual_seconds_per_real_second": self.virtual_seconds_per_real_second,
            "env_get_time_url": env_get_time_url, "synced": synced, "skipped": skipped,
        }
        if self.timezone is not None:
            configuration.update(timezone=self.timezone, armed_at=self.armed_virtual_time)
        context.metadata.setdefault("clock_configurations", []).append(configuration)
        logger.info(f"sync_env_clock env={self.env_id}: armed rate={self.virtual_seconds_per_real_second}; "
                    f"synced={[s['service'] for s in synced]} skipped={[s['service'] for s in skipped]}")
        if not synced:
            logger.warning(f"sync_env_clock env={self.env_id}: clock armed but NO server synced "
                           f"({[(s['service'], s.get('reason')) for s in skipped]}) — every tool still answers wall time")
        return context

    async def _sync_one(self, deployed, gateway_url: str, service: str, env_get_time_url: Optional[str], mcp: bool = True,
                        tolerate: Optional[bool] = None, server: Optional[BackingServer] = None) -> dict:
        """Invoke sync_time on one server, an MCP server or with ``mcp=False`` a website, if its card advertises
        clock/v1; skip a server with no card or no clock/v1 when ``tolerate`` (tolerate_missing_sync_time by default),
        else raise. ``execute`` tolerates every website, so one that doesn't follow the clock never fails the step; one
        that follows it but can't take the task's timezone does, as an MCP server does. ``server`` is the server as the
        env's card lists it (``legacy_protocol.server_card``); without it, the card is ``service``'s on ``deployed``'s
        stored card, or read live at its gateway path."""
        tolerate = self.tolerate_missing_sync_time if tolerate is None else tolerate
        from agent_env.env import legacy_protocol

        if server is not None:
            base_url, card = await legacy_protocol.server_card(server, gateway_url, timeout=self.timeout_seconds)
        else:
            base_url, card = await legacy_protocol.child_env_card(deployed, gateway_url, service, timeout=self.timeout_seconds, mcp=mcp)
        if card is None:
            if tolerate:
                return {"synced": False, "reason": "no_env_card"}
            raise RuntimeError(f"{service} has no env card: it serves none, its card isn't named {service!r}, or the deploy couldn't read it")
        if protocol_v1.find_extension(card, EXT_CLOCK_URI) is None:
            if tolerate:
                logger.info(f"sync_env_clock: {service} does not advertise {EXT_CLOCK_URI}; skipping")
                return {"synced": False, "reason": "not_advertised"}
            raise RuntimeError(f"{service} does not advertise {EXT_CLOCK_URI}")
        if not env_get_time_url:
            raise RuntimeError(f"gateway did not expose env_get_time_url; cannot sync {service}")
        params = {"env_get_time_url": env_get_time_url}
        if self.timezone is not None:
            method = protocol_v1.find_extension_method(card, EXT_CLOCK_URI, "sync_time") or {}
            if "timezone" not in ((method.get("request") or {}).get("properties") or {}):
                raise RuntimeError(f"{service} advertises {EXT_CLOCK_URI} but its sync_time takes no timezone; "
                                   f"rebuild its image or drop timezone={self.timezone!r} from the task")
            params["timezone"] = self.timezone
        result = await protocol_v1.invoke_extension(base_url, card, EXT_CLOCK_URI,
                                                    params=params,
                                                    timeout=self.timeout_seconds,
                                                    headers=deployed.control_headers)
        return {"synced": True, "result": result}
