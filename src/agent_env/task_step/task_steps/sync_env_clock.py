"""Arm a deployed env gateway's virtual clock, then sync its backing servers to it.

Arms the clock (set_time), reads the gateway's server-reachable env_get_time_url from /clock/state,
and invokes the sync_time extension (clock/v1) on each backing server that advertises it, over the
{gateway}/svc/mcp-{service} proxy. Non-advertisers are skipped when tolerate_missing_sync_time
(default True). Place right before prompt_agent so the virtual timeline starts at agent-start.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol import client as protocol_v1

from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

_SVC_PROXY_PREFIX = "svc/mcp-"


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
    ):
        """virtual_seconds_per_real_second = clock speed (1.0 real time, 0 frozen, 86400 cap =
        1 real s -> 1 virtual day). For ~V virtual seconds over an ~R-real-second task, pass V/R."""
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(virtual_time, str):
            raise ValueError("virtual_time must be an RFC3339 string")
        self.env_id = env_id
        self.virtual_time = virtual_time
        self.virtual_seconds_per_real_second = virtual_seconds_per_real_second
        self.tolerate_missing_sync_time = tolerate_missing_sync_time
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["virtual_time"] = self.virtual_time
        base["virtual_seconds_per_real_second"] = self.virtual_seconds_per_real_second
        base["tolerate_missing_sync_time"] = self.tolerate_missing_sync_time
        base["timeout_seconds"] = self.timeout_seconds
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
        )

    def _service_names(self, deployed) -> list[str]:
        """Backing MCP server names for the env (mirrors apply_server_config's enumeration)."""
        from agent_env.env.env import Env
        env = Env.get(deployed.env_id, deployed.env_version)
        names = [e.environment_name for e in (getattr(env, "mcp_server_envs", None) or [])]
        if not names and getattr(env, "environment_name", None):
            names = [env.environment_name]
        return names

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")
        gateway_url = deployed.gateway_url.rstrip("/")

        async with httpx.AsyncClient() as client:
            resp = await client.put(f"{gateway_url}/clock/set-time",
                                    json={"virtual_time": self.virtual_time, "virtual_seconds_per_real_second": self.virtual_seconds_per_real_second},
                                    timeout=self.timeout_seconds)
            if resp.status_code >= 400:
                raise RuntimeError(f"clock arm failed (HTTP {resp.status_code}): {resp.text}")
            state = (await client.get(f"{gateway_url}/clock/state", timeout=self.timeout_seconds)).json()
        env_get_time_url = state.get("env_get_time_url")

        synced, skipped = [], []
        for service in self._service_names(deployed):
            base_url = f"{gateway_url}/{_SVC_PROXY_PREFIX}{service}"
            outcome = await self._sync_one(base_url, service, env_get_time_url)
            (synced if outcome.get("synced") else skipped).append({"service": service, "environment": service, **outcome})

        context.metadata.setdefault("clock_configurations", []).append({
            "step_id": self.id, "env_id": self.env_id,
            "virtual_time": self.virtual_time, "virtual_seconds_per_real_second": self.virtual_seconds_per_real_second,
            "env_get_time_url": env_get_time_url, "synced": synced, "skipped": skipped,
        })
        logger.info(f"sync_env_clock env={self.env_id}: armed rate={self.virtual_seconds_per_real_second}; "
                    f"synced={[s['service'] for s in synced]} skipped={[s['service'] for s in skipped]}")
        if not synced:
            logger.warning(f"sync_env_clock env={self.env_id}: clock armed but NO server synced "
                           f"({[(s['service'], s.get('reason')) for s in skipped]}) — every tool still answers wall time")
        return context

    async def _sync_one(self, base_url: str, service: str, env_get_time_url: Optional[str]) -> dict:
        """Invoke sync_time on one server if it advertises clock/v1; skip (or raise) per the flag."""
        try:
            card = await protocol_v1.get_card(base_url, timeout=self.timeout_seconds)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404 and self.tolerate_missing_sync_time:
                return {"synced": False, "reason": "no_env_card"}
            raise
        if protocol_v1.find_extension(card, EXT_CLOCK_URI) is None:
            if self.tolerate_missing_sync_time:
                logger.info(f"sync_env_clock: {service} does not advertise {EXT_CLOCK_URI}; skipping")
                return {"synced": False, "reason": "not_advertised"}
            raise RuntimeError(f"{service} does not advertise {EXT_CLOCK_URI}")
        if not env_get_time_url:
            raise RuntimeError(f"gateway did not expose env_get_time_url; cannot sync {service}")
        result = await protocol_v1.invoke_extension(base_url, card, EXT_CLOCK_URI,
                                                    params={"env_get_time_url": env_get_time_url},
                                                    timeout=self.timeout_seconds)
        return {"synced": True, "result": result}
