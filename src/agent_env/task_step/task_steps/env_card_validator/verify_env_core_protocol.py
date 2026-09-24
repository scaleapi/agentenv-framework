"""Verify core environment-protocol operation support (env mirror of VerifyCoreA2AProtocolStep)."""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import httpx
from agentenv_protocol import METHOD_ADD, METHOD_GET, METHOD_RESET, RPC_PATH

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

# (method, probe_params); None = no mutation-free probe — resolve from the advertised operations.
_CORE_OPERATIONS = [
    (METHOD_RESET, None),
    (METHOD_ADD, {"parts": []}),
    (METHOD_GET, {}),
]

VALIDATED_ENVIRONMENT_PROTOCOL_KEY = "validated_environment_protocol"


class VerifyCoreEnvironmentProtocolStep(TaskStep):
    type: ClassVar[str] = "verify_env_core_protocol"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id = env_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyCoreEnvironmentProtocolStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"])

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agentenv_protocol import client as protocol_v1
        from agent_env.env.env import Env

        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise RuntimeError(f"Env '{self.env_id}' not found in context.deployed_envs")

        card = None
        try:
            card = await protocol_v1.get_card(deployed.gateway_url)
        except Exception as e:
            logger.warning(f"env card for '{self.env_id}' inaccessible at {deployed.gateway_url}: {type(e).__name__}: {e}")

        protocol = {}
        for method, probe_params in _CORE_OPERATIONS:
            if probe_params is not None:
                supported = await self._probe_operation(deployed.gateway_url, method, probe_params)
            else:
                supported = self._advertised_support(card, method)
            protocol[method] = {"supported": supported}

        logger.info(f"core environment protocol for '{self.env_id}': " + " ".join(f"{m}={v['supported']}" for m, v in protocol.items()))

        env = Env.get(self.env_id, deployed.env_version)
        env.update_metadata({**env.metadata, VALIDATED_ENVIRONMENT_PROTOCOL_KEY: protocol})
        context.metadata.setdefault("verifications", {})["env_core_protocol"] = protocol
        return context

    @staticmethod
    def _advertised_support(card: Optional[dict], method: str) -> Optional[bool]:
        """Per the backing card's advertised operations; None if unresolvable. An absent
        key predates advertisement, when construction required the full data trio."""
        if not card:
            return None
        children = card.get("children_environments") or []
        backing = children[0] if len(children) == 1 else (card if not children else None)
        if backing is None:
            return None
        advertised = (backing.get("capabilities") or {}).get("operations")
        return True if advertised is None else method in advertised

    async def _probe_operation(self, gateway_url: str, method: str, params: dict) -> Optional[bool]:
        """Mutation-free: -32601 = unregistered; -32000 = gateway couldn't forward (None);
        anything else — a result, or -32602 from the empty-parts add probe — proves registration."""
        body = {"jsonrpc": "2.0", "id": f"verify-{method}", "method": method, "params": params}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(f"{gateway_url.rstrip('/')}{RPC_PATH}", json=body)
            error = (response.json() or {}).get("error") or {}
        except Exception as e:
            logger.warning(f"operation probe {method} for '{self.env_id}' failed: {e}")
            return None
        if error.get("code") == -32601:
            return False
        if error.get("code") == -32000:
            return None
        return True
