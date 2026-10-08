"""Task step that applies a task's behavior config to deployed servers.

At setup (after the env is deployed + universe loaded, before the agent runs)
this step invokes each backing MCP server's ``set_*`` config extensions over the
gateway. Extensions are advertised on the server's ``EnvironmentCard`` (served at
``/.well-known/agent-env.json``) under ``capabilities.extensions[]`` and invoked
at their advertised REST endpoint (e.g. ``/agentenv/ext/set_errors``).

Config lives on the backing server, not on the gateway itself. Each server's card
comes from the env card stored on the deployed record, and its endpoints are invoked
through the gateway's ``/svc/mcp-{service}`` proxy; a record without a stored card
reads it live there. The agent never sees the card or these endpoints; only this
harness step invokes them.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

import httpx
from agentenv_protocol import client as protocol_v1

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class ApplyServerConfigError(RuntimeError):
    """Raised when a config directive cannot be applied (bad target / extension
    not advertised / server rejected the args)."""


class ConfigDirective:
    """A single config extension invocation against one backing server.

    - ``service``: backing MCP server name (the env's ``environment_name``); the
      gateway proxy key is ``mcp-<service>``.
    - ``uri``: the extension's stable card URI (e.g. ``urn:agentenv:set-errors/v1``).
      This is the discovery key on the card, NOT a fetchable URL.
    - ``args``: the extension's POST body (e.g. ``{"tool_name", "error_rate",
      "error_type"}`` for set_errors).
    """

    def __init__(self, service: str, uri: str, args: dict[str, Any]):
        if not isinstance(service, str) or not service:
            raise ValueError("directive.service must be a non-empty string")
        if not isinstance(uri, str) or not uri:
            raise ValueError("directive.uri must be a non-empty string")
        if not isinstance(args, dict):
            raise ValueError("directive.args must be a dict")
        self.service = service
        self.uri = uri
        self.args = args

    def to_dict(self) -> dict:
        # Dual-write. This feeds the `tasks` and `task_steps`
        # documents only. An exporter builds its own dict from these attributes,
        # so a frozen `{'service': ...}` export format is unaffected by the twin here.
        return {"service": self.service, "environment": self.service, "uri": self.uri, "args": self.args}

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigDirective":
        # Reads both spellings. Legacy-first: it is the spelling every
        # SDK version has always written, so it is never the stale half of a pair.
        service = data["service"] if "service" in data else data.get("environment")
        if service is None:
            raise ValueError("directive requires a 'service' (or 'environment') key")
        return cls(service=service, uri=data["uri"], args=data.get("args") or {})


class ApplyServerConfigStep(TaskStep):
    type: ClassVar[str] = "apply_server_config"
    entity_refs = (EntityRef.env("env_id"),)

    # A directive whose ``service`` is this sentinel is a broadcast: at execution
    # it expands to one directive per MCP server in the env (same uri + args).
    # Lets a caller apply an env-wide config (e.g. set_acting_user across every
    # server) without enumerating services itself. Pairs naturally with
    # ``tolerate_unadvertised`` so servers that don't opt in are simply skipped.
    _BROADCAST: ClassVar[str] = "*"

    def __init__(
        self,
        id: str,
        version: Optional[int],
        env_id: str,
        directives: list,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        timeout_seconds: int = 30,
        tolerate_unadvertised: bool = False,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("env_id must be a non-empty string")
        if not isinstance(directives, list) or not directives:
            raise ValueError("directives must be a non-empty list")
        self.env_id = env_id
        self.directives = [
            d if isinstance(d, ConfigDirective) else ConfigDirective.from_dict(d)
            for d in directives
        ]
        self.timeout_seconds = timeout_seconds
        # When True, a directive whose extension isn't advertised on the target
        # server's card is skipped (recorded) instead of failing the step. Use
        # for broadcast directives applied across every service in an env (e.g.
        # set_acting_user), where some services may not opt into the extension.
        # A failing live card read or an extension that IS advertised but rejects
        # the args (e.g. unresolvable persona) still fails loud regardless; a child
        # env missing from the stored card is skipped (no_env_card).
        self.tolerate_unadvertised = tolerate_unadvertised

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["env_id"] = self.env_id
        base["directives"] = [d.to_dict() for d in self.directives]
        base["timeout_seconds"] = self.timeout_seconds
        base["tolerate_unadvertised"] = self.tolerate_unadvertised
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "ApplyServerConfigStep":
        return cls(
            **cls._base_from_dict(data),
            env_id=data["env_id"],
            directives=data["directives"],
            timeout_seconds=data.get("timeout_seconds", 30),
            tolerate_unadvertised=data.get("tolerate_unadvertised", False),
        )

    def _expand_directives(self, deployed) -> list[ConfigDirective]:
        """Expand any broadcast directive (service == ``*``) into one directive
        per MCP server in the env, resolved from the env definition. The env's
        child service names aren't otherwise enumerable here (the deployed card
        list carries no roster), so we read them from ``Env.get`` — matching how
        deploy_env / load_artifact resolve the env. Non-broadcast directives
        pass through untouched."""
        if not any(d.service == self._BROADCAST for d in self.directives):
            return self.directives

        from agent_env.env.env import Env

        env = Env.get(deployed.env_id, deployed.env_version)
        environment_names = [
            e.environment_name for e in (getattr(env, "mcp_server_envs", None) or [])
        ]
        # A single-server env (a bare MCPServerEnv) exposes `environment_name` and no
        # `mcp_server_envs`; mirror snapshot_env / exporter enumeration and fall
        # back to it, else a broadcast silently matches nothing on single-server envs.
        if not environment_names and getattr(env, "environment_name", None):
            environment_names = [env.environment_name]
        # A broadcast that resolves to zero services would apply nothing while the
        # step reports success: the exact silent misconfig this step prevents. Fail
        # loud (tolerate_unadvertised governs advertisement, not missing servers).
        if not environment_names:
            raise ApplyServerConfigError(
                f"apply_server_config: broadcast directive matched no MCP servers "
                f"in env {self.env_id!r} (env exposes neither mcp_server_envs nor a "
                f"environment_name); nothing to apply, likely a task authoring error"
            )
        expanded: list[ConfigDirective] = []
        for d in self.directives:
            if d.service != self._BROADCAST:
                expanded.append(d)
                continue
            for svc in environment_names:
                expanded.append(ConfigDirective(service=svc, uri=d.uri, args=d.args))
        return expanded

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        deployed = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed is None:
            raise ApplyServerConfigError(f"Env '{self.env_id}' not found in context.deployed_envs")

        # Record each directive in context metadata the moment it's accepted, so a
        # later failure still leaves an accurate audit trail of what the server was
        # actually armed with — a partial apply must not look like a no-op.
        changes = context.metadata.setdefault("server_config_changes", [])
        # Directives skipped because the target server doesn't advertise the
        # extension (only when tolerate_unadvertised). Kept separate from
        # `changes` so the audit trail never shows a skip as an applied change.
        skipped = context.metadata.setdefault("server_config_skipped", [])
        # Expand broadcast (service="*") directives to one-per-env-service.
        # Env.get inside _expand_directives can raise (missing env / version /
        # network), so wrap it to keep the broadcast path's failure contract the
        # same ApplyServerConfigError as the per-directive HTTP calls below.
        try:
            directives = self._expand_directives(deployed)
        except ApplyServerConfigError:
            raise
        except Exception as e:
            raise ApplyServerConfigError(
                f"Failed to expand broadcast directives for env {self.env_id!r}: "
                f"{type(e).__name__}: {e}"
            ) from e
        from agent_env.env import legacy_protocol
        from agent_env.env.env import gateway_url_of

        # One card per service — a task typically sets several directives (e.g. errors on
        # multiple tools) against the same server. A None card marks a child env with no
        # card: a server predating env-card support, one whose card name differs, or one
        # whose card the deploy couldn't read.
        cards: dict[str, tuple[str, Optional[dict]]] = {}
        for directive in directives:
            try:
                if directive.service not in cards:
                    cards[directive.service] = await legacy_protocol.child_env_card(
                        deployed, gateway_url_of(deployed), directive.service, timeout=self.timeout_seconds
                    )
                base_url, card = cards[directive.service]
                if card is None:
                    if not self.tolerate_unadvertised:
                        raise ApplyServerConfigError(
                            f"Service {directive.service!r} has no env card (env={self.env_id}), so {directive.uri!r} "
                            f"can't be applied: it serves none, its card isn't named {directive.service!r}, or the deploy couldn't read it"
                        )
                    logger.warning(
                        f"apply_server_config: service {directive.service!r} serves no "
                        f"environment card (env={self.env_id}); skipping {directive.uri!r} "
                        f"(tolerate_unadvertised)"
                    )
                    skipped.append({
                        "step_id": self.id,
                        "env_id": self.env_id,
                        "service": directive.service,
                        "environment": directive.service,
                        "uri": directive.uri,
                        "reason": "no_env_card",
                    })
                    continue
                ext = protocol_v1.find_extension(card, directive.uri)
                if ext is None:
                    if self.tolerate_unadvertised:
                        logger.warning(
                            f"apply_server_config: {directive.uri!r} not advertised on "
                            f"service {directive.service!r} (env={self.env_id}); skipping "
                            f"(tolerate_unadvertised)"
                        )
                        skipped.append({
                            "step_id": self.id,
                            "env_id": self.env_id,
                            "service": directive.service,
                            "environment": directive.service,
                            "uri": directive.uri,
                            "reason": "extension_not_advertised",
                        })
                        continue
                    raise ApplyServerConfigError(
                        f"Extension {directive.uri!r} not advertised on card for service "
                        f"{directive.service!r} (env={self.env_id}). "
                        f"Advertised: {[e.get('uri') for e in (card.get('capabilities') or {}).get('extensions') or []]}"
                    )
                result = await protocol_v1.invoke_extension(
                    base_url, card, directive.uri, params=directive.args,
                    timeout=self.timeout_seconds,
                )
            except ApplyServerConfigError:
                raise
            except httpx.HTTPStatusError as e:
                raise ApplyServerConfigError(
                    f"Applying {directive.uri!r} on service {directive.service!r} "
                    f"(env={self.env_id}) failed: HTTP {e.response.status_code} {e.response.text}"
                ) from e
            except Exception as e:
                raise ApplyServerConfigError(
                    f"Applying {directive.uri!r} on service {directive.service!r} "
                    f"(env={self.env_id}) failed: {type(e).__name__}: {e}"
                ) from e

            logger.info(
                f"apply_server_config: applied {directive.uri} on service={directive.service} "
                f"env={self.env_id} args={directive.args}: {result}"
            )
            changes.append({
                "step_id": self.id,
                "env_id": self.env_id,
                "service": directive.service,
                "environment": directive.service,
                "uri": directive.uri,
                "args": directive.args,
                "result": result,
            })

        return context
