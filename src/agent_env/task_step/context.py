"""Context dataclass for task step execution."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from agent_env.env.env import DeployedEnv
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, host_url_for
from agent_env.providers.sandbox_providers.sandbox_provider import reachable_url

_REDACTED_KEYS = {
    "litellm_api_key", "judge_litellm_api_key", "usersim_api_key", "remote_tokens", "cf_access_client_secret",
}


def _holds_redacted_key(value: Any) -> bool:
    """Whether ``value`` carries a ``_REDACTED_KEYS`` entry at any depth."""
    return isinstance(value, dict) and any(
        k in _REDACTED_KEYS or _holds_redacted_key(v) for k, v in value.items()
    )


def regraft_redacted_keys(source: dict, target: dict) -> None:
    """Copy every ``_REDACTED_KEYS`` value in ``source`` into ``target`` in place; the inverse
    of ``_strip_redacted_keys`` for a context rebuilt from a stored document."""
    for k, v in source.items():
        if k in _REDACTED_KEYS:
            target[k] = v
        elif isinstance(v, dict):
            child = target.get(k)
            if not isinstance(child, dict):
                # A secrets-only parent leaves nothing to rebuild from, so recreate it —
                # but never over a value the rebuild did write.
                if target.get(k) is not None or not _holds_redacted_key(v):
                    continue
                child = target[k] = {}
            regraft_redacted_keys(v, child)


def _strip_redacted_keys(value: Any) -> Any:
    """Recursively strip `_REDACTED_KEYS` from any nested dicts in `value`.

    Single source of truth used by both `to_safe_dict` (full-blob writes via
    `record_task_failure`) and the path-level diff in `context_ops` so secrets
    never reach Mongo regardless of which code path persists them.
    """
    if isinstance(value, dict):
        return {k: _strip_redacted_keys(v) for k, v in value.items() if k not in _REDACTED_KEYS}
    if isinstance(value, list):
        return [_strip_redacted_keys(v) for v in value]
    return value


@dataclass
class DeployedAgent:
    agent_name: str
    api_url: str
    sandbox_id: str | None = None
    sandbox_type: str | None = None
    a2a_url: str | None = None
    a2a_card: dict | None = None
    instance_id: str | None = None
    role: str | None = None
    network_policy: dict | None = None
    # Installed straight onto the sandbox host (install_agent host mode), not into a container.
    on_host: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> DeployedAgent:
        return cls(
            agent_name=data["agent_name"],
            api_url=data["api_url"],
            sandbox_id=data.get("sandbox_id"),
            sandbox_type=data.get("sandbox_type"),
            a2a_url=data.get("a2a_url"),
            a2a_card=data.get("a2a_card"),
            instance_id=data.get("instance_id"),
            role=data.get("role"),
            network_policy=data.get("network_policy"),
            on_host=bool(data.get("on_host", False)),
        )

    def url_for(self, sandbox_type: str | None, *, on_host: bool = False) -> str:
        """This agent's A2A URL as an agent on ``sandbox_type`` reaches it, in a container there or, ``on_host``, on
        the sandbox's host. A local sandbox's host is this machine, which reaches the URL as it is. An agent outside
        our sandboxes, a human say, is at a URL this machine reaches."""
        url = self.a2a_url or self.api_url
        if on_host and sandbox_type == LocalSandbox.type:
            return url
        if not self.sandbox_id:
            return host_url_for(url, sandbox_type)
        return reachable_url(url, from_sandbox_type=self.sandbox_type, to_sandbox_type=sandbox_type)


@dataclass
class DeployedSandbox:
    sandbox_name: str
    sandbox_id: str
    sandbox_mode: str
    sandbox_type: str | None = None
    tunnel_urls: dict[str, str] | None = None
    vnc_url: str | None = None
    instance_id: str | None = None
    created_at_utc: str | None = None
    expires_at_utc: str | None = None
    network_policy: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> DeployedSandbox:
        return cls(
            sandbox_name=data["sandbox_name"],
            sandbox_id=data["sandbox_id"],
            sandbox_mode=data["sandbox_mode"],
            sandbox_type=data.get("sandbox_type"),
            tunnel_urls=data.get("tunnel_urls"),
            vnc_url=data.get("vnc_url"),
            instance_id=data.get("instance_id"),
            created_at_utc=data.get("created_at_utc"),
            expires_at_utc=data.get("expires_at_utc"),
            network_policy=data.get("network_policy"),
        )


def dual_keyed(legacy: str, neutral: str, value: Any) -> dict[str, Any]:
    """A renamed run-context key under both its names: readers on older versions read the legacy one."""
    return {legacy: value, neutral: value}


def read_dual_keyed(data: dict[str, Any], legacy: str, neutral: str) -> Any:
    """A renamed run-context key, legacy name first while both are written: a raw-doc writer that knows only the
    legacy name leaves the neutral one stale. A legacy key holding None falls through to the neutral one."""
    value = data.get(legacy)
    return value if value is not None else data.get(neutral)


# (legacy, neutral) keys of the PromptResponse fields that were renamed. Stored documents and raw-doc readers use the
# legacy keys, so to_dict writes each beside its neutral one and from_dict reads either.
_PROMPT_RESPONSE_LEGACY_KEYS = (
    ("agent_trajectory_s3_uri", "agent_trajectory_object_url"),
    ("agent_trajectory_s3_prefix", "agent_trajectory_object_prefix"),
    ("target_agent_per_turn_trajectory_s3_uris", "target_agent_per_turn_trajectory_object_urls"),
)


@dataclass
class PromptResponse:
    prompt_id: str
    response: str
    prompt_text: str | None = None
    agent_trajectory_file_path: str | None = None
    # A None entry means "identical to prompt_text" — the first turn of a
    # prompt-mode step is not stored twice.
    source_agent_per_turn_prompt_parts: list[list[dict] | None] | None = None
    tool_call_count: int | None = None
    model: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    agent_session_id: str = ""
    a2a_context_id: str | None = None
    agent_name: str | None = None
    step_id: str | None = None
    structured_output: Any = None
    agent_trajectory_object_url: str | None = None
    agent_trajectory_object_prefix: str | None = None
    target_agent_per_turn_trajectory_object_urls: list[str | None] | None = None

    def to_dict(self) -> dict[str, Any]:
        """The stored form: every field, plus the legacy key of each renamed one."""
        data = dataclasses.asdict(self)
        for legacy, neutral in _PROMPT_RESPONSE_LEGACY_KEYS:
            value = data[neutral]
            data[legacy] = list(value) if isinstance(value, list) else value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> PromptResponse:
        return cls(
            prompt_id=data["prompt_id"],
            response=data["response"],
            prompt_text=data.get("prompt_text"),
            agent_trajectory_object_url=read_dual_keyed(data, "agent_trajectory_s3_uri", "agent_trajectory_object_url"),
            agent_trajectory_object_prefix=read_dual_keyed(
                data, "agent_trajectory_s3_prefix", "agent_trajectory_object_prefix"
            ),
            agent_trajectory_file_path=data.get("agent_trajectory_file_path"),
            target_agent_per_turn_trajectory_object_urls=read_dual_keyed(
                data, "target_agent_per_turn_trajectory_s3_uris", "target_agent_per_turn_trajectory_object_urls"
            ),
            source_agent_per_turn_prompt_parts=data.get("source_agent_per_turn_prompt_parts"),
            tool_call_count=data.get("tool_call_count"),
            model=data.get("model"),
            error_type=data.get("error_type"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            agent_session_id=data.get("agent_session_id", ""),
            a2a_context_id=data.get("a2a_context_id"),
            agent_name=data.get("agent_name"),
            step_id=data.get("step_id"),
            structured_output=data.get("structured_output"),
        )


@dataclass
class TaskStepContext:
    deployed_envs: list[DeployedEnv] = field(default_factory=list)
    deployed_agents: list[DeployedAgent] = field(default_factory=list)
    deployed_sandboxes: list[DeployedSandbox] = field(default_factory=list)
    prompt_responses: list[PromptResponse] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_model: str | None = None
    default_agent_model: str | None = None
    agent_artifact_id: str | None = None
    agent_harness: str | None = None
    instance_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The stored form: ``asdict``, with each prompt response's legacy keys (``PromptResponse.to_dict``)."""
        d = dataclasses.asdict(dataclasses.replace(self, prompt_responses=[]))
        d["prompt_responses"] = [response.to_dict() for response in self.prompt_responses]
        return d

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a dict representation with sensitive keys recursively removed."""
        d = self.to_dict()
        if "metadata" in d:
            d["metadata"] = _strip_redacted_keys(d["metadata"])
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskStepContext:
        return cls(
            deployed_envs=[DeployedEnv.from_dict(e) for e in data.get("deployed_envs", [])],
            deployed_agents=[DeployedAgent.from_dict(a) for a in data.get("deployed_agents", [])],
            deployed_sandboxes=[DeployedSandbox.from_dict(s) for s in data.get("deployed_sandboxes", [])],
            prompt_responses=[PromptResponse.from_dict(p) for p in data.get("prompt_responses", [])],
            metadata=data.get("metadata", {}),
            agent_model=data.get("agent_model"),
            default_agent_model=data.get("default_agent_model"),
            agent_artifact_id=data.get("agent_artifact_id"),
            agent_harness=data.get("agent_harness"),
            instance_id=data.get("instance_id"),
        )
