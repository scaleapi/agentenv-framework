"""Prompt a deployed agent task step."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional
import httpx
from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent, conversation_store
from agent_env.a2a_agent import protocol
from agent_env.a2a_agent.object_transfer import (
    TrajectoryUpload,
    fetch_trajectory,
    send_and_wait,
    trajectory_mode,
)
from agent_env.a2a_agent.staging import draining, staged_changelogs, transfer_store
from agent_env.providers.sandbox_providers.local_sandbox import transfer_sandbox_type
from agent_env.config.model import MODEL_PARAMS_RESERVED
from agent_env.env.gateway.constants import EXT_CLOCK_URI, EXT_TRIGGERS_URI, TRIGGER_IN_FLIGHT_STATUSES
from agent_env.store import DuplicateKeyError, get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import RetryConfig, TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.sandbox_utils.sandbox_utils import (
    agent_error_text,
    fetch_container_logs,
)
from agent_env.task_step.snapshot_utils.agent_state_capture import (
    store_trajectory,
    trajectory_object_url,
)
from agent_env.task_step.snapshot_utils.snapshot_series import SnapshotConfig, SnapshotSeries

logger = logging.getLogger(__name__)


def _parse_structured_output(text: str) -> Optional[dict]:
    """Best-effort parse of the response text as a StructuredOutput JSON object (last embedded
    object, or None for free text). Fallback for harnesses that don't surface the typed
    ``structured_output`` on the A2A DataPart."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):  # tolerate a ```json … ``` fence
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Fast path: the whole response is the JSON object.
    if s.startswith("{"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    # Embedded in prose: scan top-level {...} objects and return the LAST — so an earlier
    # JSON-looking snippet can't win over the real output (which agents emit last).
    decoder = json.JSONDecoder()
    last: Optional[dict] = None
    i = s.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(s, i)
        except ValueError:
            i = s.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            last = obj
        i = s.find("{", end)
    return last


def _duplicates_prompt_text(parts: list[dict], prompt_text: Optional[str]) -> bool:
    """True when ``parts`` is exactly the single text part already persisted as
    ``PromptResponse.prompt_text`` — i.e. the first conversation turn of a
    prompt-mode step. That text can be huge (a prompt can inline base64 images), so
    storing it a second time in ``source_agent_per_turn_prompt_parts`` roughly
    doubles the task-run context doc. Readers treat a ``None`` per-turn entry as "same as
    ``prompt_text``"."""
    if prompt_text is None:
        return False
    return parts == [{"kind": "text", "text": prompt_text}]


def _file_uris(parts: list[dict]) -> list[str]:
    files = (part.get("file") for part in parts if part.get("kind") == "file")
    return [file["uri"] for file in files if isinstance(file, dict) and isinstance(file.get("uri"), str)]


_DEFAULT_USER_SIM_OUTPUT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Your next reply"},
            "done": {"type": "boolean", "description": "Set true when you're satisfied and the conversation should end."},
        },
        "required": ["message", "done"],
    },
}


class PromptAgentTaskStep(TaskStep):
    type: ClassVar[str] = "prompt_agent"
    entity_refs = (EntityRef.env("snapshot_config.env_id"),)
    DEFAULT_PROMPT_TIMEOUT_SECONDS: ClassVar[int] = 600
    DEFAULT_POLL_INTERVAL_SECONDS: ClassVar[int] = 2

    def __init__(
        self,
        id: str,
        version: Optional[int],
        prompt: Optional[str] = None,
        parts: Optional[list[dict]] = None,
        prompt_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        timeout_seconds: int = DEFAULT_PROMPT_TIMEOUT_SECONDS,
        trajectory_output_prefix: Optional[str] = None,
        model: Optional[str] = None,
        model_params: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        max_turns: Optional[int] = None,
        output_format: Optional[dict[str, Any]] = None,
        max_thinking_tokens: Optional[int] = None,
        effort: Optional[str] = None,
        harness: Optional[str] = None,
        agentenv_tools: Optional[list[str]] = None,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        context_id: Optional[str] = None,
        max_conversation_turns: int = 1,
        user_agent_name: str = "human_agent",
        user_a2a_url: Optional[str] = None,
        user_agent_timeout_seconds: int = 600,
        user_output_format: Optional[dict[str, Any]] = None,
        user_model: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
        retry_config: Optional[RetryConfig | dict] = None,
        *,
        snapshot_config: Optional[SnapshotConfig] = None,
    ):
        # `prompt` is the legacy text-only field; `parts` is the multimodal
        # form that's the authoritative A2A wire shape. Exactly one must be set:
        # callers pre-parts pass `prompt="..."`; multimodal callers pass
        # `parts=[{"kind":"text",...}, {"kind":"file","file":{...}}, ...]`.
        if prompt is None and parts is None:
            raise ValueError("PromptAgentTaskStep requires either `prompt` or `parts`")
        if prompt is not None and parts is not None:
            raise ValueError("PromptAgentTaskStep accepts `prompt` OR `parts`, not both")
        if parts is not None:
            self._validate_parts(parts)
        if max_conversation_turns < 1:
            raise ValueError("max_conversation_turns must be >= 1")
        if model_params:
            reserved = set(model_params) & MODEL_PARAMS_RESERVED
            if reserved:
                raise ValueError(f"model_params may not set reserved keys {sorted(reserved)} (agent-env/the harness set these)")
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error, retry_config=retry_config)
        self.prompt_id = prompt_id or uuid.uuid4().hex
        self.prompt = prompt
        self.parts = parts
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.timeout_seconds = timeout_seconds
        self.trajectory_output_prefix = trajectory_output_prefix
        self.model = model
        self.model_params = model_params
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.output_format = output_format
        self.max_thinking_tokens = max_thinking_tokens
        self.effort = effort
        self.harness = harness
        self.agentenv_tools = agentenv_tools
        self.poll_interval_seconds = poll_interval_seconds
        self.context_id = context_id
        self.max_conversation_turns = max_conversation_turns
        self.user_agent_name = user_agent_name
        self.user_a2a_url = user_a2a_url
        self.user_agent_timeout_seconds = user_agent_timeout_seconds
        self.user_output_format = user_output_format
        self.user_model = user_model
        self.snapshot_config = snapshot_config

    @staticmethod
    def _validate_parts(parts: list[dict]) -> None:
        """Reject malformed parts at construction. Each entry must conform to
        the A2A `Part` discriminated union (text/file/data) per a2a-sdk 0.3.26."""
        from a2a.types import Part
        if not isinstance(parts, list):
            raise ValueError(f"`parts` must be a list, got {type(parts).__name__}")
        for i, p in enumerate(parts):
            if not isinstance(p, dict):
                raise ValueError(f"parts[{i}] must be a dict, got {type(p).__name__}")
            try:
                Part.model_validate(p)
            except Exception as e:
                raise ValueError(f"parts[{i}] is not a valid A2A Part: {e}") from e

    def _effective_parts(self) -> list[dict]:
        """Return the A2A wire-form parts list. If `parts` was set explicitly,
        return it; otherwise materialize a singleton text part from the legacy
        `prompt` field. Callers should treat the result as read-only and use
        `_apply_seed` to get a substituted copy before sending."""
        if self.parts is not None:
            return self.parts
        return [{"kind": "text", "text": self.prompt}]

    def _apply_seed(self, parts: list[dict], seed: dict) -> list[dict]:
        """Return a deep-copied parts list with seed substitutions applied.
        `<key>` placeholders are replaced inside text parts' `text` and inside
        file parts' `uri`/`name` (so tasks-as-templates with an object URL
        ending `seeds/<seed_id>/x.png` resolve per-run)."""
        if not seed:
            return parts
        out = copy.deepcopy(parts)
        for p in out:
            kind = p.get("kind")
            if kind == "text":
                for k, v in seed.items():
                    p["text"] = p["text"].replace(f"<{k}>", str(v))
            elif kind == "file":
                f = p.get("file") or {}
                for field in ("uri", "name"):
                    if field in f and isinstance(f[field], str):
                        for k, v in seed.items():
                            f[field] = f[field].replace(f"<{k}>", str(v))
        return out

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["prompt_id"] = self.prompt_id
        base["prompt"] = self.prompt
        base["parts"] = self.parts
        base["agent_name"] = self.agent_name
        base["timeout_seconds"] = self.timeout_seconds
        base["trajectory_output_prefix"] = self.trajectory_output_prefix
        base["model"] = self.model
        base["model_params"] = self.model_params
        base["system_prompt"] = self.system_prompt
        base["max_turns"] = self.max_turns
        base["output_format"] = self.output_format
        base["max_thinking_tokens"] = self.max_thinking_tokens
        base["effort"] = self.effort
        base["harness"] = self.harness
        base["agentenv_tools"] = self.agentenv_tools
        base["poll_interval_seconds"] = self.poll_interval_seconds
        base["context_id"] = self.context_id
        base["max_conversation_turns"] = self.max_conversation_turns
        base["user_agent_name"] = self.user_agent_name
        base["user_a2a_url"] = self.user_a2a_url
        base["user_agent_timeout_seconds"] = self.user_agent_timeout_seconds
        base["user_output_format"] = self.user_output_format
        base["user_model"] = self.user_model
        if self.snapshot_config:
            base["snapshot_config"] = self.snapshot_config.to_dict()
        return base

    @classmethod
    def from_dict(cls, data: dict) -> PromptAgentTaskStep:
        return cls(
            **cls._base_from_dict(data),
            retry_config=data.get("retry_config"),
            prompt=data.get("prompt"),
            parts=data.get("parts"),
            prompt_id=data.get("prompt_id"),
            agent_name=data.get("agent_name"),
            timeout_seconds=data.get("timeout_seconds", cls.DEFAULT_PROMPT_TIMEOUT_SECONDS),
            trajectory_output_prefix=data.get("trajectory_output_prefix"),
            model=data.get("model"),
            model_params=data.get("model_params"),
            system_prompt=data.get("system_prompt"),
            max_turns=data.get("max_turns"),
            output_format=data.get("output_format"),
            max_thinking_tokens=data.get("max_thinking_tokens"),
            effort=data.get("effort"),
            harness=data.get("harness"),
            agentenv_tools=data.get("agentenv_tools"),
            poll_interval_seconds=data.get("poll_interval_seconds", cls.DEFAULT_POLL_INTERVAL_SECONDS),
            context_id=data.get("context_id"),
            max_conversation_turns=data.get("max_conversation_turns", 1),
            user_agent_name=data.get("user_agent_name", "human_agent"),
            user_a2a_url=data.get("user_a2a_url"),
            user_agent_timeout_seconds=data.get("user_agent_timeout_seconds", 600),
            user_output_format=data.get("user_output_format"),
            user_model=data.get("user_model"),
            snapshot_config=(
                SnapshotConfig.from_dict(data["snapshot_config"])
                if data.get("snapshot_config")
                else None
            ),
        )

    def _trajectory_prefix(self) -> str:
        if self.trajectory_output_prefix:
            return self.trajectory_output_prefix
        config = get_config()
        return config.get_object_store().object_url(
            f"{config.get_artifact_key_prefix()}prompt_agent_trajectories/prompt_id={self.prompt_id}/"
        )

    def _warn_on_snapshot_budget(self) -> None:
        """Flag snapshot settings that will not capture what the caller asked for.

        Both cadences spend one shared budget, so a conversation longer than
        ``max_snapshots`` silently stops being captured partway through — the run
        still succeeds, and only the tail of the curve is missing. Warned at startup
        because by the time the rows are graded the run is over.
        """
        cfg = self.snapshot_config
        if not cfg or not cfg.per_turn:
            return
        if self.max_conversation_turns < 2:
            logger.warning(
                f"{self.id}: per_turn snapshotting is enabled but "
                f"max_conversation_turns={self.max_conversation_turns} (<2); there is "
                "no turn boundary to capture at, so only the final capture runs"
            )
            return
        # One boundary per turn except the last, which teardown's final capture covers.
        boundaries = self.max_conversation_turns - 1
        if cfg.ticks:
            logger.info(
                f"{self.id}: per_turn and interval_seconds={cfg.interval_seconds} share "
                f"the {cfg.max_snapshots}-capture budget, so ticks can consume slots "
                f"the {boundaries} turn boundaries would have used"
            )
        if boundaries > cfg.max_snapshots:
            logger.warning(
                f"{self.id}: max_snapshots={cfg.max_snapshots} is below the "
                f"{boundaries} turn boundaries in this conversation; captures stop "
                f"after turn {cfg.max_snapshots} and the rest are recorded as skipped"
            )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        # A changelog staged on the agent moves into the store while it works, and the rest when it is done.
        async with draining(staged_changelogs(context.metadata, agent_name=self.agent_name)):
            return await self._execute(context)

    async def _execute(self, context: TaskStepContext) -> TaskStepContext:
        # Minted here, not in `_execute_conversation`, so the series can address the
        # same session before the first turn.
        a2a_context_id = self.context_id or uuid.uuid4().hex
        if not self.snapshot_config:
            return await self._execute_conversation(context, a2a_context_id=a2a_context_id)
        series = SnapshotSeries(
            step_id=self.id,
            agent_name=self.agent_name,
            prompt_id=self.prompt_id,
            a2a_context_id=a2a_context_id,
            config=self.snapshot_config,
            trajectory_output_prefix=self._trajectory_prefix(),
            instance_id=context.instance_id,
        )
        self._warn_on_snapshot_budget()
        series.start(context)
        try:
            result = await self._execute_conversation(
                context, a2a_context_id=a2a_context_id, series=series
            )
        finally:
            # On failure too: a run that died is still a point worth grading.
            await series.finish(context)
        # After the finally, so it cannot replace the agent's own exception.
        series.raise_if_final_capture_missing()
        return result

    async def _execute_conversation(
        self,
        context: TaskStepContext,
        *,
        a2a_context_id: str,
        series: Optional[SnapshotSeries] = None,
    ) -> TaskStepContext:
        agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if agent is None:
            raise RuntimeError(f"Agent with name '{self.agent_name}' not found in context.deployed_agents")
        target_url = agent.a2a_url or agent.api_url
        user_sim = next((a for a in context.deployed_agents if a.agent_name == self.user_agent_name), None)
        user_url = self.user_a2a_url or (
            (user_sim.a2a_url or user_sim.api_url) if user_sim else None
        )
        if user_url is None and user_sim is None and self.max_conversation_turns > 1:
            user_url = get_config().get_default_human_a2a_url()

        trajectory_output_prefix = self._trajectory_prefix()

        system_prompt = self.system_prompt
        seed = context.metadata.get("seed", {})
        if seed and system_prompt:
            for key, value in seed.items():
                system_prompt = system_prompt.replace(f"<{key}>", str(value))
        initial_parts = self._apply_seed(self._effective_parts(), seed)
        prompt_text = initial_parts[0]["text"] if self.prompt is not None else None

        model = context.agent_model or self.model or context.default_agent_model
        harness = context.agent_harness or self.harness
        overrides = context.metadata.get("user_overrides", {})
        effort = overrides.get("agent_effort") or self.effort
        max_thinking_tokens = overrides.get("agent_max_thinking_tokens") or self.max_thinking_tokens
        max_turns = overrides.get("agent_max_turns") or self.max_turns

        card = agent.a2a_card or {}
        desired_config = {
            "model": model, "system_prompt": system_prompt, "effort": effort,
            "harness": harness, "max_turns": max_turns,
            "max_thinking_tokens": max_thinking_tokens,
            "output_format": self.output_format, "timeout_seconds": self.timeout_seconds,
            "task_id": context.metadata.get("task_id"),
            "agentenv_tools": self.agentenv_tools,
        }
        desired_config = {k: v for k, v in desired_config.items() if v is not None}
        model_params = get_config().get_model_params(self.model_params)
        if model_params:
            desired_config["model_params"] = model_params
        negotiated = A2AAgent.negotiate_agent_config(card, desired_config)
        if negotiated:
            await protocol.post_agent_config(target_url + negotiated.endpoint, negotiated.fields)

        is_user_sim = user_sim is not None and self.user_a2a_url is None
        if is_user_sim:
            user_output_format = self.user_output_format or _DEFAULT_USER_SIM_OUTPUT_FORMAT
            user_config = {k: v for k, v in {"output_format": user_output_format, "model": self.user_model}.items() if v is not None}
            negotiated = A2AAgent.negotiate_agent_config(user_sim.a2a_card or {}, user_config)
            if negotiated:
                await protocol.post_agent_config(user_url + negotiated.endpoint, negotiated.fields)
                logger.info(f"Configured user-sim '{self.user_agent_name}' with structured output_format for done-signal")

        triggers_registered = is_user_sim and any(
            r.get("agent_name") == self.user_agent_name
            for r in context.metadata.get("agent_trigger_registrations", [])
        )
        trig_ext = A2AAgent.find_extension(user_sim.a2a_card or {}, A2AAgent.EXT_TRIGGERS) if is_user_sim else None
        decide_url = state_url = None
        if trig_ext and triggers_registered:
            decide_method = A2AAgent.extension_method(user_sim.a2a_card, A2AAgent.EXT_TRIGGERS, "decide") or {}
            decide_url = user_url + decide_method.get("endpoint", "/ext/triggers/decide")
            state_url = user_url + (trig_ext.get("params") or {}).get("endpoint", "/ext/triggers")
            if self.max_conversation_turns < 2:
                logger.warning(
                    f"user agent '{self.user_agent_name}' advertises {A2AAgent.EXT_TRIGGERS} but "
                    f"max_conversation_turns={self.max_conversation_turns} (<2); the trigger engine is never invoked")

        solver_context_id = a2a_context_id
        if self.context_id:
            run_component = context.instance_id or uuid.uuid4().hex
            conversation_id = f"{self.context_id}-{run_component}-{len(run_component)}"
        else:
            conversation_id = solver_context_id
        try:
            conversation_store.create_conversation(
                conversation_id=conversation_id,
                task_instance_id=context.instance_id or "",
                source_agent_name=self.user_agent_name,
                target_agent_name=self.agent_name,
            )
        except DuplicateKeyError:
            # reuse a pre-existing conversation on retry/resume (store raises agent_env.store.DuplicateKeyError, not pymongo's)
            logger.info(f"Reusing existing conversation {conversation_id} on re-run")
        context.metadata.setdefault("a2a_conversations", {})[self.id] = conversation_id
        logger.info(
            f"Started multi-turn conversation {conversation_id} (max {self.max_conversation_turns} turns) "
            f"between {self.user_agent_name} and {self.agent_name}"
        )

        current_user_parts = initial_parts
        final_terminal = protocol.TerminalResponse(response_text="")
        final_task_id = ""
        per_turn_trajectory_urls: list[str | None] = []
        source_agent_per_turn_prompt_parts: list[list[dict] | None] = []
        traj_ext_cached = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
        final_state: str = TaskState.completed.value
        trajectory_url: Optional[str] = None
        # What the target has been sent: of the files its replies name, the only ones a user-sim is made able
        # to read, so a reply naming any other object a store owns can't read it out through the user-sim.
        sent_to_target: set[str] = set()

        for turn in range(self.max_conversation_turns):
            # `target_a2a_task_id` is the client A2A message id sent to the target
            # agent and recorded on the conversation as `a2a_task_id`.
            # This id is also used as part of the key name for the trajectory S3 object.
            target_a2a_task_id = uuid.uuid4().hex

            # The turn records the objects' own URLs, and only once the parts are ready to send, so an
            # object the agent can't be sent leaves no turn waiting.
            def record_turn() -> None:
                sent_to_target.update(_file_uris(current_user_parts))
                conversation_store.add_a2a_task(
                    conversation_id=conversation_id,
                    parts=current_user_parts,
                    a2a_task_id=target_a2a_task_id,
                    role="user",
                )
                # Turn 0 of a prompt-mode step re-sends exactly the text already
                # persisted as PromptResponse.prompt_text; store a None placeholder
                # instead of a second copy so the doc doesn't carry the prompt twice.
                # The list stays index-aligned with the per-turn trajectory URIs.
                source_agent_per_turn_prompt_parts.append(
                    None
                    if turn == 0 and _duplicates_prompt_text(current_user_parts, prompt_text)
                    else list(current_user_parts)
                )

            sent_task_id, result = await send_and_wait(
                target_url, current_user_parts, agent=agent, message_id=target_a2a_task_id,
                context_id=solver_context_id, timeout_seconds=self.timeout_seconds,
                poll_interval_seconds=self.poll_interval_seconds, before_send=record_turn,
            )
            target_state = result["status"]["state"]
            status_msg = (result.get("status") or {}).get("message") or {}
            final_terminal = protocol.TerminalResponse.from_message(status_msg)
            agent_response_parts = status_msg.get("parts") or [{"kind": "text", "text": final_terminal.response_text}]

            conversation_store.complete_a2a_task(
                conversation_id=conversation_id,
                parts=agent_response_parts,
                role="agent",
            )

            if traj_ext_cached:
                turn_traj_uri: str | None = None
                try:
                    turn_traj_uri = await self._fetch_trajectory(
                        target_url, traj_ext_cached, sent_task_id, trajectory_output_prefix,
                        target_a2a_task_id, card=card, sandbox_type=transfer_sandbox_type(agent),
                    )
                except Exception as e:
                    logger.warning(f"Turn {turn+1} trajectory fetch failed (continuing): {e}")
                per_turn_trajectory_urls.append(turn_traj_uri)

            final_task_id = sent_task_id
            final_state = target_state
            logger.info(f"Turn {turn+1}/{self.max_conversation_turns} agent response ({target_state}): {final_terminal.response_text[:120]}...")

            if target_state == TaskState.failed:
                conversation_store.mark_closed(conversation_id)
                break
            if turn == self.max_conversation_turns - 1:
                break

            conv_doc = conversation_store.get_conversation(conversation_id)
            if conv_doc and conv_doc.get("status") != "active":
                logger.info(f"Conversation {conversation_id} no longer active (status={conv_doc.get('status')}); ending multi-turn")
                break

            # After every `break` above: a turn that ends the conversation is covered
            # by teardown's final capture instead of being captured twice.
            if series is not None:
                await series.capture_turn(context, turn + 1)

            if trig_ext and triggers_registered:
                env_triggers = await self._read_env_triggers(context)
                (context.metadata.setdefault("env_trigger_snapshots", {})
                    .setdefault(self.id, []).append({"turn": turn + 1, "envs": env_triggers}))
                resp = await self._decide(decide_url, turn + 1, final_terminal.response_text, conversation_id, env_triggers)
                (context.metadata.setdefault("agent_trigger_firings", {})
                    .setdefault(self.id, []).append({"turn": turn + 1, "fired": resp.get("fired", [])}))
                logger.info(f"Turn {turn+1} agent-trigger decide: fired={resp.get('fired', [])} done={resp.get('done')}")
                if bool(resp.get("done")):
                    current_user_parts = resp.get("parts") or []
                    conversation_store.add_a2a_task(conversation_id, parts=current_user_parts,
                                                    a2a_task_id=uuid.uuid4().hex, role="user")
                    conversation_store.mark_closed(conversation_id)
                    break
                if resp.get("fired"):
                    current_user_parts = resp.get("parts") or []
                    continue
                # no trigger fired -> fall through to the LLM user-sim for this turn (hybrid)

            user_a2a_task_id = uuid.uuid4().hex
            try:
                # A user-sim runs in a sandbox agent-env deployed; a human peer, registered or named by
                # user_a2a_url, has none and reads the store itself.
                _, user_result = await send_and_wait(
                    user_url, agent_response_parts,
                    agent=user_sim if is_user_sim and user_sim.sandbox_id else None, shareable=sent_to_target,
                    message_id=user_a2a_task_id, context_id=conversation_id,
                    timeout_seconds=self.user_agent_timeout_seconds,
                    poll_interval_seconds=self.poll_interval_seconds,
                )
            except TimeoutError as e:
                logger.warning(f"user_a2a_url timeout for conversation {conversation_id} ({e}); marking abandoned")
                conversation_store.mark_closed(conversation_id)
                break

            user_state = (user_result.get("status") or {}).get("state")
            if user_state != TaskState.completed:
                logger.info(f"user_a2a_url returned state={user_state}; ending multi-turn")
                conversation_store.mark_closed(conversation_id)
                break

            user_msg = (user_result.get("status") or {}).get("message") or {}
            current_user_parts = user_msg.get("parts") or []
            if not current_user_parts:
                logger.warning(f"user_a2a_url returned no parts; ending multi-turn")
                conversation_store.mark_closed(conversation_id)
                break

            if is_user_sim:
                raw_text = next((p["text"] for p in current_user_parts if p.get("kind") == "text"), "")
                try:
                    parsed = json.loads(raw_text)
                    message_text = parsed["message"]
                    done = bool(parsed["done"])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"user-sim returned malformed output despite output_format ({e}); using raw text as next prompt; raw={raw_text[:200]!r}")
                    current_user_parts = [{"kind": "text", "text": raw_text}]
                else:
                    extras = {k: v for k, v in parsed.items() if k not in ("message", "done")}
                    if extras:
                        (context.metadata.setdefault("usersim_turn_outputs", {})
                            .setdefault(self.id, []).append({"turn": turn + 1, "fields": extras}))
                    current_user_parts = [{"kind": "text", "text": message_text}]
                    if done:
                        logger.info(f"user-sim signaled done=true at turn {turn+1}; ending multi-turn")
                        conversation_store.mark_closed(conversation_id)
                        break

        await self._persist_env_trigger_state(context)

        final_conv = conversation_store.get_conversation(conversation_id)
        if final_conv and final_conv.get("status") == "active":
            conversation_store.mark_closed(conversation_id)

        if trig_ext and triggers_registered:  # best-effort firing-log readback (provenance only; never fails the run)
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(state_url, timeout=15)
                    r.raise_for_status()
                    context.metadata.setdefault("agent_trigger_state", {})[self.id] = r.json().get("firing_log", [])
            except Exception as e:
                logger.warning(f"agent-trigger state readback failed (continuing): {e}")

        trajectory_url = next(
            (uri for uri in reversed(per_turn_trajectory_urls) if uri is not None),
            None,
        )

        container_logs: Optional[str] = None
        if final_state == TaskState.failed:
            logger.error(
                "A2A task %s failed (agent=%s sandbox=%s error_type=%s exception=%s). "
                "Raw A2A status message: %s",
                final_task_id, self.agent_name, getattr(agent, "sandbox_id", None),
                final_terminal.error_type, final_terminal.error_class, json.dumps(status_msg, default=str)[:4000],
            )
            container_logs = await fetch_container_logs(agent)
            if container_logs:
                logger.error(
                    "Agent container logs for failed A2A task %s (agent=%s):\n%s",
                    final_task_id, self.agent_name, container_logs,
                )

        final_error_for_response: Optional[str] = None
        if final_state == TaskState.failed:
            final_error_for_response = (
                agent_error_text(final_terminal.error_message, final_terminal.response_text)
                or container_logs
                or None
            )

        # Prefer the agent's typed structured output; fall back to parsing the response text.
        structured_output = (
            final_terminal.structured_output
            if final_terminal.structured_output is not None
            else _parse_structured_output(final_terminal.response_text)
        )

        context.prompt_responses.append(PromptResponse(
            prompt_id=self.prompt_id,
            response=final_terminal.response_text,
            prompt_text=prompt_text,
            agent_trajectory_object_url=trajectory_url,
            agent_trajectory_object_prefix=trajectory_output_prefix,
            target_agent_per_turn_trajectory_object_urls=per_turn_trajectory_urls or None,
            source_agent_per_turn_prompt_parts=source_agent_per_turn_prompt_parts or None,
            tool_call_count=final_terminal.tool_call_count,
            model=model,
            error_type=final_terminal.error_type,
            error_code=final_terminal.error_code,
            error_message=final_error_for_response,
            agent_session_id=final_task_id,
            a2a_context_id=solver_context_id,
            agent_name=self.agent_name,
            step_id=self.id,
            structured_output=structured_output,
        ))

        if final_state == TaskState.failed:
            detail_parts = []
            if final_terminal.error_type:
                detail_parts.append(f"error_type={final_terminal.error_type}")
            if final_terminal.error_code:
                detail_parts.append(f"error_code={final_terminal.error_code}")
            if final_terminal.error_class:
                detail_parts.append(f"exception={final_terminal.error_class}")
            detail_parts.append(f"task_id={final_task_id}")
            sandbox_id = getattr(agent, "sandbox_id", None)
            if sandbox_id:
                detail_parts.append(f"agent_sandbox={sandbox_id}")
            detail = ", ".join(detail_parts)
            body = agent_error_text(final_terminal.error_message, final_terminal.response_text)
            if container_logs:
                body = f"{body}\n{container_logs}" if body else container_logs
            if not body:
                body = (
                    "agent returned no error detail and no container logs were "
                    f"retrievable — inspect the agent sandbox directly (sandbox={sandbox_id})"
                )
            raise RuntimeError(f"A2A task failed ({detail}): {body}")

        return context

    async def _fetch_trajectory(
        self, a2a_url: str, traj_ext: dict, a2a_server_task_id: str, trajectory_output_prefix: str,
        target_a2a_task_id: str, *, sandbox_type: str | None, card: dict | None = None,
    ) -> str | None:
        get_method, get_path = A2AAgent.operation(traj_ext, "get")
        store = transfer_store(get_config().get_object_store(), a2a_url, card, sandbox_type=sandbox_type)
        mode = trajectory_mode(get_method, store, by="task_id", sandbox_type=sandbox_type)
        if mode is None:
            raise RuntimeError(
                "Agent advertises no trajectory get form this object store can serve"
            )
        # Fetched by a2a_server_task_id, the id the A2A server produced; stored under
        # target_a2a_task_id, the client message id persisted on the conversation as
        # a2a_task_id, so the FE can resolve it.
        upload = None
        if mode == "objects":
            upload = await asyncio.to_thread(
                TrajectoryUpload.to,
                store, trajectory_object_url(trajectory_output_prefix, name=target_a2a_task_id, store=store),
            )
        fetched = await fetch_trajectory(a2a_url + get_path, {"task_id": a2a_server_task_id}, upload=upload, store=store)
        return await asyncio.to_thread(store_trajectory, fetched, trajectory_output_prefix, name=target_a2a_task_id)

    async def _read_env_triggers(self, context: TaskStepContext) -> dict:
        """Snapshot each deployed env's triggers state as {env_id: {trigger_id: status}}; {} when its card offers none; fail-closed on a read error."""
        env_triggers: dict = {}
        for d in context.deployed_envs:
            if not d.supports(EXT_TRIGGERS_URI, "state"):
                env_triggers[d.env_id] = {}
                continue
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    state = await d.invoke(EXT_TRIGGERS_URI, "state", timeout=15)
                    env_triggers[d.env_id] = {t["id"]: t.get("status") for t in state.get("triggers", [])}
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    await asyncio.sleep(0.5 * (attempt + 1))
            if last_exc is not None:
                raise RuntimeError(f"fail-closed: could not read /triggers/state for env '{d.env_id}': {last_exc}")
        return env_triggers

    async def _decide(self, decide_url: str, turn: int, solver_message: str, context_id: str, env_triggers: dict) -> dict:
        """Call the usersim's /decide for the triggers that should fire this turn."""
        payload = {"turn": turn, "solver_message": solver_message, "context_id": context_id, "env_triggers": env_triggers}
        async with httpx.AsyncClient() as client:
            resp = await client.post(decide_url, json=payload, timeout=self.user_agent_timeout_seconds)
            if resp.status_code >= 400:
                raise RuntimeError(f"agent-trigger /decide failed (HTTP {resp.status_code}): {resp.text}")
            return resp.json()

    _CAPTURE_BUDGET_SECONDS: ClassVar[float] = 90

    async def _persist_env_trigger_state(self, context: TaskStepContext) -> None:
        """Snapshot each trigger-registered env's /triggers/state to the configured object store, plus a
        metadata summary. Never raises and is bounded by _CAPTURE_BUDGET_SECONDS."""
        try:
            await asyncio.wait_for(
                self._capture_env_trigger_state(context), timeout=self._CAPTURE_BUDGET_SECONDS)
        except Exception as e:
            detail = (f"capture budget of {self._CAPTURE_BUDGET_SECONDS}s exceeded"
                      if isinstance(e, TimeoutError) else f"{type(e).__name__}: {str(e)[:200]}")
            logger.warning(f"env trigger state capture aborted (continuing): {detail}")
            registered_env_ids = self._registered_env_ids(context)
            for deployed in context.deployed_envs:
                if deployed.env_id not in registered_env_ids:
                    continue
                entries = context.metadata.setdefault("env_trigger_state", {})
                if deployed.env_id not in entries:
                    entries[deployed.env_id] = {
                        "error": detail,
                        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                        "capture_step_id": self.id,
                    }

    @staticmethod
    def _registered_env_ids(context: TaskStepContext) -> set[str]:
        registrations = context.metadata.get("env_trigger_registrations") or []
        return {r["env_id"] for r in registrations if isinstance(r, dict) and r.get("env_id")}

    async def _capture_env_trigger_state(self, context: TaskStepContext) -> None:
        from agent_env.env.env import gateway_url_of

        registered_env_ids = self._registered_env_ids(context)
        if not registered_env_ids:
            return
        store = get_config().get_object_store()
        for deployed in context.deployed_envs:
            if deployed.env_id not in registered_env_ids:
                continue
            entry: dict[str, Any] = {
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "capture_step_id": self.id,
            }
            try:
                state = await self._fetch_settled_trigger_state(deployed)
                clock = await self._get_clock_state(deployed)
                # Stamped at the read: the settle poll above can take a minute, which at a high
                # rate is many virtual days of anchor error.
                clock_read_at_utc = datetime.now(timezone.utc).isoformat()
                entry["statuses"] = {t["id"]: t.get("status") for t in state.get("triggers", [])}
                # A recurrence returns to "armed" after every arrival, so only fire_count shows it ran.
                entry["triggers"] = {
                    t["id"]: {k: t.get(k) for k in (
                        "type", "status", "fire_count", "next_mark", "failure_count", "last_failure_at",
                    )}
                    for t in state.get("triggers", [])
                }
                # len(events) alone plateaus at the gateway's log cap.
                dropped = state.get("events_dropped") or 0
                entry["event_count"] = len(state.get("events", [])) + dropped
                entry["events_dropped"] = dropped
                entry["capture_is_final"] = self._capture_is_final(state)
                payload = {
                    "instance_id": context.instance_id,
                    "env_id": deployed.env_id,
                    "gateway_url": gateway_url_of(deployed),
                    "capture_source": "prompt_agent",
                    "captured_at_utc": entry["captured_at_utc"],
                    "clock": clock,
                    "clock_read_at_utc": clock_read_at_utc,
                    "capture_is_final": entry["capture_is_final"],
                    "state": state,
                }
                instance_key = context.instance_id or f"adhoc-{uuid.uuid4().hex[:12]}"
                key = (
                    f"{get_config().get_artifact_key_prefix()}env_trigger_state/"
                    f"instance_id={instance_key}/{deployed.env_id}-{uuid.uuid4().hex[:8]}.json"
                )
                entry["object_url"] = await asyncio.to_thread(
                    store.put, key, json.dumps(payload, indent=2, default=str).encode(),
                    content_type="application/json",
                )
            except Exception as e:
                logger.warning(f"env trigger state capture failed for env '{deployed.env_id}' (continuing): {e}")
                entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            context.metadata.setdefault("env_trigger_state", {})[deployed.env_id] = entry

    @staticmethod
    def _is_settling(state: dict) -> bool:
        """Whether a trigger still has fire work outstanding.

        An unbounded recurrence is excluded: it re-enters ``firing`` every arrival and never settles."""
        for t in state.get("triggers", []):
            # `pending` too: the queue itself, not a label, so it outlives a status we do not know.
            if t.get("status") not in TRIGGER_IN_FLIGHT_STATUSES and not t.get("pending"):
                continue
            when = t.get("when")
            if when is None:
                # Older gateway with no ``when``: a scheduled next arrival is the only tell
                # it's a recurrence.
                if t.get("next_mark"):
                    continue
                return True
            if when.get("type") == "time" and "every" in when and not ("count" in when or "until" in when):
                continue
            return True
        return False

    @staticmethod
    def _capture_is_final(state: dict) -> Optional[bool]:
        """Whether the record can still change after this capture; None when not determinable.

        An older gateway has no ``type``, so an unresolved event-anchored trigger is
        indistinguishable from an action trigger — claiming completeness there would be a false signal."""
        rows = state.get("triggers", [])
        if any("type" not in t for t in rows):
            return None
        return not any(t.get("status") in TRIGGER_IN_FLIGHT_STATUSES or t.get("pending")
                       or (t["type"] == "time" and t.get("status") == "armed") for t in rows)

    async def _fetch_settled_trigger_state(self, deployed, settle_timeout_seconds: float = 60) -> dict:
        """Full /triggers/state, re-polled briefly while a one-shot trigger is mid-``firing``."""
        state = await self._get_trigger_state(deployed)
        deadline = time.monotonic() + settle_timeout_seconds
        while self._is_settling(state) and time.monotonic() < deadline:
            await asyncio.sleep(5)
            state = await self._get_trigger_state(deployed)
        return state

    async def _get_clock_state(self, deployed) -> Optional[dict]:
        """The gateway clock at capture time; None when its card offers no clock state, or it is unreachable."""
        try:
            # Tight: runs inside the shared capture budget, after the settle poll.
            return await deployed.invoke(EXT_CLOCK_URI, "state", timeout=5)
        except Exception as e:
            logger.warning(f"clock state capture failed for env '{deployed.env_id}' (continuing): {e}")
            return None

    async def _get_trigger_state(self, deployed) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                return await deployed.invoke(EXT_TRIGGERS_URI, "state", timeout=15)
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"could not read /triggers/state for env '{deployed.env_id}': {last_exc}")
