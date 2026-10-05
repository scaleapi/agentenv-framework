"""Verify agent output against rubric criteria using an LLM judge."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from agent_env.a2a_agent.object_transfer import (
    TrajectoryUpload,
    fetch_trajectory,
    trajectory_mode,
)
from agent_env.config import get_config
from agent_env.config.model import ModelParam
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    SCREENSHOT_GROUNDING,
    JudgeOutputFormat,
    JudgeOutputFormatSpec,
    ResultRows,
    evidence_checks_instruction,
    get_judge_output_format_spec,
    per_criterion_grounding,
    trajectory_mistakes_rows_from_evidence,
)
from agent_env.task_step.snapshot_utils.agent_state_capture import (
    store_trajectory,
    trajectory_object_url,
)
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score
from agent_env.task_step.task_steps.sandbox_utils.sandbox_utils import (
    agent_error_text,
    fetch_container_logs,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.frame_selection import (
    LabeledFrame,
    Transcript,
    action_log,
    select_frames_per_criterion,
    transcript_from_raw,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    CompactionType,
    FrameStrategy,
    ImageFrame,
    TrajectoryFilter,
    TrajectoryText,
    compact_otel_trajectory,
    final_image_frames,
    strip_trajectory_images,
)

logger = logging.getLogger(__name__)


# The judge echoes each criterion's id back and results are matched by exact string.
# Models drop or transpose a character when copying a 36-char UUID, and one mismatched
# id discards the whole judge response. Short positional keys (c1..cN) remove that
# transcription surface; the real ids stay the source of truth and are restored on
# the graded rows.
def _to_short_keyed_criteria(criteria: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Return (criteria re-keyed as c1..cN, {short_id: real_id}). Does not mutate the input.

    Raises ValueError if the criterion ids are not unique non-empty strings — a
    malformed rubric that generated keys would otherwise mask.
    """
    ids = [c.get("id") for c in criteria]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("rubric criteria must have unique, non-empty string ids")
    short_criteria: list[dict] = []
    real_by_short: dict[str, str] = {}
    for index, criterion in enumerate(criteria, start=1):
        short_id = f"c{index}"
        real_by_short[short_id] = criterion.get("id")
        short_criteria.append({**criterion, "id": short_id})
    return short_criteria, real_by_short


def _restore_criterion_ids(results: list[dict], real_by_short: dict[str, str]) -> None:
    """Rewrite each row's short id (c1..cN) back to its real criterion id, in place.

    Rows whose id is not a known short key are left untouched, so this is safe to run
    over default/failure rows or an already-real id.
    """
    for row in results:
        real_id = real_by_short.get(row.get("id"))
        if real_id is not None:
            row["id"] = real_id


_FRAME_LABEL_ID = re.compile(r"\[(c\d+)\]")


def _restore_frame_labels(rows: list[dict], real_by_short: dict[str, str]) -> None:
    """Rewrite the short criterion keys inside cited frame labels ("[c1][c3] action 17") to the real ids, in place."""
    for row in rows:
        frame = row.get("frame")
        if isinstance(frame, str) and frame:
            row["frame"] = _FRAME_LABEL_ID.sub(lambda m: f"[{real_by_short.get(m.group(1), m.group(1))}]", frame)


# A rubric_evidence_with_mistakes step writes its trajectory-mistakes entry under its own verifier_id plus this
# suffix (``trajectory_mistakes_key``).
TRAJECTORY_MISTAKES_SUFFIX = "-trajectory-mistakes"


def trajectory_mistakes_key(verifier_id: str) -> str:
    """The ``verifications`` key a ``rubric_evidence_with_mistakes`` step writes its second (trajectory_mistakes-
    shaped) entry under: the step's own ``verifier_id`` + ``TRAJECTORY_MISTAKES_SUFFIX``. A reader that knows
    the verifier id rebuilds the key from it; the entry also carries ``source_verifier_id``."""
    return f"{verifier_id}{TRAJECTORY_MISTAKES_SUFFIX}"


# A context_facts_for_judge path: dotted segments of letters, digits, `_` and `-` (dict keys, or a list index).
_CONTEXT_FACT_PATH = re.compile(r"^[\w-]+(\.[\w-]+)*$")
CONTEXT_FACT_MAX_CHARS = 600                    # longer rendered values are cut and end in "…"
CONTEXT_FACT_NOT_AVAILABLE = "(not available)"  # rendered for a path that does not resolve
_MISSING = object()


def _lookup_dotted_path(root: object, path: str) -> object:
    node = root
    for segment in path.split("."):
        if isinstance(node, dict):
            if segment not in node:
                return _MISSING
            node = node[segment]
        elif isinstance(node, list) and segment.isdigit() and int(segment) < len(node):
            node = node[int(segment)]
        else:
            return _MISSING
    return node


def _render_fact(value: object) -> str:
    """Compact JSON with sorted keys; stored key order when the keys cannot be sorted; ``str`` when not JSON."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        pass
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def resolve_context_facts(facts: dict[str, str], metadata: dict) -> list[tuple[str, str]]:
    """Resolve ``{label: dotted path}`` against ``metadata`` into ``[(label, rendered value)]``, in order.
    A path walks dicts by key and lists by index; one that does not resolve renders as
    ``CONTEXT_FACT_NOT_AVAILABLE`` so the judge sees the fact was requested. Never raises."""
    rendered: list[tuple[str, str]] = []
    for label, path in facts.items():
        value = _lookup_dotted_path(metadata, path)
        if value is _MISSING:
            text = CONTEXT_FACT_NOT_AVAILABLE
        else:
            text = _render_fact(value)
            if len(text) > CONTEXT_FACT_MAX_CHARS:
                text = text[:CONTEXT_FACT_MAX_CHARS] + "…"
        rendered.append((label, text))
    return rendered


# Only fall back from structured output when the error clearly says the response
# format itself is unsupported. Image/provider/transient failures should retry as-is.
_SCHEMA_TERMS = (
    "response_format",
    "response format",
    "json_schema",
    "json schema",
    "response schema",
    "structured output",
)
_UNSUPPORTED_TERMS = (
    "not supported",
    "unsupported",
    "not allowed",
    "does not support",
    "cannot be used",
    "incompatible",
)


def _looks_like_response_format_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _SCHEMA_TERMS) and any(u in msg for u in _UNSUPPORTED_TERMS)


def _image_url_block(frame: ImageFrame) -> dict:
    """Adapt a neutral trajectory image to LiteLLM's image block shape."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{frame.media_type};base64,{frame.data}"},
    }


@dataclass
class _CompactedTrajectory:
    """Every representation the filter produced; the caller picks what its judge consumes."""

    container_s3_uri: Optional[str] = None       # trajectory file (loaded into an agent-judge container)
    tool_result_files: list[tuple[str, str]] = field(default_factory=list)
    inline_text: Optional[str] = None            # pre-rendered text (screenshot); None for DEFAULT (read lazily)
    image_blocks: list[dict] = field(default_factory=list)  # frame images (screenshot); a label text block before each per-criterion
    compact_s3_uri: Optional[str] = None         # compacted URI, if produced; recorded on the result only
    per_criterion: bool = False                  # labelled per-criterion frames were attached (False when that selection found none)

    @classmethod
    def empty(cls) -> "_CompactedTrajectory":
        return cls()

    @classmethod
    def for_default(
        cls, container_s3_uri: Optional[str], tool_result_files: list[tuple[str, str]],
        compact_s3_uri: Optional[str] = None,
    ) -> "_CompactedTrajectory":
        return cls(container_s3_uri=container_s3_uri, tool_result_files=tool_result_files,
                   compact_s3_uri=compact_s3_uri)

    @classmethod
    def for_screenshot(
        cls, inline_text: Optional[str], image_blocks: list[dict],
        container_s3_uri: Optional[str] = None, per_criterion: bool = False,
    ) -> "_CompactedTrajectory":
        # container_s3_uri = raw file, kept so an agent judge (no inline frames) can still load it.
        return cls(inline_text=inline_text, image_blocks=image_blocks or [], container_s3_uri=container_s3_uri,
                   per_criterion=per_criterion)


def _image_count(blocks: list[dict]) -> int:
    """Images among judge message blocks (per-criterion frames interleave a ``FRAME:`` text block before each)."""
    return sum(1 for b in blocks if b.get("type") == "image_url")


def _labeled_image_blocks(frames: list[LabeledFrame]) -> list[dict]:
    """``FRAME: <label>`` text block, then the image — the judge cites the label in each row's ``frame``."""
    blocks: list[dict] = []
    for f in frames:
        blocks.append({"type": "text", "text": f"FRAME: {f.label}"})
        blocks.append(_image_url_block(f.frame))
    return blocks


def _is_per_criterion(tf: TrajectoryFilter | None) -> bool:
    return (tf is not None and tf.compaction_type == CompactionType.SCREENSHOT
            and tf.frame_strategy == FrameStrategy.PER_CRITERION)


class RubricsVerifierTaskStep(TaskStep):
    type: ClassVar[str] = "rubrics_verifier"
    entity_refs = (EntityRef.agent("judge_a2a_agent_id"),)
    DEFAULT_JUDGE_TIMEOUT_SECONDS: ClassVar[int] = 1000
    DEFAULT_MAX_RETRIES: ClassVar[int] = 3
    DEFAULT_POLL_INTERVAL_SECONDS: ClassVar[int] = 10
    TRAJECTORY_CONTAINER_PATH: ClassVar[str] = "/tmp/prompt_trajectory.json"
    # Multi-turn agent judge: base for a per-run subdir holding turn_NN.json files.
    TRAJECTORY_CONTAINER_DIR: ClassVar[str] = "/tmp/prompt_trajectory_turns"
    _MULTITURN_DIR_NOTE: ClassVar[str] = (
        "\n\nNOTE: This is a multi-turn conversation. The trajectory path above is a "
        "DIRECTORY containing {n} files named turn_01.json, turn_02.json, … — one per "
        "conversation turn, in chronological order. Read every file, in order, to judge the "
        "full conversation; actions the agent took in earlier turns count."
    )
    _CONTEXT_FACTS_LEAD: ClassVar[str] = (
        "These values were measured from the run's record. Rely on them over what the screenshots or "
        "trajectory text appear to show."
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        criteria: list[dict],
        prompt_id: str,
        default_model: str = "claude-sonnet-4-6",
        default_model_api_base: Optional[str] = None,
        score_aggregator: Optional[ScoreAggregator] = None,
        verifier_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        trajectory_filter: Optional[TrajectoryFilter] = None,
        max_thinking_tokens: Optional[int] = None,
        effort: Optional[str] = None,
        use_agent_judge: bool = True,
        use_trajectory: bool = True,
        judge_a2a_agent_id: Optional[str] = None,
        judge_sandbox_type: Optional[str] = None,
        judge_timeout_seconds: Optional[int] = None,
        output_format: JudgeOutputFormat | str = JudgeOutputFormat.RUBRIC_BINARY,
        grading_policy_prompt: Optional[str] = None,
        context_facts_for_judge: Optional[dict[str, str]] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        # Task-authored rules the criteria alone do not state, appended as a "## Grading policy" section;
        # with rubric_evidence its findings come back as `checks`.
        self.grading_policy_prompt = grading_policy_prompt
        # {label: dotted path into context.metadata}, resolved at execute time (``resolve_context_facts``)
        # and shown as a "## Context facts" section the judge trusts over the pixels.
        if context_facts_for_judge is not None:
            if not isinstance(context_facts_for_judge, dict) or not all(
                isinstance(label, str) and label and isinstance(path, str) and _CONTEXT_FACT_PATH.match(path)
                for label, path in context_facts_for_judge.items()
            ):
                raise ValueError(
                    "context_facts_for_judge must map non-empty labels to dotted paths of identifiers "
                    "(letters, digits, '_' and '-' per segment), e.g. "
                    '{"items in cart": "verifications.cart-check.count"}.'
                )
        self.context_facts_for_judge = context_facts_for_judge
        if isinstance(output_format, str):
            output_format = JudgeOutputFormat(output_format)
        self.output_format = output_format
        self.verifier_id = verifier_id or uuid.uuid4().hex
        self.criteria = criteria
        self.prompt_id = prompt_id
        self.default_model = default_model
        self.default_model_api_base = default_model_api_base
        if isinstance(score_aggregator, str):
            score_aggregator = ScoreAggregator(score_aggregator)
        self.score_aggregator = score_aggregator or ScoreAggregator.ALL_PASS
        self.agent_name = agent_name
        if isinstance(trajectory_filter, dict):
            trajectory_filter = TrajectoryFilter.from_dict(trajectory_filter)
        self.trajectory_filter = trajectory_filter
        self.max_thinking_tokens = max_thinking_tokens
        self.effort = effort
        self.use_agent_judge = use_agent_judge
        self.use_trajectory = use_trajectory
        self.judge_a2a_agent_id = judge_a2a_agent_id
        self.judge_sandbox_type = judge_sandbox_type
        self.judge_timeout_seconds = judge_timeout_seconds or self.DEFAULT_JUDGE_TIMEOUT_SECONDS
        if self.default_model_api_base and self.use_agent_judge:
            # Agent judges route through their deployed runtime, so this direct-judge
            # default would otherwise be silently ignored.
            raise ValueError(
                "default_model_api_base only applies to the direct LLM judge "
                "(use_agent_judge=False); it is ignored by the agent judge."
            )
        if self._spec().cites_evidence and not (
            not self.use_agent_judge and self.use_trajectory and _is_per_criterion(self.trajectory_filter)
        ):
            # rubric_evidence / rubric_evidence_with_mistakes: the rows cite labelled per-criterion frames; only
            # the direct judge gets them.
            raise ValueError(
                f"output_format={self.output_format.value} requires the direct LLM judge (use_agent_judge=False) "
                "with use_trajectory=True and a trajectory_filter with compaction_type=screenshot and "
                "frame_strategy=per_criterion."
            )

    def _spec(self) -> JudgeOutputFormatSpec:
        return get_judge_output_format_spec(self.output_format)

    @property
    def _writes_mistakes_entry(self) -> bool:
        """rubric_evidence_with_mistakes: the same judge call also detects trajectory mistakes and a second entry,
        shaped and scored as a standalone trajectory_mistakes step's, is written under
        ``trajectory_mistakes_key(self.verifier_id)`` (see ``_write_entries``)."""
        return self.output_format is JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES

    def _entry_format_fields(self) -> dict:
        """``format`` on the verification entry — the shape of its rows (``JudgeOutputFormatSpec.stored_format``)
        — plus ``judge_output_format``, the format the judge actually ran, when the two differ."""
        stored = self._spec().stored_format
        if stored is None:
            return {"format": self.output_format.value}
        return {"format": stored, "judge_output_format": self.output_format.value}

    def _mistakes_entry_fields(self) -> dict:
        """``format`` / ``source_verifier_id`` of the second entry written under ``trajectory_mistakes_key``."""
        return {"format": JudgeOutputFormat.TRAJECTORY_MISTAKES.value, "source_verifier_id": self.verifier_id}

    def _write_skipped_entries(self, context: TaskStepContext, row_id: str, message: str) -> None:
        """Store a zero-score entry with one row saying why the run was not judged — under ``verifier_id``
        and, for rubric_evidence_with_mistakes, ``trajectory_mistakes_key(verifier_id)``."""
        def skipped(fields: dict) -> dict:
            return {
                **fields,
                "results": [{"id": row_id, "score": 0, "result": False, "message": message}],
                "score": 0,
            }
        verifications = context.metadata.setdefault("verifications", {})
        verifications[self.verifier_id] = skipped(self._entry_format_fields())
        if self._writes_mistakes_entry:
            verifications[trajectory_mistakes_key(self.verifier_id)] = skipped(self._mistakes_entry_fields())

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["verifier_id"] = self.verifier_id
        base["criteria"] = self.criteria
        base["prompt_id"] = self.prompt_id
        base["default_model"] = self.default_model
        base["default_model_api_base"] = self.default_model_api_base
        base["score_aggregator"] = self.score_aggregator.value
        base["agent_name"] = self.agent_name
        base["trajectory_filter"] = self.trajectory_filter.to_dict() if self.trajectory_filter else None
        base["max_thinking_tokens"] = self.max_thinking_tokens
        base["effort"] = self.effort
        base["use_agent_judge"] = self.use_agent_judge
        base["use_trajectory"] = self.use_trajectory
        base["judge_a2a_agent_id"] = self.judge_a2a_agent_id
        base["judge_sandbox_type"] = self.judge_sandbox_type
        base["judge_timeout_seconds"] = self.judge_timeout_seconds
        base["output_format"] = self.output_format.value
        base["grading_policy_prompt"] = self.grading_policy_prompt
        base["context_facts_for_judge"] = self.context_facts_for_judge
        return base

    @classmethod
    def from_dict(cls, data: dict) -> RubricsVerifierTaskStep:
        from agent_env.config import get_config

        raw_agg = data.get("score_aggregator")
        raw_tf = data.get("trajectory_filter")
        return cls(
            **cls._base_from_dict(data),
            criteria=data["criteria"],
            prompt_id=data["prompt_id"],
            default_model=data.get("default_model") or get_config().get_model_for_role("judge") or "claude-sonnet-4-6",
            default_model_api_base=data.get("default_model_api_base"),
            score_aggregator=ScoreAggregator(raw_agg) if raw_agg else None,
            verifier_id=data.get("verifier_id"),
            agent_name=data.get("agent_name"),
            trajectory_filter=TrajectoryFilter.from_dict(raw_tf) if raw_tf else None,
            max_thinking_tokens=data.get("max_thinking_tokens"),
            effort=data.get("effort"),
            use_agent_judge=data.get("use_agent_judge", True),
            use_trajectory=data.get("use_trajectory", True),
            judge_a2a_agent_id=data.get("judge_a2a_agent_id"),
            judge_sandbox_type=data.get("judge_sandbox_type"),
            judge_timeout_seconds=data.get("judge_timeout_seconds"),
            output_format=data.get("output_format", "rubric_binary"),
            grading_policy_prompt=data.get("grading_policy_prompt"),
            context_facts_for_judge=data.get("context_facts_for_judge"),
        )

    def _resolved_criteria(self, context: TaskStepContext) -> list[dict]:
        """Criteria for this run: a per-run ``step_params`` override replaces the stored list.

        Replace-on-presence, like ``load_artifact`` — an explicit ``[]`` replaces rather
        than reverting (presence check, not truthiness). Returns a copy; never mutates
        ``self``, which is shared across concurrent rollouts.
        """
        step_overrides = self.step_param_overrides(context)
        return list(step_overrides["criteria"]) if "criteria" in step_overrides else self.criteria

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        prompt_response = next((pr for pr in context.prompt_responses if pr.prompt_id == self.prompt_id), None)
        if prompt_response is None:
            raise RuntimeError(f"PromptResponse with prompt_id='{self.prompt_id}' not found in context")

        if prompt_response.error_type:
            logger.warning(f"Skipping verification '{self.verifier_id}': prompt had error_type={prompt_response.error_type}")
            self._write_skipped_entries(
                context, "prompt_error", f"Skipped: prompt had error_type={prompt_response.error_type}")
            return context

        per_turn_uris = [u for u in (prompt_response.target_agent_per_turn_trajectory_s3_uris or []) if u]
        if self.use_trajectory and not (per_turn_uris or prompt_response.agent_trajectory_s3_uri):
            raise RuntimeError(f"No trajectory available for prompt_id='{self.prompt_id}'")

        overrides = context.metadata.get("user_overrides", {})
        # Resolved into a local, not onto self: one step instance is shared across
        # concurrent rollouts, so mutating self would leak the override between them.
        use_agent_judge_override = overrides.get("use_agent_judge")
        use_agent_judge = use_agent_judge_override if use_agent_judge_override is not None else self.use_agent_judge
        criteria = self._resolved_criteria(context)
        # The judge sees/echoes short keys (c1..cN), not UUIDs; frame labels use them too, hence before compaction.
        criteria_for_judge, real_by_short = _to_short_keyed_criteria(criteria)
        apply_filter_override = overrides.get("apply_trajectory_filter")
        if apply_filter_override is False:
            filter_to_apply = None
        elif apply_filter_override is True:
            filter_to_apply = self.trajectory_filter or TrajectoryFilter()
        else:
            filter_to_apply = self.trajectory_filter

        # Multi-turn: judge every turn (fed from the per-turn URIs already on the
        # PromptResponse), not just the last. DEFAULT compaction runs per turn; SCREENSHOT
        # across turns isn't supported yet — fail fast rather than silently judging only the
        # last turn and producing misleading results.
        multiturn = self.use_trajectory and len(per_turn_uris) >= 2
        if multiturn and filter_to_apply is not None and filter_to_apply.compaction_type == CompactionType.SCREENSHOT:
            raise RuntimeError(
                f"Verifier '{self.verifier_id}': SCREENSHOT compaction is not supported on a "
                f"multi-turn run ({len(per_turn_uris)} turns) — it would silently judge only the "
                "last turn. Use the DEFAULT filter (compacted per turn) or no filter."
            )

        per_criterion_filter = _is_per_criterion(filter_to_apply)
        # Per-criterion frames are image attachments for a direct judge; the agent judge loads the raw file.
        if use_agent_judge and per_criterion_filter:
            raise RuntimeError(
                f"Verifier '{self.verifier_id}': frame_strategy=per_criterion only applies to the "
                "direct LLM judge (use_agent_judge=False)."
            )

        # Per-run dir so per-turn files can't collide with a prior run's on a reused (agent_name)
        # judge sandbox — a shorter run would otherwise leave stale later-turn files behind.
        multiturn_dir = f"{self.TRAJECTORY_CONTAINER_DIR}/{uuid.uuid4().hex[:12]}" if multiturn else None

        # Single-file compaction is only needed off the multi-turn path. It reads the trajectory,
        # and for the DEFAULT filter writes a compacted copy back, so it runs on a thread.
        compacted = _CompactedTrajectory.empty() if multiturn else await asyncio.to_thread(
            self._compact_trajectory, prompt_response, filter_to_apply, criteria=criteria,
            label_ids={real: short for short, real in real_by_short.items()})
        if not use_agent_judge:
            compacted.tool_result_files = []  # only an agent judge's container reads them; they can be large
        if self._spec().cites_evidence and not compacted.per_criterion:
            fmt = self.output_format.value
            if per_criterion_filter:
                # The selection found no screenshots (none recorded, or malformed): a fact about the run,
                # not the configuration — a skipped entry, as for a prompt error.
                logger.warning(
                    f"Skipping verification '{self.verifier_id}': output_format={fmt} needs "
                    "per-criterion frames, but the trajectory holds no screenshots."
                )
                self._write_skipped_entries(
                    context, "no_screenshots",
                    f"Skipped: {fmt} needs per-criterion frames, but the trajectory holds no screenshots")
                return context
            # The constructor checked the stored filter; a run-time override can still drop it, leaving a
            # prompt that demands `frame` / `evidence_status` with nothing to cite.
            raise RuntimeError(
                f"Verifier '{self.verifier_id}': output_format={fmt} needs per-criterion frames, "
                "but this run's trajectory filter attached none (user_overrides.apply_trajectory_filter=False?)."
            )

        auto_deployed_judge = None
        judge_agent = None
        judge_a2a_url: Optional[str] = None
        judge_sandbox_id: Optional[str] = None
        judge_agent_card: dict = {}
        if use_agent_judge:
            if self.agent_name is not None:
                agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
                if agent is None:
                    raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents")
                if not agent.sandbox_id:
                    raise RuntimeError(f"Agent '{self.agent_name}' has no sandbox_id")
                judge_a2a_url = agent.a2a_url or agent.api_url
                judge_sandbox_id = agent.sandbox_id
                judge_agent_card = agent.a2a_card or {}
                judge_agent = agent
            else:
                from agent_env.a2a_agent import A2AAgent
                from agent_env.config import get_config

                config = get_config()
                api_key = (
                    overrides.get("judge_litellm_api_key")
                    or overrides.get("litellm_api_key")
                    or config.get_litellm_api_key()
                )
                litellm_base_url = (
                    overrides.get("judge_litellm_base_url")
                    or overrides.get("litellm_base_url")
                    or config.get_litellm_base_url()
                )
                env_vars = {"LITELLM_API_KEY": api_key, "LITELLM_BASE_URL": litellm_base_url}
                resolved_judge_id = self.judge_a2a_agent_id or config.get_default_a2a_agent_id()
                a2a_agent = A2AAgent.get(resolved_judge_id)
                deploy_kwargs = {"env_vars": env_vars}
                resolved_sandbox_type = overrides.get("agent_sandbox") or self.judge_sandbox_type
                if resolved_sandbox_type:
                    deploy_kwargs["sandbox_type"] = resolved_sandbox_type
                auto_deployed_judge = await a2a_agent.deploy(**deploy_kwargs)
                judge_a2a_url = auto_deployed_judge.a2a_url
                judge_sandbox_id = auto_deployed_judge.sandbox_id
                judge_agent_card = auto_deployed_judge.agent_card
                judge_agent = auto_deployed_judge
                logger.info(f"Auto-deployed judge A2A agent '{resolved_judge_id}' at {judge_a2a_url}")
            if self.use_trajectory:
                from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_agent_sandbox_provider
                judge_sandbox_type = auto_deployed_judge.sandbox_type if auto_deployed_judge else (agent.sandbox_type if self.agent_name else None)
                provider = build_sandbox_provider(judge_sandbox_type) if judge_sandbox_type else get_agent_sandbox_provider()
                sandbox = await provider.get_sandbox(judge_sandbox_id)
                if multiturn:
                    await self._load_per_turn_trajectories(sandbox, per_turn_uris, multiturn_dir, filter_to_apply)
                else:
                    await sandbox.write_file_from_s3(compacted.container_s3_uri, self.TRAJECTORY_CONTAINER_PATH)
                    if compacted.tool_result_files:
                        await self._load_tool_result_files(sandbox, compacted.tool_result_files)
                        compacted.tool_result_files = []

        try:
            model = overrides.get("judge_model") or self.default_model
            criteria_for_prompt = [
                {**c, "negative": True} if (c.get("weight") is not None and c["weight"] < 0) else c
                for c in criteria_for_judge
            ]
            criteria_json = json.dumps(criteria_for_prompt, indent=2)
            agent_prompt_text = prompt_response.prompt_text or ""
            judge_trajectory_s3_uri: Optional[str] = None
            judge_effort = overrides.get("judge_effort") or self.effort
            judge_max_thinking_tokens = overrides.get("judge_max_thinking_tokens") or self.max_thinking_tokens
            loaded_for_judge: list[dict] = []
            if use_agent_judge and self.agent_name is not None:
                loaded_for_judge = [
                    e for e in (context.metadata.get("loaded_file_artifact_universes") or [])
                    if e.get("agent_name") == self.agent_name
                ]
            context_facts = (resolve_context_facts(self.context_facts_for_judge, context.metadata)
                             if self.context_facts_for_judge else [])
            image_blocks: list[dict] = []
            if use_agent_judge:
                # Agent judge can't use inline screenshot frames; it loads the raw file.
                if filter_to_apply and filter_to_apply.compaction_type == CompactionType.SCREENSHOT:
                    logger.warning(
                        f"compaction_type=screenshot only applies on the direct-LLM judge; "
                        f"verifier '{self.verifier_id}' uses an agent judge — raw trajectory "
                        "passed through unchanged."
                    )
                trajectory_path = None
                if self.use_trajectory:
                    trajectory_path = multiturn_dir if multiturn else self.TRAJECTORY_CONTAINER_PATH
                eval_prompt = self._build_eval_prompt(
                    agent_prompt=agent_prompt_text,
                    agent_response=prompt_response.response,
                    criteria_json=criteria_json,
                    trajectory_path=trajectory_path,
                    loaded_artifacts=loaded_for_judge,
                    context_facts=context_facts,
                )
                if self.use_trajectory and multiturn:
                    eval_prompt += self._MULTITURN_DIR_NOTE.format(n=len(per_turn_uris))
            else:
                image_blocks = compacted.image_blocks
                # Direct judge embeds the trajectory inline; DEFAULT has no pre-rendered text.
                if multiturn:
                    trajectory_inline = await asyncio.to_thread(self._merge_per_turn_text, per_turn_uris, filter_to_apply)
                else:
                    trajectory_inline = compacted.inline_text
                    if trajectory_inline is None and compacted.container_s3_uri:
                        trajectory_inline = await asyncio.to_thread(self._read_trajectory_text, compacted.container_s3_uri)
                eval_prompt = self._build_eval_prompt(
                    agent_prompt=agent_prompt_text,
                    agent_response=prompt_response.response,
                    criteria_json=criteria_json,
                    trajectory_inline=trajectory_inline,
                    image_blocks=image_blocks,
                    per_criterion_frames=compacted.per_criterion,
                    context_facts=context_facts,
                )
                if image_blocks:
                    logger.info(
                        f"Verifier '{self.verifier_id}': direct judge — {_image_count(image_blocks)} "
                        f"image(s){' chosen per criterion' if compacted.per_criterion else ''}, "
                        f"eval prompt {len(eval_prompt)} chars (~{len(eval_prompt) // 4} tokens)."
                    )

            verification_results, judge_output_retries, judge_output_discrepancies, judge_trajectory_s3_uri = (
                await self._run_judge_with_output_retries(
                    eval_prompt=eval_prompt,
                    context=context,
                    model=model,
                    judge_a2a_url=judge_a2a_url,
                    judge_agent_card=judge_agent_card,
                    judge_agent=judge_agent,
                    judge_effort=judge_effort,
                    judge_max_thinking_tokens=judge_max_thinking_tokens,
                    use_agent_judge=use_agent_judge,
                    criteria=criteria_for_judge,
                    image_blocks=image_blocks,
                )
            )
            # Rows come back keyed by c1..cN; restore the real criterion ids, in the cited frame labels too.
            _restore_criterion_ids(verification_results, real_by_short)
            if isinstance(verification_results, ResultRows):
                _restore_frame_labels(verification_results, real_by_short)
                _restore_frame_labels(verification_results.checks, real_by_short)
        finally:
            if auto_deployed_judge is not None:
                from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_agent_sandbox_provider
                try:
                    judge_type = auto_deployed_judge.sandbox_type
                    provider = build_sandbox_provider(judge_type) if judge_type else get_agent_sandbox_provider()
                    sandbox = await provider.get_sandbox(auto_deployed_judge.sandbox_id)
                    await sandbox.terminate()
                    logger.info(f"Terminated auto-deployed judge sandbox {auto_deployed_judge.sandbox_id}")
                except Exception as e:
                    logger.warning(f"Failed to terminate auto-deployed judge sandbox: {e}")

        judge_call_fields: dict = {
            "compact_trajectory_s3_uri": compacted.compact_s3_uri,
            "judge_trajectory_s3_uri": judge_trajectory_s3_uri,
        }
        if judge_output_retries:
            judge_call_fields["judge_output_retries"] = judge_output_retries
        if judge_output_discrepancies:
            judge_call_fields["judge_output_discrepancies"] = judge_output_discrepancies
        self._write_entries(context, verification_results, judge_call_fields)
        return context

    def _write_entries(self, context: TaskStepContext, verification_results: list[dict], judge_call_fields: dict) -> None:
        """Store the rubric entry under ``verifier_id`` and, for rubric_evidence_with_mistakes, the
        trajectory-mistakes entry under ``trajectory_mistakes_key(verifier_id)``; ``judge_call_fields`` (URIs,
        retries) go on both. The mistake rows are built before either entry is written, so a response they
        cannot be derived from leaves neither behind."""
        score = aggregate_score(verification_results, self.score_aggregator)
        mistake_rows: Optional[list[dict]] = None
        if self._writes_mistakes_entry:
            if not isinstance(verification_results, ResultRows):
                raise RuntimeError(
                    f"Verifier '{self.verifier_id}': the judge response carries no mistake findings "
                    f"({type(verification_results).__name__})")
            mistake_rows = trajectory_mistakes_rows_from_evidence(verification_results)
        verifications = context.metadata.setdefault("verifications", {})
        verification_entry = {
            **self._entry_format_fields(),
            "results": list(verification_results),
            "score": score,
            **judge_call_fields,
        }
        if isinstance(verification_results, ResultRows):
            # rubric_evidence: the judge's `reasoning` and policy `checks` ride on the entry, not the rows.
            verification_entry["reasoning"] = verification_results.reasoning
            verification_entry["checks"] = verification_results.checks
        verifications[self.verifier_id] = verification_entry
        logger.info(f"Verification '{self.verifier_id}' complete: {len(verification_results)} criteria, score={score}")

        if mistake_rows is not None:
            mistakes_score = aggregate_score(mistake_rows, ScoreAggregator.WEIGHTED_AVERAGE)
            mistakes_key = trajectory_mistakes_key(self.verifier_id)
            verifications[mistakes_key] = {
                **self._mistakes_entry_fields(),
                "results": mistake_rows,
                "score": mistakes_score,
                **judge_call_fields,
            }
            logger.info(
                f"Verification '{mistakes_key}' complete (from '{self.verifier_id}'): "
                f"{len(mistake_rows)} rows, score={mistakes_score}")

    def _compact_trajectory(
        self,
        prompt_response: "PromptResponse",
        filter_to_apply: Optional[TrajectoryFilter],
        criteria: Optional[list[dict]] = None,
        label_ids: Optional[dict[str, str]] = None,
    ) -> _CompactedTrajectory:
        """Compact per the filter and return every representation; judge-agnostic — the
        caller picks which one its judge path consumes. ``criteria`` / ``label_ids`` (criterion id ->
        the key the judge sees) only matter to per-criterion frame selection."""
        if not self.use_trajectory:
            return _CompactedTrajectory.empty()

        # Fall back to the last per-turn URI when agent_trajectory_s3_uri is unset (the value
        # it normally holds). The execute() guard guarantees one source exists; the raise only narrows the type.
        raw_uri = prompt_response.agent_trajectory_s3_uri or next(
            (u for u in reversed(prompt_response.target_agent_per_turn_trajectory_s3_uris or []) if u),
            None,
        )
        if raw_uri is None:
            raise RuntimeError(f"No trajectory available for prompt_id='{self.prompt_id}'")
        ctype = filter_to_apply.compaction_type if filter_to_apply else CompactionType.DEFAULT

        if filter_to_apply is not None and ctype == CompactionType.SCREENSHOT:
            raw_text = self._read_trajectory_text(raw_uri)
            return self._compact_screenshot(raw_text, raw_uri, filter_to_apply, criteria or [], label_ids or {})

        s3_uri, tool_files = raw_uri, []
        compact_s3_uri: Optional[str] = None
        if filter_to_apply and ctype == CompactionType.DEFAULT:
            s3_uri, tool_files = self._filter_trajectory(raw_uri, filter_to_apply)
            # Return the URI; never write it back onto prompt_response. It's an appended
            # item in context.prompt_responses ($addToSet, immutable) — mutating it re-adds
            # a duplicate (hub renders the trajectory twice). Recorded on the result instead.
            compact_s3_uri = s3_uri
        return _CompactedTrajectory.for_default(s3_uri, tool_files, compact_s3_uri)

    def _compact_screenshot(
        self, raw_text: str, raw_uri: str, tf: TrajectoryFilter, criteria: list[dict],
        label_ids: dict[str, str],
    ) -> _CompactedTrajectory:
        """SCREENSHOT compaction for the direct judge: the frames ``frame_strategy`` picks and the text
        ``trajectory_text`` selects."""
        per_criterion = tf.frame_strategy == FrameStrategy.PER_CRITERION
        action_log_text = tf.trajectory_text == TrajectoryText.ACTION_LOG
        transcript = transcript_from_raw(raw_text) if per_criterion or action_log_text else Transcript([], [])
        attached: list[str] = []    # frame payloads the FULL text flags as attached (final frames only)
        if per_criterion:
            # The regex knobs are matched case-insensitively (see TrajectoryFilter).
            exclude = re.compile(tf.evidence_exclude_pattern, re.I) if tf.evidence_exclude_pattern else None
            always_show = {label: re.compile(pattern, re.I) for label, pattern in (tf.always_show_actions or {}).items()}
            labeled = select_frames_per_criterion(transcript.actions, criteria, max_frames=tf.frame_budget,
                                                  exclude_pattern=exclude, always_show=always_show, label_ids=label_ids)
            image_blocks = _labeled_image_blocks(labeled)
        else:
            final = final_image_frames(raw_text, tf.screenshot_last_n)
            image_blocks = [_image_url_block(f) for f in final]
            attached = [f.data for f in final]
        if action_log_text:
            inline_text = action_log(transcript.actions, messages=transcript.messages, max_lines=tf.action_log_max_lines)
        else:
            inline_text = strip_trajectory_images(raw_text, attached=attached)
        # A trajectory with no screenshots selects nothing: the judge gets the text alone, with no frame rules.
        return _CompactedTrajectory.for_screenshot(inline_text, image_blocks, container_s3_uri=raw_uri,
                                                   per_criterion=per_criterion and bool(image_blocks))

    def _filter_trajectory(self, s3_uri: str, trajectory_filter: TrajectoryFilter) -> tuple[str, list[tuple[str, str]]]:
        """Download trajectory from S3, apply compact filter, re-upload as a compact version.

        Returns:
            (compact_s3_uri, tool_result_files) where tool_result_files is a list of
            (container_path, content_json) pairs for externalized tool results.
        """
        from agent_env.config import get_config

        config = get_config()
        object_store = config.get_object_store()
        spans = json.loads(object_store.get(s3_uri))
        filtered, tool_result_files = compact_otel_trajectory(spans, trajectory_filter)
        compact_key = f"{config.get_artifact_key_prefix()}compacted-trajectories/{uuid.uuid4().hex}.json"
        object_url = object_store.put(compact_key, json.dumps(filtered).encode(), content_type="application/json")
        return object_url, tool_result_files

    async def _load_tool_result_files(self, sandbox, tool_result_files: list[tuple[str, str]]) -> None:
        """Write externalized tool result files into the agent container."""
        for container_path, content in tool_result_files:
            await sandbox.write_file_from_text(content, container_path)
        logger.info(f"Loaded {len(tool_result_files)} tool result files into container")

    async def _load_trajectory_into_container(self, sandbox_id: str, s3_uri: str) -> None:
        """Load trajectory from S3 into an existing agent's container."""
        from agent_env.providers.sandbox_providers.sandbox_provider import get_agent_sandbox_provider
        sandbox = await get_agent_sandbox_provider().get_sandbox(sandbox_id)
        await sandbox.write_file_from_s3(s3_uri, self.TRAJECTORY_CONTAINER_PATH)

    def _read_trajectory_text(self, s3_uri: str) -> str:
        from agent_env.config import get_config
        return get_config().get_object_store().get(s3_uri).decode("utf-8", errors="replace")

    async def _load_per_turn_trajectories(
        self, sandbox, per_turn_uris: list[str], container_dir: str,
        trajectory_filter: Optional[TrajectoryFilter] = None,
    ) -> None:
        """Load each turn into the judge container as turn_NN.json under ``container_dir``.

        No filter → direct S3 pull per turn. DEFAULT filter → compact each turn in memory and
        write as text, its externalized tool-result files namespaced ``turn_NN_`` to avoid
        collisions, then loaded into the sandbox. The write methods create ``container_dir``
        inside the container, so no separate mkdir is needed.
        """
        tool_result_files: list[tuple[str, str]] = []
        for i, uri in enumerate(per_turn_uris, start=1):
            dest = f"{container_dir}/turn_{i:02d}.json"
            if trajectory_filter is not None:
                content, turn_files = await asyncio.to_thread(
                    self._filter_trajectory_content, uri, trajectory_filter, f"turn_{i:02d}_"
                )
                await sandbox.write_file_from_text(content, dest)
                tool_result_files.extend(turn_files)
            else:
                await sandbox.write_file_from_s3(uri, dest)
        if tool_result_files:
            await self._load_tool_result_files(sandbox, tool_result_files)
        logger.info(
            f"Loaded {len(per_turn_uris)} per-turn trajectories into {container_dir}"
            f"{' (filtered)' if trajectory_filter else ''}, {len(tool_result_files)} tool-result files"
        )

    def _merge_per_turn_text(
        self, per_turn_uris: list[str], trajectory_filter: Optional[TrajectoryFilter] = None,
    ) -> str:
        """Concatenate every turn's trajectory text, turn-marked, for the direct-LLM judge
        (embedded inline; in-memory, no S3 write). A DEFAULT filter compacts each turn; its
        externalized files are dropped, as on the single-turn direct-judge path (no container)."""
        parts = []
        for i, uri in enumerate(per_turn_uris, start=1):
            if trajectory_filter is not None:
                text, _ = self._filter_trajectory_content(uri, trajectory_filter, f"turn_{i:02d}_")
            else:
                text = self._read_trajectory_text(uri)
            parts.append(f"===== Turn {i} =====\n{text}")
        return "\n\n".join(parts)

    def _filter_trajectory_content(
        self, s3_uri: str, trajectory_filter: TrajectoryFilter, result_file_prefix: str = "",
    ) -> tuple[str, list[tuple[str, str]]]:
        """DEFAULT-compact one trajectory in memory (no S3 re-upload), returning
        (compacted_json_text, tool_result_files). ``result_file_prefix`` namespaces the
        externalized filenames so per-turn files don't collide. In-memory sibling of
        ``_filter_trajectory`` (which re-uploads to S3 for the single-file path)."""
        spans = json.loads(self._read_trajectory_text(s3_uri))
        filtered, tool_result_files = compact_otel_trajectory(
            spans, trajectory_filter, result_file_prefix=result_file_prefix,
        )
        return json.dumps(filtered), tool_result_files

    async def _prompt_llm_judge(
        self,
        eval_prompt: str,
        *,
        model: str,
        context: TaskStepContext,
        image_blocks: Optional[list[dict]] = None,
    ) -> dict:
        """Direct LiteLLM judge — no agent boot, no tools. Returns {response, trajectory_s3_uri}.

        When ``image_blocks`` are provided the user message becomes multimodal
        (text + images); otherwise it stays a plain string, identical to before."""
        import litellm

        from agent_env.config import get_config

        overrides = context.metadata.get("user_overrides", {})
        config = get_config()
        call_config = config.resolve_model_call(
            model,
            default_api_key=overrides.get("judge_litellm_api_key") or overrides.get("litellm_api_key"),
            base_override=overrides.get("judge_litellm_base_url")
            or overrides.get("litellm_base_url")
            or self.default_model_api_base,
        )
        user_content = (
            [{"type": "text", "text": eval_prompt}, *image_blocks]
            if image_blocks
            else eval_prompt
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "rubrics_verification",
                "schema": self._spec().output_format["schema"],
                "strict": True,
            },
        }
        # Some endpoints reject json_schema with images. Fall back once on that exact
        # error shape; malformed JSON is handled by the outer output-retry loop.
        use_response_format = True

        last_exc: Exception | None = None
        for attempt in range(1, self.DEFAULT_MAX_RETRIES + 1):
            try:
                request_kwargs: dict = dict(
                    model=model,
                    messages=[{"role": "user", "content": user_content}],
                    timeout=self.judge_timeout_seconds,
                    **call_config.client_kwargs(),
                )
                if use_response_format:
                    request_kwargs[ModelParam.RESPONSE_FORMAT] = response_format
                response = await litellm.acompletion(**request_kwargs)
                content = response.choices[0].message.content or ""
                return {"response": content, "trajectory_s3_uri": None}
            except Exception as exc:
                last_exc = exc
                if image_blocks and use_response_format and _looks_like_response_format_error(exc):
                    logger.warning(
                        f"Multimodal judge call with json_schema failed for verifier "
                        f"'{self.verifier_id}' ({type(exc).__name__}: {exc}); retrying "
                        f"WITHOUT response_format (text-completion + parse)."
                    )
                    use_response_format = False
                    continue
                if attempt < self.DEFAULT_MAX_RETRIES:
                    logger.warning(f"LLM judge attempt {attempt}/{self.DEFAULT_MAX_RETRIES} failed ({type(exc).__name__}: {exc}), retrying...")
                else:
                    logger.error(f"LLM judge failed after {self.DEFAULT_MAX_RETRIES} attempts ({type(exc).__name__}: {exc})")
        raise RuntimeError(f"LLM judge failed after {self.DEFAULT_MAX_RETRIES} attempts") from last_exc

    async def _configure_judge_a2a(
        self,
        *,
        judge_a2a_url: str,
        judge_agent_card: dict,
        model: str,
        judge_effort: Optional[str],
        judge_max_thinking_tokens: Optional[int],
        context: TaskStepContext,
    ) -> None:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.a2a_agent import protocol

        desired_config: dict = {
            "model": model,
            "effort": judge_effort,
            "max_thinking_tokens": judge_max_thinking_tokens,
            "output_format": self._spec().output_format,
            "timeout_seconds": self.judge_timeout_seconds,
            "task_id": context.metadata.get("task_id"),
        }
        desired_config = {k: v for k, v in desired_config.items() if v is not None}
        config_ext = A2AAgent.find_extension(judge_agent_card, A2AAgent.EXT_AGENT_CONFIG)
        if config_ext and desired_config:
            supported = (config_ext.get("params") or {}).get("methods", {}).get("set", {}).get("request", {}).get("supported", [])
            config_payload = {k: v for k, v in desired_config.items() if k in supported}
            if config_payload:
                endpoint = (config_ext.get("params") or {}).get("endpoint", "/ext/agent-config")
                await protocol.post_agent_config(judge_a2a_url + endpoint, config_payload)
                logger.info(f"Set judge agent config: {list(config_payload.keys())}")

    async def _invoke_judge_a2a(
        self,
        *,
        eval_prompt: str,
        judge_a2a_url: str,
        judge_agent_card: dict,
        judge_agent,
    ) -> dict:
        """Send one evaluation prompt to an A2A judge and return its response."""
        from agent_env.a2a_agent import protocol

        task_id, _ = await protocol.send_a2a_message(
            judge_a2a_url,
            parts=[{"kind": "text", "text": eval_prompt}],
            message_id=uuid.uuid4().hex,
            context_id=None,
            timeout_seconds=self.judge_timeout_seconds,
        )
        result = await protocol.poll_a2a_task(
            judge_a2a_url,
            task_id,
            timeout_seconds=self.judge_timeout_seconds,
            poll_interval_seconds=self.DEFAULT_POLL_INTERVAL_SECONDS,
        )
        state = result["status"]["state"]
        status_msg = (result.get("status") or {}).get("message") or {}
        tr = protocol.TerminalResponse.from_message(status_msg)
        if state == "failed":
            sandbox_id = getattr(judge_agent, "sandbox_id", None)
            logger.error(
                "Judge A2A task %s failed (verifier=%s sandbox=%s error_type=%s exception=%s). "
                "Raw A2A status message: %s",
                task_id, self.verifier_id, sandbox_id,
                tr.error_type, tr.error_class, json.dumps(status_msg, default=str)[:4000],
            )
            detail = ", ".join(filter(None, [
                f"error_type={tr.error_type}" if tr.error_type else None,
                f"exception={tr.error_class}" if tr.error_class else None,
                f"task_id={task_id}",
                f"judge_sandbox={sandbox_id}" if sandbox_id else None,
            ]))
            body = agent_error_text(tr.error_message, tr.response_text)
            # Fetched here, inside execute()'s try, so it runs before the finally
            # terminates an auto-deployed judge sandbox.
            container_logs = await fetch_container_logs(judge_agent)
            if container_logs:
                logger.error(
                    "Judge container logs for failed A2A task %s (verifier=%s):\n%s",
                    task_id, self.verifier_id, container_logs,
                )
                body = f"{body}\n{container_logs}" if body else container_logs
            raise RuntimeError(f"Judge A2A task failed ({detail}): {body}")
        judge_trajectory_s3_uri = await self._fetch_judge_trajectory(
            judge_a2a_url=judge_a2a_url,
            judge_agent_card=judge_agent_card,
            a2a_server_task_id=task_id,
            sandbox_type=getattr(judge_agent, "sandbox_type", None),
        )
        return {"response": tr.response_text, "trajectory_s3_uri": judge_trajectory_s3_uri}

    async def _fetch_judge_trajectory(
        self,
        *,
        judge_a2a_url: str,
        judge_agent_card: dict,
        a2a_server_task_id: str,
        sandbox_type: Optional[str],
    ) -> Optional[str]:
        """Best-effort capture of the judge's own reasoning trajectory, via the same
        EXT_TRAJECTORY extension `prompt_agent` uses, keyed by the A2A task id. Retries
        the fetch up to DEFAULT_MAX_RETRIES times; never raises — a fetch failure is a
        log line, not a broken run."""
        from agent_env.a2a_agent import A2AAgent

        traj_ext = A2AAgent.find_extension(judge_agent_card, A2AAgent.EXT_TRAJECTORY)
        if not traj_ext:
            return None
        if not (traj_ext.get("params") or {}).get("endpoint"):
            # Don't guess a path — the extension is self-describing precisely so callers
            # never have to assume where it lives (it may differ by judge/version).
            logger.warning(
                "Verifier '%s': judge's EXT_TRAJECTORY extension has no params.endpoint — "
                "skipping trajectory capture", self.verifier_id,
            )
            return None
        get_method, get_path = A2AAgent.operation(traj_ext, "get")
        config = get_config()
        store = config.get_object_store()
        prefix = store.object_url(f"{config.get_artifact_key_prefix()}judge_trajectories/verifier_id={self.verifier_id}/")
        mode = trajectory_mode(get_method, store, by="task_id", sandbox_type=sandbox_type)
        if mode is None:
            logger.warning(
                "Verifier '%s': judge advertises no trajectory get form this object store "
                "can serve — skipping trajectory capture", self.verifier_id,
            )
            return None
        # Named per call rather than by the judge's own task id, which a judge could reuse
        # to overwrite another run's trajectory.
        upload = None
        if mode == "objects":
            try:
                upload = await asyncio.to_thread(TrajectoryUpload.to, store, trajectory_object_url(prefix, store=store))
            except Exception as exc:
                logger.warning(
                    "Verifier '%s': judge trajectory grant unavailable (continuing without it): %s",
                    self.verifier_id, exc,
                )
                return None
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.DEFAULT_MAX_RETRIES + 1):
            try:
                fetched = await fetch_trajectory(
                    judge_a2a_url + get_path, {"task_id": a2a_server_task_id}, upload=upload
                )
                return await asyncio.to_thread(store_trajectory, fetched, prefix)
            except Exception as exc:
                last_exc = exc
                if attempt < self.DEFAULT_MAX_RETRIES:
                    logger.warning(
                        "Verifier '%s': judge trajectory fetch attempt %d/%d failed (%s), retrying...",
                        self.verifier_id, attempt, self.DEFAULT_MAX_RETRIES, exc,
                    )
        logger.warning(
            "Verifier '%s': judge trajectory fetch failed after %d attempts (continuing without it): %s",
            self.verifier_id, self.DEFAULT_MAX_RETRIES, last_exc,
        )
        return None

    async def _run_judge_with_output_retries(
        self,
        *,
        eval_prompt: str,
        context: TaskStepContext,
        model: str,
        judge_a2a_url: Optional[str],
        judge_agent_card: dict,
        judge_agent,
        judge_effort: Optional[str],
        judge_max_thinking_tokens: Optional[int],
        use_agent_judge: bool,
        criteria: list[dict],
        image_blocks: Optional[list[dict]] = None,
    ) -> tuple[list[dict], int, list[dict], Optional[str]]:
        """Invoke the judge, retrying with corrective feedback when output ids/count are wrong.

        ``use_agent_judge`` is the run-resolved value (passed in, not read off self — see execute()).
        """
        spec = self._spec()
        prompt = eval_prompt
        discrepancy_log: list[dict] = []
        judge_trajectory_s3_uri: Optional[str] = None

        if use_agent_judge:
            if not judge_a2a_url:
                raise RuntimeError("Judge A2A URL is required when use_agent_judge=True")
            await self._configure_judge_a2a(
                judge_a2a_url=judge_a2a_url,
                judge_agent_card=judge_agent_card,
                model=model,
                judge_effort=judge_effort,
                judge_max_thinking_tokens=judge_max_thinking_tokens,
                context=context,
            )

        for attempt in range(1, self.DEFAULT_MAX_RETRIES + 1):
            if use_agent_judge:
                data = await self._invoke_judge_a2a(
                    eval_prompt=prompt,
                    judge_a2a_url=judge_a2a_url,
                    judge_agent_card=judge_agent_card,
                    judge_agent=judge_agent,
                )
            elif image_blocks:
                data = await self._prompt_llm_judge(
                    prompt, model=model, context=context, image_blocks=image_blocks,
                )
            else:
                data = await self._prompt_llm_judge(prompt, model=model, context=context)

            logger.info(f"Judge response: {data['response'][:200]}...")
            judge_trajectory_s3_uri = data.get("trajectory_s3_uri")

            results, discrepancy = spec.diagnose_response(data["response"], criteria)
            if discrepancy is None:
                assert results is not None
                return results, attempt - 1, discrepancy_log, judge_trajectory_s3_uri

            logger.warning(
                "Judge rubric output discrepancy (attempt %d/%d, verifier_id=%s): %s",
                attempt,
                self.DEFAULT_MAX_RETRIES,
                self.verifier_id,
                discrepancy,
            )
            discrepancy_log.append(discrepancy.to_dict())
            if attempt == self.DEFAULT_MAX_RETRIES:
                raise ValueError(
                    f"Judge returned {discrepancy.returned_count} results but expected "
                    f"{discrepancy.expected_count} criteria"
                )

            prompt = spec.format_correction_prompt(
                eval_prompt=eval_prompt,
                previous_response=data["response"],
                discrepancy=discrepancy,
            )

        raise RuntimeError("Judge output retry loop exited unexpectedly")

    def _build_eval_prompt(
        self,
        *,
        agent_prompt: str,
        agent_response: str,
        criteria_json: str,
        trajectory_path: Optional[str] = None,
        trajectory_inline: Optional[str] = None,
        loaded_artifacts: Optional[list[dict]] = None,
        image_blocks: Optional[list[dict]] = None,
        per_criterion_frames: bool = False,
        context_facts: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """``context_facts`` are rendered ``(label, value)`` pairs (``resolve_context_facts``); ``image_blocks``
        are the blocks attached to the judge message, labels included when ``per_criterion_frames``."""
        spec = self._spec()
        # Only the rubric_evidence templates have this placeholder; the others ignore the argument.
        checks_instruction = evidence_checks_instruction(has_policy=bool(self.grading_policy_prompt))
        if trajectory_path is not None:
            base = spec.prompt_template_path.substitute(
                agent_prompt=agent_prompt,
                agent_response=agent_response,
                criteria_json=criteria_json,
                trajectory_path=trajectory_path,
                checks_instruction=checks_instruction,
            )
        elif trajectory_inline is not None:
            base = spec.prompt_template_inline.substitute(
                agent_prompt=agent_prompt,
                agent_response=agent_response,
                criteria_json=criteria_json,
                trajectory_content=trajectory_inline,
                checks_instruction=checks_instruction,
            )
        else:
            base = spec.prompt_template_no_trajectory.substitute(
                agent_prompt=agent_prompt,
                agent_response=agent_response,
                criteria_json=criteria_json,
                checks_instruction=checks_instruction,
            )
        if loaded_artifacts:
            base += "\n\n" + self._build_loaded_artifacts_block(loaded_artifacts)
        if context_facts:
            # Facts before the policy: they are inputs, the policy is how to read them.
            base += "\n\n## Context facts\n" + self._CONTEXT_FACTS_LEAD + "\n" + "\n".join(
                f"- {label}: {value}" for label, value in context_facts)
        if self.grading_policy_prompt:
            base += "\n\n## Grading policy\n" + self.grading_policy_prompt.strip()
        if image_blocks:
            # Images are attached to the judge message separately; tell it how to use them.
            n = _image_count(image_blocks)
            if per_criterion_frames:
                base += per_criterion_grounding(n, cites_evidence=spec.cites_evidence)
            else:
                base += SCREENSHOT_GROUNDING.format(n=n)
        return base

    @staticmethod
    def _build_loaded_artifacts_block(entries: list[dict]) -> str:
        lines = [
            "## Files Available For Inspection",
            "",
            "The following file artifacts have been pre-loaded into your container so you can verify the agent's claims about files:",
            "",
        ]
        for e in entries:
            dest = (e.get("destination_path") or "/tmp/file_artifacts").rstrip("/")
            artifact_id = e.get("id") or "<unknown>"
            raw_files = e.get("files") or []
            file_names = list(raw_files.keys()) if isinstance(raw_files, dict) else list(raw_files)
            if file_names:
                preview = ", ".join(file_names[:8])
                if len(file_names) > 8:
                    preview += f", ... ({len(file_names)} files total)"
                lines.append(f"- `{artifact_id}` at `{dest}/` — contains: {preview}")
            else:
                lines.append(f"- `{artifact_id}` at `{dest}/`")
        lines += [
            "",
            "When the criteria reference files the agent produced, claimed to produce, or operates on, "
            "inspect these locations using Bash (`ls`, `cat`, `find`, `grep`, `wc -c`, `file`) or Read tools. "
            "The agent may have written files in nested subdirectories — use `find` to search recursively. "
            "Do not rely solely on the agent's response text — verify against the actual files when possible.",
        ]
        return "\n".join(lines)
