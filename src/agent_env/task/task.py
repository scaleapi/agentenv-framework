"""Base Task model for AgentEnv."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import itertools
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

from agent_env.plugins import _registration
from agent_env.store.routing import run_scope
from agent_env.task_step.context import TaskStepContext, regraft_redacted_keys
from agent_env.task_step.context_ops import ContextUpdateOps, build_context_update_ops
from agent_env.task_step.task_step import RetryConfig, TaskStep

if TYPE_CHECKING:
    from agent_env.task.store import TaskInstance, TaskQuery


logger = logging.getLogger(__name__)


def _identity(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, default=str)


def _rolled_back_sandboxes(before: dict, after: dict) -> dict:
    """The sandbox coordinates of every deployed env / agent / sandbox present in the stored
    context ``before`` an undo and absent ``after`` it, shaped like the context's own slices
    so the executor's cleanup can scan the marker exactly as it scans a context. The undo is
    what removes these from the live and persisted context, so without this record an
    orphaned sandbox from the failed attempt is discoverable nowhere.
    """
    def gone(key: str) -> list[dict]:
        kept = {_identity(e) for e in after.get(key) or []}
        return [e for e in before.get(key) or [] if _identity(e) not in kept]

    return {
        "deployed_envs": [
            {"sandbox_id": e.get("sandbox_id"), "sandbox_type": e.get("sandbox_type"),
             "sandbox_ids": e.get("sandbox_ids") or {}}
            for e in gone("deployed_envs")
        ],
        "deployed_agents": [
            {"sandbox_id": a.get("sandbox_id"), "sandbox_type": a.get("sandbox_type")}
            for a in gone("deployed_agents") if a.get("sandbox_id")
        ],
        "deployed_sandboxes": [
            {"sandbox_id": x.get("sandbox_id"), "sandbox_type": x.get("sandbox_type")}
            for x in gone("deployed_sandboxes")
        ],
    }


def _rebuild_context(context: TaskStepContext, stored: dict) -> None:
    """Replace ``context``'s fields in place from the context the store rebuilt. In place,
    because the scheduler and every step hold this object. The stored document never holds
    secrets (``_REDACTED_KEYS``), so they are grafted back from the live context first."""
    stored = copy.deepcopy(stored)
    regraft_redacted_keys({"metadata": context.metadata}, stored)
    rebuilt = TaskStepContext.from_dict(stored)
    for f in dataclasses.fields(TaskStepContext):
        setattr(context, f.name, getattr(rebuilt, f.name))


def _overlapping_survivors(
    rollback_ids: set[str], completed_ids: set[str], windows: dict[str, tuple[int, int]],
) -> list[str]:
    """Completed steps outside the span whose execution window overlapped a span member's.

    Until per-step isolation lands, a step's journaled diff is taken against the
    shared in-memory context, so a step that ran while a span member ran may have journaled
    the member's writes; replaying it would rebuild the span's world dirty. Windows are
    (dispatch tick, end tick); a step never dispatched has none and cannot overlap.
    """
    members = [windows[x] for x in rollback_ids if x in windows]
    return sorted(
        s for s in completed_ids - rollback_ids
        if s in windows and any(windows[s][0] < end and start < windows[s][1] for start, end in members)
    )


def substitute_seed_in_steps(steps: list["TaskStep"], seed: dict) -> list["TaskStep"]:
    """Round-trip each step through its own to_dict/from_dict with ``<key>``
    seed placeholders replaced in every string leaf. Returns a NEW list, never
    mutating the input; idempotent (an already-substituted string has no tokens
    left). Used by Task.run(); a retry re-dispatches the same substituted step
    objects, so it executes exactly what the original run executed."""
    if not seed:
        return list(steps)
    from agent_env.task_step.registry import get_task_step_registry

    def _sub(value):
        if isinstance(value, str):
            for k, v in seed.items():
                value = value.replace(f"<{k}>", str(v))
            return value
        if isinstance(value, list):
            return [_sub(x) for x in value]
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        return value

    from agent_env.task_step.task_step import attach_retry_config

    registry = get_task_step_registry()
    out: list["TaskStep"] = []
    for step in steps:
        data = _sub(step.to_dict())
        out.append(attach_retry_config(registry[step.type].from_dict(data), data))
    return out


def _dependency_ids(steps: list["TaskStep"]) -> dict[str, set[str]]:
    """Direct dependency ids per step; depends_on None = all prior (scheduler semantics)."""
    deps: dict[str, set[str]] = {}
    for i, step in enumerate(steps):
        if step.depends_on is None:
            deps[step.id] = {prior.id for prior in steps[:i]}
        else:
            deps[step.id] = {d.task_step_id for d in step.depends_on}
    return deps


def _ancestor_ids(step_id: str, deps: dict[str, set[str]]) -> set[str]:
    """Transitive dependency ancestry of ``step_id`` under ``deps``."""
    seen: set[str] = set()
    frontier = list(deps.get(step_id, ()))
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(deps.get(current, ()))
    return seen


def _dependents_closure(root: str, dependents: dict[str, list[str]]) -> set[str]:
    """``root`` plus every step that transitively depends on it.

    The rollback span: rebuilding the world from ``root`` invalidates everything
    built on top of it, not just the failed step's own ancestry. A step that
    depends on a span member but is not an ancestor of the failed step would
    otherwise be left in ``completed`` while its inputs are torn down, so the
    retry would run against a half-built world.
    """
    seen: set[str] = {root}
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for nxt in dependents.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


async def _roll_back_span(
    context: TaskStepContext,
    instance_id: str | None,
    steps: list["TaskStep"],
    dependents: dict[str, list[str]],
    failed_id: str,
    resume_from: str,
    cancelled_ids: set[str],
    attempt: int,
    windows: dict[str, tuple[int, int]],
    completed_ids: set[str],
    log_prefix: str,
) -> tuple[set[str], int] | None:
    """Roll back a failed span so the scheduler can re-dispatch it on a clean world.

    The span is ``resume_from`` plus everything that transitively depends on it (the world is
    rebuilt from there, so every step built on top re-runs; the failed step is in that closure
    by ``_validate_dag``'s ancestry check), plus any sibling cancelled mid-execute (it may
    hold partial writes). Returns the rolled-back step ids and the epoch to dispatch under, or
    None to abort: the task then fails with the original error and the live context untouched.

    The rollback is the store's ``undo_steps``: it replays the surviving steps' journaled
    diffs over the run's seed in one CAS, so everything the span wrote goes, whatever it was
    (typed entries, custom metadata, an overwritten scalar), with no per-producer bookkeeping.
    The scheduler's own record rides in the same CAS: the recovered failures flagged
    ``retried`` (kept in ``failed_steps`` as audit, but a consumer reading that list must not
    report a recovered run as failed) and a ``retry_resets`` marker naming the span and the
    sandboxes the undo stripped, for the executor's cleanup to reclaim. The live context is
    then rebuilt from the returned document.

    Abort cases: no task-instance record (nothing to undo through); the store refuses (a run
    that predates the journal, or a surviving step with no entry); the write fails; or a
    surviving step ran concurrently with a span member, whose diff may carry the span's writes
    (``_overlapping_survivors``). The write is shielded from cancellation and, once landed,
    the live context is always brought in line with it.
    """
    from agent_env.task import store as _store

    span_ids = _dependents_closure(resume_from, dependents)
    span_order = [s.id for s in steps if s.id in span_ids]
    rollback_ids = span_ids | cancelled_ids
    logger.warning(
        "%sStep %s failed; rolling back span %s and re-dispatching from %s (attempt %d).",
        log_prefix, failed_id, span_order, resume_from, attempt,
    )
    if not instance_id:
        logger.error(
            "%sCannot retry step %s: this run has no task-instance record to roll back "
            "through; failing with the original error.", log_prefix, failed_id,
        )
        return None
    overlapping = _overlapping_survivors(rollback_ids, completed_ids, windows)
    if overlapping:
        logger.error(
            "%sCannot retry step %s: steps %s completed while the span was executing, so "
            "their journaled diffs may include the span's writes (exact per-step diffs need "
            "per-step isolation, which is not yet implemented); failing with the original error.",
            log_prefix, failed_id, overlapping,
        )
        return None

    failed_steps = copy.deepcopy(context.metadata.get("failed_steps", []))
    for f in failed_steps:
        # A caller-seeded context may carry a legacy non-dict entry; leave it alone.
        if isinstance(f, dict) and f.get("step_id") in rollback_ids:
            f["retried"] = True
    resets = copy.deepcopy(context.metadata.get("retry_resets", []))
    marker = {
        "attempt": attempt,
        "step_id": failed_id,
        "retry_from_step_id": resume_from,
        "replayed": span_order,
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }

    def _audit(before: dict, after: dict) -> ContextUpdateOps:
        # Runs inside the store's CAS with the context as stored and as replayed.
        entry = {**marker, "rolled_back_sandboxes": _rolled_back_sandboxes(before, after)}
        return ContextUpdateOps(sets={
            "context.metadata.failed_steps": failed_steps,
            "context.metadata.retry_resets": resets + [entry],
        })

    persist = asyncio.ensure_future(
        _store.undo_steps(instance_id, rollback_ids, extra_ops=_audit, bump_epoch=True)
    )
    try:
        doc = await asyncio.shield(persist)
    except asyncio.CancelledError:
        # The write runs in a thread and lands regardless; wait for it so the live
        # context matches the stored record before the cancellation unwinds.
        landed = (await asyncio.gather(persist, return_exceptions=True))[0]
        if isinstance(landed, dict):
            _rebuild_context(context, landed["context"])
        raise
    except ValueError as refusal:
        logger.error(
            "%sCannot retry step %s: %s; failing with the original error.",
            log_prefix, failed_id, refusal,
        )
        return None
    if doc is None:
        logger.error(
            "%sSpan rollback for step %s could not be persisted (instance_id=%s); "
            "failing the task instead of retrying on an inconsistent record.",
            log_prefix, failed_id, instance_id,
        )
        return None
    _rebuild_context(context, doc["context"])
    return rollback_ids, doc.get("attempt_epoch", 0)


@dataclass
class _SchedulerState:
    """Initial state the DAG scheduler needs to drive execution.

    - `dependents[step_id]`: steps released when `step_id` completes.
    - `pending[step_id]`: count of deps that still need to complete before `step_id` runs.
    - `completed`: seeded with steps skipped by `start_step`.
    - `ready`: steps whose deps are all satisfied and can launch immediately.
    """
    step_by_id: dict[str, "TaskStep"]
    dependents: dict[str, list[str]]
    pending: dict[str, int]
    completed: set[str]
    ready: list["TaskStep"]


class Task:
    """A task is an ordered sequence of task steps.

    Steps are stored inline and executed by run() according to the DAG
    encoded in each step's depends_on. Independent steps run concurrently.
    """

    type: ClassVar[str] = "task"

    def __init__(
        self,
        id: str,
        version: Optional[int],
        steps: list[TaskStep] | None = None,
    ):
        self.id = id
        self.version = version
        self.steps: list[TaskStep] = steps or []
        self._validate_dag(self.steps)

    @classmethod
    def _validate_dag(cls, steps: list[TaskStep]) -> None:
        ids = {s.id for s in steps}
        seen: set[str] = set()
        for i, step in enumerate(steps):
            if step.id in seen:
                raise ValueError(
                    f"Duplicate step id '{step.id}' at position {i}; step ids must be unique within a task"
                )
            seen.add(step.id)
            prior = {s.id for s in steps[:i]}
            retry_config = getattr(step, "retry_config", None)
            if retry_config is not None:
                # Trust the task author's resume point, but catch an obviously
                # invalid one at creation: it must be the step itself (re-run just
                # this step) or an earlier dependency ancestor. We do not judge
                # which spanned steps are replay-safe.
                ancestors = _ancestor_ids(step.id, _dependency_ids(steps[: i + 1]))
                if (
                    retry_config.retry_from_step_id != step.id
                    and retry_config.retry_from_step_id not in ancestors
                ):
                    raise ValueError(
                        f"Step '{step.id}' at position {i} declares retry_config.retry_from_step_id "
                        f"'{retry_config.retry_from_step_id}' which is not in the step's dependency "
                        f"ancestry (ancestor ids: {sorted(ancestors)})"
                    )
            if step.depends_on is None:
                continue
            for dep in step.depends_on:
                if dep.task_step_id in prior:
                    continue
                why = ("doesn't come before it (forward refs are not allowed: a step depends only on the steps "
                       "before it)" if dep.task_step_id in ids else
                       f"names no step in this task (steps before it: {sorted(prior)})")
                raise ValueError(f"Step '{step.id}' at position {i} declares depends_on '{dep.task_step_id}', which {why}")

    def preflight(self) -> list[str]:
        """Every step's resolvable-config problems, in step order; empty when fine.

        Covers what needs the stores to answer; `_validate_dag` already covers ids and edges.
        """
        problems: list[str] = []
        for step in self.steps:
            problems.extend(step.preflight())
        return problems

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        from agent_env.task_step.registry import get_task_step_registry

        registry = get_task_step_registry()
        steps = []
        for step_dict in data.get("steps", []):
            step_type = step_dict["type"]
            step_cls = registry.get(step_type)
            if step_cls is None:
                raise ValueError(
                    f"Unknown task step type: {step_type}{_registration.failure_note(_registration.TASK_STEPS, step_type)}"
                )
            from agent_env.task_step.task_step import attach_retry_config

            steps.append(attach_retry_config(step_cls.from_dict(step_dict), step_dict))
        return cls(
            id=data["id"],
            version=data.get("version"),
            steps=steps,
        )

    async def run(
        self,
        on_step_start: Callable[[int, int, TaskStep, TaskStepContext], None] | None = None,
        on_step_complete: Callable[[int, int, TaskStep, TaskStepContext, float], None] | None = None,
        agent_model: str | None = None,
        agent_artifact_id: str | None = None,
        start_step: int = 0,
        end_step: int | None = None,
        context: TaskStepContext | None = None,
        instance_id: str | None = None,
    ) -> TaskStepContext:
        """Run steps per their depends_on DAG. Per-step `fail_task_on_error`
        (default True) controls halt-on-failure; failures land in
        context.metadata['failed_steps']. The run is scoped to this task's id, so under namespace
        routing an ``@local`` task's records stay in the local stores.
        """
        with run_scope(self.id):
            return await self._run(
                on_step_start, on_step_complete, agent_model, agent_artifact_id,
                start_step, end_step, context, instance_id,
            )

    async def _run(
        self,
        on_step_start: Callable[[int, int, TaskStep, TaskStepContext], None] | None,
        on_step_complete: Callable[[int, int, TaskStep, TaskStepContext, float], None] | None,
        agent_model: str | None,
        agent_artifact_id: str | None,
        start_step: int,
        end_step: int | None,
        context: TaskStepContext | None,
        instance_id: str | None,
    ) -> TaskStepContext:
        from .store import (
            TaskStepResult,
            TaskStepStatus,
            register_task_instance,
            seed_task_instance_context,
        )

        if context is None:
            context = TaskStepContext()
        context.metadata.setdefault("run_group_id", uuid.uuid4().hex)

        # Seed substitution for EVERY step field, not just prompts. The hub's
        # "Seed values" box writes context.metadata["seed"], and until now only
        # PromptAgentTaskStep._apply_seed consumed it — so a task could
        # parameterize what an LLM reads but not which env it deploys or which
        # artifact it loads. A `<multienv_id>` placeholder in deploy_env.env_id
        # was simply never substituted, and the step tried to deploy the literal
        # string. Applied by round-tripping each step through its own
        # to_dict/from_dict with `<key>` replaced in every string leaf: the
        # steps re-validate themselves, and prompt-level substitution keeps
        # working unchanged (a substituted prompt has no tokens left to apply).
        #
        # Into a LOCAL list, never self.steps: a Task object reused across runs
        # (--k batches) must not carry run 1's substituted values into run 2,
        # and concurrent runs must not race on a shared rewrite. And applied on
        # EVERY start_step, not only 0: a resume reloads the task with its
        # placeholders intact, so skipping substitution would hand later steps
        # literal `<multienv_id>` strings. Substitution is idempotent — an
        # already-replaced string has no tokens left.
        seed = context.metadata.get("seed") or {}
        steps = substitute_seed_in_steps(self.steps, seed) if seed else self.steps
        if start_step < 0 or start_step >= len(steps):
            raise ValueError(f"start_step={start_step} out of range [0, {len(steps)})")
        if end_step is not None and (end_step <= start_step or end_step > len(steps)):
            raise ValueError(f"end_step={end_step} out of range ({start_step}, {len(steps)}]")
        if agent_model is not None:
            context.agent_model = agent_model
        if agent_artifact_id is not None:
            context.agent_artifact_id = agent_artifact_id

        # Pre-format a bracketed prefix for pass@k-correlated log lines.
        # agent-env's log pipeline doesn't index ad-hoc `key=value` fields, so
        # we surface the id as a leading tag (mirroring the `[uid]` /
        # `[passAtK]` convention used upstream in the hub backend's pass@k workflows).
        # Empty string when not running under pass@k so non-pass@k callers
        # (playground, eval) don't get a noisy `[None]` prefix.
        pass_at_k_execution_id = context.metadata.get("pass_at_k_execution_id")
        log_prefix = f"[{pass_at_k_execution_id}] " if pass_at_k_execution_id else ""

        total = len(steps)
        completed_steps = [
            TaskStepResult(step_id=step.id, status=TaskStepStatus.SUCCESS)
            for step in steps[:start_step]
        ]
        instance = register_task_instance(
            self.id,
            self.version,
            total,
            start_step,
            instance_id=instance_id,
            completed_steps=completed_steps,
        )
        instance_id = instance.instance_id if instance else None
        context.instance_id = instance_id
        if instance_id:
            seed_task_instance_context(instance_id, context)

        state = self._build_scheduler_state(steps, start_step, end_step)
        # _drive_dag drains in-flight steps and records the failure itself if it raises.
        first_error, first_error_step_id = await self._drive_dag(
            steps, state, total, context, instance_id,
            on_step_start, on_step_complete, log_prefix,
        )
        if first_error is not None:
            await self._handle_failure(first_error_step_id, first_error, context, instance_id)
        return context

    async def _drive_dag(
        self,
        steps: list[TaskStep],
        state: "_SchedulerState",
        total: int,
        context: TaskStepContext,
        instance_id: str | None,
        on_step_start,
        on_step_complete,
        log_prefix: str,
    ) -> tuple[BaseException | None, str | None]:
        """The DAG scheduler, with in-loop rollback-and-re-dispatch retry.

        Launches ready steps per their depends_on, records each on the instance,
        and returns ``(first_error, first_error_step_id)``. Cancels + drains
        in-flight steps on cancellation/error before re-raising.

        When a step that declares ``retry_config`` fails with retries left
        (whether or not it is tolerant), the scheduler drains in-flight work,
        rolls back the failed span through the store's step journal
        (``_roll_back_span`` -> ``undo_steps``), rebuilds the live context from
        the result, and re-dispatches the span as ordinary steps. Attempt 1 and a
        retry run the same guard-free path; the only difference is the
        ``attempt_epoch`` a completion write carries, which fences a straggler
        from a rolled-back attempt.
        """
        from agent_env.task import store as _store

        index_of = {s.id: i for i, s in enumerate(steps)}
        # Dependency ids per active step (mirrors _build_scheduler_state), used to
        # re-arm after a rollback.
        active_ids = set(state.step_by_id)
        dep_ids: dict[str, set[str]] = {}
        for i, s in enumerate(steps):
            if s.id not in active_ids:
                continue
            if s.depends_on is None:
                dep_ids[s.id] = {p.id for p in steps[:i] if p.id in active_ids}
            else:
                dep_ids[s.id] = {d.task_step_id for d in s.depends_on}

        in_flight: dict[str, asyncio.Task] = {}
        task_to_step_id: dict[asyncio.Task, str] = {}
        first_error: BaseException | None = None
        first_error_step_id: str | None = None
        retry_counts: dict[str, int] = {}
        epoch = 0
        # Execution windows (dispatch tick, end tick) per step, for the retry's concurrency
        # gate: a step that ran while a span member ran may have journaled its writes.
        clock = itertools.count()
        started: dict[str, int] = {}
        ended: dict[str, int] = {}

        async def _heartbeat_step(
            step: TaskStep, pre_context: TaskStepContext, status, dispatch_epoch: int,
        ) -> None:
            if not instance_id:
                return
            ops = build_context_update_ops(pre_context, context)
            # Module attribute (not a bound import) so record_step_complete stays
            # patchable; the upsert-by-step_id keeps a re-dispatched step
            # idempotent, and dispatch_epoch fences a rolled-back straggler.
            await _store.record_step_complete(
                instance_id, step.id, ops, total, Task._utc_now_str(),
                status=status, attempt_epoch=dispatch_epoch,
            )

        def _unblock_dependents(step_id: str) -> None:
            state.completed.add(step_id)
            for dep_id in state.dependents[step_id]:
                state.pending[dep_id] -= 1
                if state.pending[dep_id] == 0 and dep_id not in state.completed:
                    state.ready.append(state.step_by_id[dep_id])

        def _rearm() -> None:
            """Recompute readiness from ``state.completed`` after a rollback so the
            re-dispatched span (and any other uncompleted, dep-satisfied step) runs.
            Nothing is in flight here: the drain preceding a rollback emptied it."""
            for sid in state.step_by_id:
                state.pending[sid] = sum(1 for d in dep_ids[sid] if d not in state.completed)
            state.ready = [
                state.step_by_id[s.id] for s in steps
                if s.id in active_ids and s.id not in state.completed and state.pending[s.id] == 0
            ]

        async def _run_one(step: TaskStep, idx: int, dispatch_epoch: int) -> None:
            if on_step_start:
                result = on_step_start(idx, total, step, context)
                if inspect.isawaitable(result):
                    await result
            # Snapshot context before dispatch so we can emit only the fields
            # this step actually changed — concurrent sibling writes to other
            # fields are preserved by Mongo's single-document atomicity.
            pre_context = copy.deepcopy(context)
            started_mono = time.monotonic()
            started_utc = datetime.now(timezone.utc).isoformat()
            try:
                await step.execute(context)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                duration = time.monotonic() - started_mono
                fatal = getattr(step, "fail_task_on_error", True)
                context.metadata.setdefault("failed_steps", []).append({
                    "step_id": step.id, "step_type": step.type,
                    "error": str(exc), "error_type": type(exc).__name__,
                    "started_at_utc": started_utc, "duration_seconds": duration,
                    "is_fatal": fatal,
                })
                if not fatal:
                    await _heartbeat_step(step, pre_context, _store.TaskStepStatus.FAILURE, dispatch_epoch)
                raise
            duration = time.monotonic() - started_mono
            await _heartbeat_step(step, pre_context, _store.TaskStepStatus.SUCCESS, dispatch_epoch)
            logger.info(
                "%sStep %s (%s) completed in %.1fs",
                log_prefix, step.id, step.type, duration,
            )
            if on_step_complete:
                result = on_step_complete(idx, total, step, context, duration)
                if inspect.isawaitable(result):
                    await result

        async def _drain_in_flight() -> set[str]:
            """Cancel and await every in-flight step; return the ids cancelled.

            A sibling cancelled mid-execute may hold partial context writes, so the
            caller rolls those ids back too. Cancellation cannot interrupt an
            in-flight persist; the epoch fence, not the cancel, stops a drained
            step's late write."""
            cancelled = set(in_flight)
            tick = next(clock)
            for sid in cancelled:
                ended[sid] = tick
            for t in list(in_flight.values()):
                t.cancel()
            await asyncio.gather(*in_flight.values(), return_exceptions=True)
            in_flight.clear()
            task_to_step_id.clear()
            return cancelled

        try:
            while state.ready or in_flight:
                while state.ready and first_error is None:
                    step = state.ready.pop(0)
                    started[step.id] = next(clock)
                    t = asyncio.create_task(_run_one(step, index_of[step.id], epoch))
                    in_flight[step.id] = t
                    task_to_step_id[t] = step.id
                if not in_flight:
                    break
                done, _ = await asyncio.wait(
                    list(in_flight.values()), return_when=asyncio.FIRST_COMPLETED,
                )
                # Classify the batch: a failed step with retry budget is a retry
                # candidate (tolerance applies only once the budget is spent); a
                # tolerant failure continues; any other failure is terminal.
                retryable: list[tuple[TaskStep, RetryConfig, BaseException]] = []
                for t in done:
                    step_id = task_to_step_id.pop(t)
                    in_flight.pop(step_id)
                    ended[step_id] = next(clock)
                    exc = t.exception()
                    if exc is None:
                        _unblock_dependents(step_id)
                        continue
                    step_obj = state.step_by_id[step_id]
                    retry_cfg = getattr(step_obj, "retry_config", None)
                    if retry_cfg is not None and retry_counts.get(step_id, 0) < retry_cfg.max_retries:
                        retryable.append((step_obj, retry_cfg, exc))
                        continue
                    if not getattr(step_obj, "fail_task_on_error", True):
                        logger.warning(
                            "%sStep %s failed (tolerant, continuing): %s", log_prefix, step_id, exc,
                        )
                        _unblock_dependents(step_id)
                        continue
                    # Every fatal failure is logged; only the first is the task's error.
                    logger.error("%sStep %s failed (fatal): %s", log_prefix, step_id, exc)
                    if first_error is None:
                        first_error, first_error_step_id = exc, step_id

                if first_error is not None:
                    continue  # a terminal fatal wins; the loop drains what's left
                if len(retryable) > 1:
                    # Two overlapping or independent spans can't be rebuilt in one
                    # pass (a non-rolled-back loser would silently re-run on a dirty
                    # world), so a multi-retryable batch is terminal.
                    step_obj, _, exc = retryable[0]
                    first_error, first_error_step_id = exc, step_obj.id
                    logger.error(
                        "%s%d retryable steps failed in one batch (%s); failing the task "
                        "rather than retrying overlapping spans.",
                        log_prefix, len(retryable), ", ".join(s.id for s, _, _ in retryable),
                    )
                elif len(retryable) == 1:
                    step_obj, retry_cfg, exc = retryable[0]
                    retry_counts[step_obj.id] = retry_counts.get(step_obj.id, 0) + 1
                    cancelled = await _drain_in_flight()
                    now_tick = next(clock)
                    outcome = await _roll_back_span(
                        context, instance_id, steps, state.dependents,
                        failed_id=step_obj.id, resume_from=retry_cfg.retry_from_step_id,
                        cancelled_ids=cancelled, attempt=retry_counts[step_obj.id] + 1,
                        windows={sid: (t0, ended.get(sid, now_tick)) for sid, t0 in started.items()},
                        completed_ids=set(state.completed), log_prefix=log_prefix,
                    )
                    if outcome is None:
                        # Could not roll back (refused, unpersisted, or a concurrent survivor):
                        # fail with the original error, live context untouched.
                        first_error, first_error_step_id = exc, step_obj.id
                    else:
                        rollback_ids, epoch = outcome
                        for sid in rollback_ids:
                            state.completed.discard(sid)
                        _rearm()
        except (asyncio.CancelledError, Exception) as exc:
            await _drain_in_flight()
            if instance_id:
                # Record with the fatal step's attribution: a cancellation landing
                # during the drain would otherwise overwrite the instance record
                # with the CancelledError and a null failing-step id.
                await _store.record_task_failure(
                    instance_id, first_error_step_id, context,
                    first_error if first_error is not None else exc, Task._utc_now_str(),
                )
            raise

        return first_error, first_error_step_id

    def _build_scheduler_state(self, steps: list, start_step: int, end_step: int | None = None) -> _SchedulerState:
        # Truncate steps at end_step (exclusive); depends_on only ever points at
        # earlier steps, so active steps never reference truncated ones.
        active_steps = steps[:end_step] if end_step is not None else steps
        step_by_id = {s.id: s for s in active_steps}
        dependents: dict[str, list[str]] = {step_id: [] for step_id in step_by_id}
        pending: dict[str, int] = {}
        for i, step in enumerate(active_steps):
            # When depends_on is None, we assume all prior steps are dependencies.
            if step.depends_on is None:
                dependency_step_ids = {s.id for s in active_steps[:i]}
            else:
                dependency_step_ids = {d.task_step_id for d in step.depends_on}
            for dependency_step_id in dependency_step_ids:
                dependents[dependency_step_id].append(step.id)
            pending[step.id] = len(dependency_step_ids)

        completed: set[str] = set()
        for s in active_steps[:start_step]:
            completed.add(s.id)
            for dependent_step_id in dependents[s.id]:
                pending[dependent_step_id] = max(0, pending[dependent_step_id] - 1)

        ready: list[TaskStep] = [
            s for i, s in enumerate(active_steps)
            if i >= start_step and pending[s.id] == 0
        ]
        return _SchedulerState(
            step_by_id=step_by_id,
            dependents=dependents,
            pending=pending,
            completed=completed,
            ready=ready,
        )

    @staticmethod
    async def _handle_failure(
        failing_step_id: str | None,
        error: BaseException,
        context: TaskStepContext,
        instance_id: str | None,
    ) -> None:
        """Record the terminal failure on the instance and re-raise it."""
        from .store import record_task_failure

        if instance_id:
            await record_task_failure(
                instance_id, failing_step_id, context, error, Task._utc_now_str(),
            )
        raise error

    @staticmethod
    def _utc_now_str() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    @classmethod
    def get(cls, id: str, version: Optional[int] = None) -> "Task":
        from .store import get_task_store

        return get_task_store().get(id, version)

    @classmethod
    def put(cls, **kwargs) -> "Task":
        from .store import get_task_store

        kwargs.setdefault("version", None)
        instance = cls(**kwargs)
        return get_task_store().put_document(instance)

    @classmethod
    def query(cls) -> "TaskQuery":
        from .store import TaskQuery, get_task_store

        return TaskQuery(get_task_store())

    @staticmethod
    def get_instance(instance_id: str) -> "TaskInstance":
        from .store import get_task_instance_store

        return get_task_instance_store().get(instance_id)
