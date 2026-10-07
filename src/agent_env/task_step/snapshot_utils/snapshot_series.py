"""The capture series: freezes agent state mid-run, on a cadence or per turn.

``prompt_agent`` optionally drives ``SnapshotSeries`` *while the prompt is still in
flight* — on a wall-clock cadence (``interval_seconds``), at every conversation-turn
boundary (``per_turn``), or both — so a run yields a curve of gradable points instead
of a single endpoint. Nothing here grades: each capture appends one row to
``context.metadata['agent_snapshots']``, and a row is gradable when it has a
``bundle_object_url`` (``capture_status`` is provenance, not a gate).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from agent_env.store.ids import derive_id, is_local_id, validate_local_id
from agent_env.task_step.context import TaskStepContext, dual_keyed
from agent_env.task_step.snapshot_utils import agent_state_capture as capture

logger = logging.getLogger(__name__)

# How much longer than a capture's own budget teardown waits for the lock, so the
# two deadlines cannot land in the same loop iteration. See `_acquire_for_final`.
_FINAL_LOCK_GRACE_SECONDS = 15


def _is_cancelling() -> bool:
    task = asyncio.current_task()
    return bool(task.cancelling()) if task is not None else False


@dataclass
class SnapshotConfig:
    """Periodic capture settings for a ``prompt_agent`` step.

    Its presence on the step is the on/off switch, so there is no half-configured
    state: every field defaults usefully or is rejected in ``__post_init__``.
    """

    interval_seconds: Optional[int] = None
    at_end: bool = True
    # One capture per AGENT turn, taken after the agent replies and before the user
    # sim's next message is sent. Independent of the ticker; both may be on.
    per_turn: bool = False
    env_id: Optional[str] = None
    timeout_seconds: int = 300
    # Interior captures only — the final one is exempt, so a series can yield
    # `max_snapshots + 1` rows.
    max_snapshots: int = 24

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("snapshot timeout_seconds must be > 0")
        if self.max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")
        if not self.at_end and not self.ticks and not self.per_turn:
            raise ValueError(
                "snapshotting with at_end=False needs a positive interval_seconds "
                f"(got {self.interval_seconds!r}) or per_turn=True; otherwise no "
                "capture ever runs"
            )

    @property
    def ticks(self) -> bool:
        """Non-positive starts no ticker: ``wait_for`` would fire at once."""
        return self.interval_seconds is not None and self.interval_seconds > 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SnapshotConfig":
        defaults = cls()
        return cls(
            interval_seconds=data.get("interval_seconds"),
            at_end=data.get("at_end", defaults.at_end),
            per_turn=data.get("per_turn", defaults.per_turn),
            env_id=data.get("env_id"),
            timeout_seconds=data.get("timeout_seconds", defaults.timeout_seconds),
            max_snapshots=data.get("max_snapshots", defaults.max_snapshots),
        )


def _rollout_base(step_id: str, instance_id: Optional[str]) -> str:
    """The id every artifact id in one series is derived from: the rollout's instance id.

    Instance-scoped, so concurrent rollouts never contend on one id and retries
    reuse it. Deliberately not derived from the context id: a caller-supplied
    ``context_id`` is shared by every rollout, so ids built from it would collide.
    """
    if instance_id:
        return instance_id
    generated = f"adhoc-{uuid.uuid4().hex[:12]}"
    logger.warning(
        "%s: no instance_id in context; using random artifact base %s "
        "(retries will not be idempotent)", step_id, generated,
    )
    return generated


class SnapshotSeries:
    """One rollout's capture series: a ticker, the in-flight lock, and teardown.

    One instance per rollout, held as a local: k concurrent rollouts share one
    ``TaskStep``, so none of this state may live on the step.

    The cadence is the gap BETWEEN captures — the ticker awaits its own, so captures
    never overlap and the lock's only other contender is teardown.
    """

    def __init__(
        self,
        *,
        step_id: str,
        agent_name: str,
        prompt_id: str,
        a2a_context_id: str,
        config: SnapshotConfig,
        trajectory_output_prefix: str,
        instance_id: Optional[str] = None,
    ):
        self.step_id = step_id
        self.agent_name = agent_name
        self.prompt_id = prompt_id
        # The id that went on the wire (`prompt_agent`'s `solver_context_id`), never
        # its conversation-doc id: every question a capture asks is agent-side, and
        # that is the only name the sidecar knows the session by.
        self.a2a_context_id = a2a_context_id
        self.config = config
        self.trajectory_output_prefix = trajectory_output_prefix
        self._base = _rollout_base(step_id, instance_id)
        self.workspace_artifact_id = derive_id(self._base, f"{step_id}-workspace")
        if is_local_id(self._base):
            validate_local_id(derive_id(self._base, f"snapshot-{step_id}-workspace"))
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._ticker: Optional[asyncio.Task] = None
        self._started_monotonic = time.monotonic()
        # Interior captures attempted — what `max_snapshots` bounds. Not a
        # row ordinal: failed rows consume attempts too, and the final is exempt.
        self._attempts = 0
        # See `_refuse_turn`.
        self._turn_limit_recorded = False
        self._rows: list[dict] = []

    # ------------------------------------------------------------- lifecycle

    def start(self, context: TaskStepContext) -> None:
        """Start ticking, if a cadence was configured."""
        if self.config.ticks:
            self._ticker = asyncio.create_task(self._tick_loop(context))

    async def finish(self, context: TaskStepContext) -> None:
        """Take the final capture, then publish the series.

        Raises only ``CancelledError``, which must propagate — a caller that
        swallowed it would complete a cancelled step silently.
        """
        cancelling = _is_cancelling()
        try:
            self._stop.set()
            if cancelling:
                # Waiting for the lock and capturing are each bounded by
                # `timeout_seconds`, so honouring them here would delay the
                # cancellation by up to twice that (10 min at defaults).
                self._record(status="skipped", reason="cancelled", is_final=True)
                return
            # Waited out even when capturing nothing here, so an in-flight interior
            # capture's S3 object and burnt version end up referenced by a row.
            acquired = await self._acquire_for_final()
            try:
                if not self.config.at_end:
                    return
                if not acquired:
                    self._record(status="skipped", is_final=True,
                                 reason="final_capture_lock_timeout")
                    return
                self._append(await self._capture_never_raising(context, is_final=True))
            finally:
                if acquired:
                    self._lock.release()
        except asyncio.CancelledError:
            # Cancellation that arrived DURING the lock wait or the capture, so the
            # flag sampled at entry is stale — the `finally` must still force the
            # ticker down or a mid-capture one survives teardown.
            cancelling = True
            self._record(status="skipped", reason="cancelled", is_final=True)
            raise
        except Exception:
            logger.exception("%s: final capture raised; see its manifest row", self.step_id)
        finally:
            try:
                # Re-sampled as well: a cancellation can be pending without having
                # been delivered at an await yet.
                await self._retire_ticker(force=cancelling or _is_cancelling())
            finally:
                # Nested: retirement re-raises a cancellation aimed at us, and the
                # rows live nowhere else. Safe here because `_publish` never awaits.
                self._publish(context)

    async def _acquire_for_final(self) -> bool:
        """Take the lock for the final capture, outlasting the capture it waits on.

        Strictly LONGER than ``timeout_seconds``, never equal: an interior capture is
        bounded by that same value, so equal bounds put both deadlines in one loop
        iteration and this waiter times out before the interior capture can release —
        recording a fatal ``skipped`` final row for a rollout that in fact completed.
        """
        timeout = self.config.timeout_seconds + _FINAL_LOCK_GRACE_SECONDS
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "%s: capture for %s still in flight after %ss; closing the series",
                self.step_id, self.a2a_context_id, timeout,
            )
            return False

    async def _retire_ticker(self, *, force: bool = False) -> None:
        """Stop the ticker before ``execute`` returns.

        Idle: cancel and await — ``cancel()`` only schedules, and an unawaited task
        outlives the step holding this rollout's context.

        Mid-capture: leave it unless ``force``. Cancelling stops our client but not
        the sidecar, which keeps tarring and orphans its object; ``stop`` is set and
        the capture bounded, so it ends on its own. One inside ``asyncio.to_thread``
        cannot be stopped regardless.
        """
        if self._ticker is None or self._ticker.done():
            return
        if self._lock.locked() and not force:
            logger.warning(
                "%s: ticker for %s is mid-capture; letting it finish rather than "
                "orphaning a sidecar upload", self.step_id, self.a2a_context_id,
            )
            return
        if self._lock.locked():
            logger.warning(
                "%s: cancelling a mid-capture ticker for %s under cancellation; its "
                "sidecar upload may be orphaned", self.step_id, self.a2a_context_id,
            )
        # Swallow the ticker's own cancellation, propagate one aimed at US — else a
        # cancelled step completes silently. `.cancelled()` reads True either way.
        parent = asyncio.current_task()
        cancels_before = parent.cancelling() if parent is not None else 0
        self._ticker.cancel()
        try:
            await self._ticker
        except asyncio.CancelledError:
            cancels_after = parent.cancelling() if parent is not None else 0
            if cancels_after > cancels_before:
                raise

    @property
    def _at_limit(self) -> bool:
        """Whether the interior budget is spent. Rechecked under the lock by both
        contenders: the increment is only safe unlocked because no await separates it
        from `acquire()`, and one added there would let both spend the last slot."""
        return self._attempts >= self.config.max_snapshots

    def _refuse_turn(self, turn: int) -> None:
        """Note that this turn boundary was skipped because the budget is spent.
        Recorded once, not once per remaining turn, so the row count stays bounded
        no matter how long the conversation runs."""
        if not self._turn_limit_recorded:
            self._turn_limit_recorded = True
            self._record(status="skipped", reason="limit_reached", turn=turn)

    async def _tick_loop(self, context: TaskStepContext) -> None:
        # Nothing awaits this task, so an escaping exception would vanish into
        # asyncio's "never retrieved" path and snapshotting would just stop.
        try:
            await self._ticks(context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "%s: snapshot ticker for %s stopped unexpectedly",
                self.step_id, self.a2a_context_id,
            )
            self._record(
                status="failed",
                reason=f"ticker_error: {type(exc).__name__}: {exc}"[:300],
            )

    async def _ticks(self, context: TaskStepContext) -> None:
        """Capture every ``interval_seconds`` until teardown or the attempt limit."""
        while True:
            try:
                # Not sleep(): teardown wakes the ticker immediately rather than
                # sleeping out the remaining interval.
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.interval_seconds
                )
                return
            except asyncio.TimeoutError:
                pass  # interval elapsed → a capture is due

            if self._at_limit:
                # Stop ticking, but leave the final capture to teardown: counting
                # it here would let ticks suppress the grade of record.
                self._record(status="skipped", reason="limit_reached")
                return

            async with self._lock:
                # Rechecked: teardown can set `stop` and take the lock while this
                # tick waits on it, and a capture started afterwards would burn a
                # universe version teardown has already stopped waiting for.
                if self._stop.is_set():
                    return
                # The budget is rechecked too — see `_at_limit`.
                if self._at_limit:
                    self._record(status="skipped", reason="limit_reached")
                    return
                self._attempts += 1
                self._append(await self._capture_never_raising(context, is_final=False))

    async def capture_turn(self, context: TaskStepContext, turn: int) -> None:
        """Capture at a conversation-turn boundary. No-op unless ``per_turn``.

        Awaited by the turn loop, not run on the ticker's task: the agent has stopped
        writing only until the user sim's reply starts it again, so the capture has to
        finish before that reply goes out. Never raises except ``CancelledError``.
        """
        if not self.config.per_turn or self._stop.is_set():
            return
        if self._at_limit:
            self._refuse_turn(turn)
            return
        async with self._lock:
            # Rechecked under the lock for the same reason `_ticks` does: teardown
            # may have closed the series while this waited on an in-flight tick.
            if self._stop.is_set():
                return
            if self._at_limit:
                self._refuse_turn(turn)
                return
            self._attempts += 1
            self._append(
                await self._capture_never_raising(context, is_final=False, turn=turn)
            )

    # --------------------------------------------------------------- capture

    async def _capture_never_raising(
        self, context: TaskStepContext, *, is_final: bool, turn: Optional[int] = None
    ) -> dict:
        """Run one capture under a hard timeout. Returns a row; never raises
        except ``CancelledError``.

        A timeout abandons the capture, possibly orphaning an S3 object — safe,
        since each capture writes to a unique prefix.
        """
        # Owned here, not built inside `_capture`, so a failure part-way through
        # still returns whatever landed before it.
        row = self._row(is_final=is_final, status="ok", turn=turn)
        try:
            return await asyncio.wait_for(
                self._capture(context, row, is_final=is_final),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "%s: capture timed out after %ss (is_final=%s turn=%s)",
                self.step_id, self.config.timeout_seconds, is_final, turn,
            )
            return self._degrade(row, f"capture_timeout_{self.config.timeout_seconds}s")
        except Exception as exc:
            logger.warning(
                "%s: capture failed (is_final=%s turn=%s, run continues): %s",
                self.step_id, is_final, turn, exc, exc_info=True,
            )
            return self._degrade(row, f"{type(exc).__name__}: {exc}"[:300])

    @staticmethod
    def _degrade(row: dict, reason: str) -> dict:
        """``partial`` when the bundle landed (still gradable), ``failed`` when it
        did not — which is fatal for the final capture."""
        if row.get("bundle_object_url"):
            row["capture_status"] = "partial"
            row["capture_reason"] = "; ".join(
                p for p in (row.get("capture_reason"), reason) if p
            )
        else:
            row["capture_status"] = "failed"
            row["capture_reason"] = reason
        return row

    async def _capture(
        self, context: TaskStepContext, row: dict, *, is_final: bool
    ) -> dict:
        """The three reads: workspace tar, trajectory-so-far, env service state.

        Not atomic and the agent writes throughout, so a degraded trajectory or env
        read yields ``partial``; only a missing tar is ``failed``. Mutates ``row``
        in place.
        """
        # One budget across all three reads, so the outer `wait_for` can bound them.
        deadline = time.monotonic() + self.config.timeout_seconds

        def remaining() -> float:
            return max(1.0, deadline - time.monotonic())  # never a zero-timeout read

        reasons: list[str] = []
        agent = next(
            (a for a in context.deployed_agents if a.agent_name == self.agent_name), None
        )
        if agent is None:
            raise RuntimeError(f"No deployed agent named '{self.agent_name}' found in context")
        a2a_url = agent.a2a_url or agent.api_url
        if not a2a_url:
            raise RuntimeError(f"Deployed agent '{self.agent_name}' has no a2a_url")
        a2a_card = agent.a2a_card or {}

        workspace = await capture.capture_workspace(
            a2a_url=a2a_url,
            a2a_card=a2a_card,
            agent_name=self.agent_name,
            a2a_context_id=self.a2a_context_id,
            artifact_id=self.workspace_artifact_id,
            timeout_seconds=remaining(),
            sandbox_type=agent.sandbox_type,
        )
        row["id"] = workspace.universe_id
        row["version"] = workspace.universe_version
        row["bundle_object_url"] = workspace.bundle_object_url

        # Every point reads its trajectory the same way, cumulatively.
        # `PromptResponse.agent_trajectory_object_url` is only the LAST turn's, so using
        # it for the final row would make that point narrower than its predecessors.
        # It stays the fallback: on a failed run the live read may be gone while the
        # per-turn upload already landed.
        traj = await capture.read_partial_trajectory(
            a2a_url=a2a_url,
            a2a_card=a2a_card,
            context_id=self.a2a_context_id,
            timeout_seconds=remaining(),
            trajectory_output_prefix=self.trajectory_output_prefix,
            sandbox_type=agent.sandbox_type,
        )
        if traj.reason and is_final:
            recorded = self._recorded_trajectory_uri(context)
            row.update(dual_keyed("trajectory_s3_uri", "trajectory_object_url", recorded))
            reasons.append(traj.reason if recorded else f"{traj.reason}; trajectory_missing")
        elif traj.reason:
            reasons.append(traj.reason)
        elif traj.object_url:
            row.update(dual_keyed("trajectory_s3_uri", "trajectory_object_url", traj.object_url))
        else:
            try:
                uploaded = await asyncio.to_thread(
                    capture.upload_trajectory, traj.trajectory,
                    self.trajectory_output_prefix,
                )
                row.update(dual_keyed("trajectory_s3_uri", "trajectory_object_url", uploaded))
            except Exception as exc:
                logger.warning("%s: trajectory upload failed: %s", self.step_id, exc)
                reasons.append("trajectory_upload_failed")

        if self.config.env_id:
            reason = await self._capture_env_state(context, row, remaining())
            if reason:
                reasons.append(reason)

        if reasons:
            # Never `failed`: disqualifying on an auxiliary read would discard the
            # whole curve the moment one degrades for the run's lifetime.
            row["capture_status"] = "partial"
            row["capture_reason"] = "; ".join(reasons)
        return row

    async def _capture_env_state(
        self, context: TaskStepContext, row: dict, timeout_seconds: float
    ) -> Optional[str]:
        """Freeze the env's service state as a ``EnvironmentUniverseArtifact`` version,
        recording its ``{id, version}`` on the row. Returns a reason on failure —
        the export is all-or-nothing, so one unreachable service costs this leg of
        the tick, not the workspace bundle."""
        from agent_env.env.env import Env, gateway_url_of
        from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep

        deployed = next(
            (d for d in context.deployed_envs if d.env_id == self.config.env_id), None
        )
        if deployed is None:
            logger.warning(
                "%s: env_id=%s not in deployed_envs (%s); skipping env capture",
                self.step_id, self.config.env_id, [d.env_id for d in context.deployed_envs],
            )
            return "env_not_deployed"
        if not gateway_url_of(deployed):
            return "env_has_no_gateway_url"

        try:
            # Pin the env version the sandbox actually runs, so service
            # enumeration can't drift if the env was re-registered since deploy.
            env = await asyncio.to_thread(Env.get, deployed.env_id, deployed.env_version)
            result = await SnapshotEnvTaskStep.snapshot_env_state(
                env=env,
                gateway_url=deployed.gateway_url,
                # Instance-scoped like the workspace id, never context-scoped: a
                # pinned `context_id` is shared by every concurrent rollout.
                snapshot_id=derive_id(self._base, f"snapshot-{self.step_id}"),
                deployed=deployed,
                export_timeout_seconds=int(timeout_seconds),
            )
        except Exception as exc:
            logger.warning(
                "%s: env state capture failed: %s", self.step_id, exc, exc_info=True,
            )
            return f"env_capture_failed: {type(exc).__name__}"

        row["env_universe_id"] = result.environment_universe_artifact_id
        row["env_universe_version"] = result.environment_universe_artifact_version
        return None

    def _recorded_trajectory_uri(self, context: TaskStepContext) -> Optional[str]:
        """This rollout's per-turn trajectory uri from ``prompt_responses``.

        Matched on ``a2a_context_id``, newest-first: a retry can restore a context
        snapshot taken during an earlier attempt, so that attempt's response for the
        same ``prompt_id`` can still be present and would otherwise attach a
        previous conversation's trajectory to this capture.
        """
        for prompt in reversed(context.prompt_responses):
            if (
                prompt.prompt_id == self.prompt_id
                and prompt.a2a_context_id == self.a2a_context_id
            ):
                return prompt.agent_trajectory_object_url
        # None is legitimate: the response is recorded only after the turn loop, so
        # a capture during an in-flight or failed run has no entry yet.
        return None

    # ------------------------------------------------------------------ rows

    def _row(
        self,
        *,
        is_final: bool,
        status: str,
        reason: Optional[str] = None,
        turn: Optional[int] = None,
    ) -> dict:
        """Build a manifest row. No stored ordinal: rank by ``version`` within a
        ``source_context_id``, which survives a filtered subset where array
        position would not. ``turn`` is 1-based and only set on turn-boundary rows."""
        return {
            "turn": turn,
            "id": None,
            "version": None,
            "bundle_object_url": None,
            "source_agent_name": self.agent_name,
            "source_context_id": self.a2a_context_id,
            "elapsed_s": int(time.monotonic() - self._started_monotonic),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "is_final": is_final,
            "capture_status": status,
            "capture_reason": reason,
            **dual_keyed("trajectory_s3_uri", "trajectory_object_url", None),
            "env_universe_id": None,
            "env_universe_version": None,
        }

    def _record(
        self, *, status: str, reason: str, is_final: bool = False,
        turn: Optional[int] = None,
    ) -> None:
        self._append(
            self._row(is_final=is_final, status=status, reason=reason, turn=turn)
        )

    def _append(self, row: dict) -> None:
        for key in [k for k, v in row.items() if v is None]:
            del row[key]
        self._rows.append(row)
        logger.info(
            "%s: recorded snapshot (v=%s is_final=%s turn=%s status=%s elapsed_s=%s "
            "env_universe_version=%s reason=%s)",
            self.step_id, row.get("version"), row["is_final"], row.get("turn"),
            row["capture_status"], row["elapsed_s"], row.get("env_universe_version"),
            row.get("capture_reason"),
        )

    def _publish(self, context: TaskStepContext) -> None:
        """Publish the collected rows to the context, once, at teardown: collecting on
        the series means a capture racing teardown cannot land after the final row."""
        context.metadata.setdefault("agent_snapshots", []).extend(self._rows)
        final = next(
            (r for r in reversed(self._rows)
             if r.get("is_final") and r.get("env_universe_id")),
            None,
        )
        if final:
            # Under the step id, so a judge step can be wired to this one via
            # `load_artifact`'s `artifact_from_step_id` without knowing it snapshots.
            context.metadata.setdefault("env_snapshotted_universes", {})[self.step_id] = {
                "id": final["env_universe_id"],
                "version": final["env_universe_version"],
            }

    # ------------------------------------------------------------- fatality

    def raise_if_final_capture_missing(self) -> None:
        """Fail the step when the agent succeeded but its final capture did not —
        otherwise the instance reports `completed` with no gradable point.

        Keyed on a landed bundle, not `capture_status`: a `partial` row with a bundle
        is gradable, while `final_capture_lock_timeout` is `skipped` yet fatal.
        """
        if not self.config.at_end:
            return
        final = next((r for r in reversed(self._rows) if r.get("is_final")), None)
        if final is None:
            # Unreachable after a successful run, and no recorded failure to
            # attribute it to — so loud, but not fatal.
            logger.warning(
                "%s: agent run succeeded but no is_final snapshot row was recorded",
                self.step_id,
            )
            return
        if final.get("bundle_object_url"):
            return
        raise RuntimeError(
            f"{self.step_id}: final capture did not land "
            f"(status={final.get('capture_status')} reason={final.get('capture_reason')}) — "
            "the rollout completed but has no gradable end state, so failing the step "
            "rather than reporting a complete instance with nothing to grade"
        )
