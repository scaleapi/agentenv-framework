"""``prompt_agent``'s mid-run capture: the kill switch, the ticker rules, the turn
boundaries, the manifest rows, and the one deliberate fatality.

The feature is off unless ``snapshot_config`` is set. Since
``prompt_agent`` is the most-used step in the repo, the inertness tests below are
load-bearing: an unconfigured step must behave exactly as it did before.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from agent_env.config import set_object_store
from agent_env.store.object_store import S3ObjectStore
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.snapshot_utils import agent_state_capture as mod
from agent_env.task_step.snapshot_utils import snapshot_series as ps
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

from .capture_stubs import BUCKET, context

TRAJ_PREFIX = f"s3://{BUCKET}/traj/"


@pytest.fixture(autouse=True)
def _s3_store_backs_prefix_minting():
    # Trajectory prefixes are minted by the configured object store; pin an S3 one so
    # they stay s3://BUCKET/... instead of resolving a filesystem path under cwd.
    set_object_store(S3ObjectStore(client=object(), bucket=BUCKET))
INSTANCE_ID = "solve-run-0123456789abcdef"
# `INSTANCE_ID`'s tail, capped at 16 — what every artifact id is scoped by.
DISCRIMINATOR = "0123456789abcdef"


def _series(
    *,
    a2a_context_id: str = "cid-1",
    instance_id: str = INSTANCE_ID,
    **cfg_overrides,
) -> ps.SnapshotSeries:
    cfg = dict(
        interval_seconds=None,
        at_end=True,
        env_id=None,
        timeout_seconds=5,
        max_snapshots=24,
    )
    cfg.update(cfg_overrides)
    return ps.SnapshotSeries(
        step_id="solve",
        agent_name="solver",
        prompt_id="p1",
        a2a_context_id=a2a_context_id,
        trajectory_output_prefix=TRAJ_PREFIX,
        config=ps.SnapshotConfig(**cfg),
        instance_id=instance_id,
    )


def _snapshotting_step(**cfg_overrides) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver",
        snapshot_config=ps.SnapshotConfig(**cfg_overrides),
    )


class _FakeWorkspace:
    """Stands in for ``capture_workspace``; counts calls and can block or fail."""

    def __init__(
        self,
        *,
        gate: asyncio.Event | None = None,
        fail_after: int | None = None,
        gate_only_first: bool = False,
    ):
        self.calls = 0
        self.gate = gate
        self.fail_after = fail_after
        # Gate the first call only, so a later capture can still succeed while an
        # earlier one is stuck.
        self.gate_only_first = gate_only_first
        # Every artifact id this fake was asked to version into, in call order.
        self.artifact_ids: list[str] = []
        # Every context id it was addressed by — the sidecar's only handle on
        # which conversation to tar.
        self.context_ids: list[str] = []

    async def __call__(self, **kwargs):
        self.calls += 1
        artifact_id = kwargs.get("artifact_id", "wsp")
        self.artifact_ids.append(artifact_id)
        self.context_ids.append(kwargs.get("a2a_context_id"))
        if self.gate is not None and not (self.gate_only_first and self.calls > 1):
            await self.gate.wait()
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("sidecar exploded")
        prefix = f"s3://{BUCKET}/agent_snapshots/{artifact_id}/{self.calls}-abc/"
        return mod.WorkspaceCapture(
            universe_id=artifact_id,
            universe_version=self.calls,
            bundle_object_url=prefix,
            capture_prefix=prefix,
        )


def _install(monkeypatch, workspace=None, *, trajectory_reason=None, upload=True):
    """Stub the three reads a capture makes. Returns the workspace fake."""
    ws = workspace or _FakeWorkspace()
    monkeypatch.setattr(mod, "capture_workspace", ws)

    async def fake_trajectory(**kwargs):
        if trajectory_reason:
            return mod.TrajectoryCapture(reason=trajectory_reason)
        return mod.TrajectoryCapture(trajectory=[{"role": "user"}])

    monkeypatch.setattr(mod, "read_partial_trajectory", fake_trajectory)
    if upload:
        monkeypatch.setattr(
            mod, "upload_trajectory", lambda traj, prefix: f"{prefix}trajectory-x.json"
        )
    return ws


def _rows(ctx: TaskStepContext) -> list[dict]:
    """Rows as published to the context — only populated once ``finish`` ran."""
    return ctx.metadata.get("agent_snapshots") or []


def _pending(series: ps.SnapshotSeries) -> list[dict]:
    """Rows collected on the series so far, before teardown publishes them."""
    return series._rows


@pytest.mark.asyncio
async def test_object_mode_trajectory_is_not_uploaded_twice(monkeypatch):
    _install(monkeypatch)
    direct_url = f"s3://{BUCKET}/prompt_agent_trajectories/direct.json"

    async def direct_trajectory(**kwargs):
        assert kwargs["trajectory_output_prefix"].startswith(f"s3://{BUCKET}/")
        return mod.TrajectoryCapture(object_url=direct_url)

    def unexpected_upload(*args, **kwargs):
        raise AssertionError("object-mode trajectory is already durable")

    monkeypatch.setattr(mod, "read_partial_trajectory", direct_trajectory)
    monkeypatch.setattr(mod, "upload_trajectory", unexpected_upload)

    row = await _series()._capture_never_raising(context(), is_final=False)

    assert row["capture_status"] == "ok"
    assert row["trajectory_s3_uri"] == direct_url


async def _until(pred, timeout=2.0, what="condition"):
    """Poll until ``pred()``; deterministic without pinning wall-clock timings."""
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


# ============================================================== kill switch

def test_an_unconfigured_step_emits_no_snapshot_keys():
    step = PromptAgentTaskStep(id="s", version=None, prompt="hi")
    doc = step.to_dict()
    assert "snapshot_config" not in doc
    # And a doc written before this feature existed still round-trips.
    assert PromptAgentTaskStep.from_dict(doc).snapshot_config is None


@pytest.mark.asyncio
async def test_an_unconfigured_step_never_builds_a_series(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("an unconfigured step must not build a SnapshotSeries")

    monkeypatch.setattr(pa_mod, "SnapshotSeries", explode)
    seen = {}

    async def fake_conversation(self, context, *, a2a_context_id=None, series=None):
        seen["a2a_context_id"] = a2a_context_id
        return context

    monkeypatch.setattr(PromptAgentTaskStep, "_execute_conversation", fake_conversation)
    ctx = TaskStepContext()
    await PromptAgentTaskStep(id="s", version=None, prompt="hi").execute(ctx)

    # The wire id is minted either way — it is the conversation's, not the series'.
    assert len(seen["a2a_context_id"]) == 32
    assert "agent_snapshots" not in ctx.metadata


def test_the_configured_config_round_trips():
    cfg = ps.SnapshotConfig(
        interval_seconds=1800, at_end=False, env_id="env-1",
        timeout_seconds=120, max_snapshots=5,
    )
    step = PromptAgentTaskStep(
        id="s", version=None, prompt="hi", snapshot_config=cfg
    )
    assert PromptAgentTaskStep.from_dict(step.to_dict()).snapshot_config == cfg


def test_a_config_in_a_step_document_is_coerced():
    """A step document is the one place a dict legitimately arrives, so `from_dict`
    is the one place that coerces — `__init__` takes the dataclass only, where a
    mistyped key is a TypeError rather than a silently defaulted field."""
    step = PromptAgentTaskStep.from_dict({
        "id": "s", "type": "prompt_agent", "prompt": "hi",
        "snapshot_config": {"interval_seconds": 90, "env_id": "env-1"},
    })
    assert step.snapshot_config == ps.SnapshotConfig(
        interval_seconds=90, env_id="env-1"
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"timeout_seconds": 0}, "timeout_seconds must be > 0"),
        ({"timeout_seconds": -5}, "timeout_seconds must be > 0"),
        ({"max_snapshots": 0}, "max_snapshots must be >= 1"),
        # Configured to snapshot, would capture nothing at all.
        ({"at_end": False}, "needs a positive interval_seconds"),
        ({"at_end": False, "interval_seconds": 0}, "needs a positive interval_seconds"),
    ],
)
def test_the_config_rejects_its_own_misconfiguration(kwargs, match):
    """Validation lives on the dataclass, so there is no way to build a step whose
    snapshot settings are individually invalid."""
    with pytest.raises(ValueError, match=match):
        ps.SnapshotConfig(**kwargs)


# ============================================ the id a capture addresses

@pytest.mark.asyncio
@pytest.mark.parametrize("context_id", [None, "restored-ctx"])
async def test_execute_gives_the_series_the_wire_id_it_passes_down(monkeypatch, context_id):
    """A capture addresses the agent by the id the turns actually use, so what
    `execute` mints must be what `_execute_conversation` puts on the wire — it
    derives `self.context_id or a2a_context_id or uuid4()` from the same value.

    `context_id` is allowed alongside snapshotting: it pins the wire id, which is
    how a conversation restored into a fresh agent is resumed (see
    `deploy_agent.agent_snapshot_target_context_id`).
    """
    _install(monkeypatch)
    built: dict = {}
    real = ps.SnapshotSeries
    monkeypatch.setattr(
        pa_mod, "SnapshotSeries", lambda **kw: (built.update(kw), real(**kw))[1]
    )
    seen: dict = {}

    async def conversation(self, context, *, a2a_context_id=None, series=None):
        seen["passed"] = a2a_context_id
        return context

    monkeypatch.setattr(PromptAgentTaskStep, "_execute_conversation", conversation)
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver",
        context_id=context_id, snapshot_config=ps.SnapshotConfig(),
    )
    await step.execute(context())

    assert built["a2a_context_id"] == seen["passed"]
    if context_id:
        assert built["a2a_context_id"] == context_id, "a pinned context_id must reach the wire verbatim"
    else:
        assert len(built["a2a_context_id"]) == 32, "otherwise a fresh session per run"


@pytest.mark.asyncio
@pytest.mark.parametrize("context_id", [None, "restored-ctx"])
async def test_the_series_id_is_what_reaches_the_wire(monkeypatch, context_id):
    """The invariant end to end: whatever `execute` hands the series is what
    `_execute_conversation` puts on `message.contextId`. Stubbing the conversation
    would hide the `or a2a_context_id` branch, so run the real one up to the send."""
    _install(monkeypatch)
    built: dict = {}
    real = ps.SnapshotSeries
    monkeypatch.setattr(
        pa_mod, "SnapshotSeries", lambda **kw: (built.update(kw), real(**kw))[1]
    )
    for fn in ("create_conversation", "add_a2a_task"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda **kw: {})
    sent: dict = {}

    class _Stop(Exception):
        pass

    async def fake_send(a2a_url, parts, message_id, wire_context_id, timeout_seconds):
        sent["context_id"] = wire_context_id
        raise _Stop  # everything after the send is irrelevant to this invariant

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", fake_send)
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver",
        context_id=context_id, snapshot_config=ps.SnapshotConfig(),
    )
    with pytest.raises(_Stop):
        await step.execute(context())

    assert sent["context_id"] == built["a2a_context_id"]


@pytest.mark.asyncio
async def test_captures_and_rows_both_use_the_wire_id(monkeypatch):
    """`source_context_id` is the wire id, matching what `snapshot_agent_state`
    records for a one-shot capture (it reads `PromptResponse.a2a_context_id`)."""
    ws = _install(monkeypatch)
    ctx = context()
    await _series(a2a_context_id="restored-ctx").finish(ctx)

    assert ws.context_ids == ["restored-ctx"]
    assert [r["source_context_id"] for r in _rows(ctx)] == ["restored-ctx"]


def test_rollouts_sharing_a_context_id_get_distinct_artifact_ids():
    """A pinned `context_id` is shared by every concurrent rollout, so artifact ids
    are scoped by the instance — deriving them from the context id, or from a
    prefix of anything built on it, would point every rollout at one artifact."""
    shared = "a-very-long-restored-context-id"  # > 16 chars, so a prefix collides
    a = _series(a2a_context_id=shared, instance_id="solve-run-1111111111111111", env_id="env-1")
    b = _series(a2a_context_id=shared, instance_id="solve-run-2222222222222222", env_id="env-1")
    # The token both the workspace and env universe ids are built from.
    assert a._discriminator != b._discriminator
    assert a.workspace_artifact_id != b.workspace_artifact_id


@pytest.mark.parametrize("interval", [None, -30])
@pytest.mark.asyncio
async def test_no_positive_cadence_starts_no_ticker(monkeypatch, interval):
    ws = _install(monkeypatch)
    series = _series(interval_seconds=interval)
    ctx = context()
    series.start(ctx)
    # A negative interval must not start a ticker: it would reach asyncio.wait_for
    # as a timeout and fire immediately, then every loop iteration after.
    assert series._ticker is None
    await series.finish(ctx)
    assert ws.calls == 1 and [r["is_final"] for r in _rows(ctx)] == [True]


# ============================================================== ticker rules

@pytest.mark.asyncio
async def test_ticks_are_recorded_in_order_with_the_final_capture_last(monkeypatch):
    ws = _install(monkeypatch)
    series = _series(interval_seconds=0.01)
    ctx = context()
    series.start(ctx)
    await _until(lambda: ws.calls >= 2, what="two interior captures")
    await series.finish(ctx)

    rows = _rows(ctx)
    assert len(rows) >= 3
    assert [r["is_final"] for r in rows] == [False] * (len(rows) - 1) + [True]
    assert {r["source_context_id"] for r in rows} == {"cid-1"}
    assert {r["source_agent_name"] for r in rows} == {"solver"}


@pytest.mark.asyncio
async def test_a_capture_slower_than_the_interval_delays_the_next_tick(monkeypatch):
    """The cadence is the gap BETWEEN captures, so a capture that outruns the
    interval never overlaps the next one — it delays it."""
    gate = asyncio.Event()
    ws = _install(monkeypatch, _FakeWorkspace(gate=gate))
    series = _series(interval_seconds=0.01)
    ctx = context()
    series.start(ctx)

    await _until(lambda: ws.calls == 1, what="the first capture to start")
    await asyncio.sleep(0.08)  # many intervals' worth
    assert ws.calls == 1, "a second capture must not start while one is in flight"
    # Against the series, not the context: nothing is published until teardown, so
    # asserting on the context here would pass for the wrong reason.
    assert _pending(series) == [], "and nothing is recorded until it returns"

    gate.set()
    await _until(lambda: ws.calls >= 2, what="the next capture once the first returned")
    await series.finish(ctx)


@pytest.mark.asyncio
async def test_teardown_awaits_the_in_flight_capture_instead_of_cancelling_it(monkeypatch):
    gate = asyncio.Event()
    ws = _install(monkeypatch, _FakeWorkspace(gate=gate))
    series = _series(interval_seconds=0.01)
    ctx = context()
    series.start(ctx)
    await _until(lambda: ws.calls == 1, what="the interior capture to start")

    finishing = asyncio.create_task(series.finish(ctx))
    await asyncio.sleep(0.05)
    # Cancelling would stop our client but not the sidecar, which keeps tarring
    # and orphans an object after we believed the capture finished.
    assert not finishing.done(), "teardown must wait for the in-flight capture"
    gate.set()
    await finishing

    rows = _rows(ctx)
    interior = [r for r in rows if not r["is_final"] and r["capture_status"] == "ok"]
    assert interior, "the awaited interior capture must still be recorded"
    assert rows[-1]["is_final"] is True


@pytest.mark.asyncio
async def test_the_ticker_exits_on_stop_rather_than_sleeping_out_the_interval(monkeypatch):
    _install(monkeypatch)
    series = _series(interval_seconds=3600)
    ctx = context()
    series.start(ctx)
    await asyncio.sleep(0)  # let the ticker park on stop.wait()

    started = time.monotonic()
    await series.finish(ctx)
    assert time.monotonic() - started < 1.0
    # Retired, not merely cancel()-scheduled: a pending task would outlive
    # `execute` holding this rollout's context.
    assert series._ticker.done()


@pytest.mark.asyncio
async def test_max_snapshots_bounds_interior_captures_but_reserves_the_final_one(monkeypatch):
    ws = _install(monkeypatch)
    series = _series(interval_seconds=0.01, max_snapshots=2)
    ctx = context()
    ctx.prompt_responses.append(
        PromptResponse(prompt_id="p1", response="ok", agent_trajectory_s3_uri="s3://b/t.json")
    )
    series.start(ctx)
    await _until(
        lambda: any(r.get("capture_reason") == "limit_reached" for r in _pending(series)),
        what="the interior limit",
    )
    interior_at_limit = ws.calls
    await series.finish(ctx)

    assert interior_at_limit == 2, "the cap bounds attempts, not successes"
    # The final capture is exempt: counting it here would let ticks suppress the
    # grade of record.
    final = [r for r in _rows(ctx) if r["is_final"]]
    assert len(final) == 1 and final[0]["capture_status"] == "ok"
    assert ws.calls == 3


@pytest.mark.asyncio
async def test_a_ticker_crash_records_a_row_rather_than_going_silent(monkeypatch):
    _install(monkeypatch)
    series = _series(interval_seconds=0.01)
    ctx = context()

    async def boom(context):
        raise RuntimeError("ticker died")

    monkeypatch.setattr(series, "_ticks", boom)
    series.start(ctx)
    await _until(
        lambda: any("ticker_error" in (r.get("capture_reason") or "") for r in _pending(series)),
        what="the ticker failure row",
    )
    await series.finish(ctx)


# ============================================================== capture rows

@pytest.mark.asyncio
async def test_a_failing_interior_capture_is_recorded_not_fatal(monkeypatch):
    _install(monkeypatch, _FakeWorkspace(fail_after=0))
    series = _series(interval_seconds=0.01)
    ctx = context()
    series.start(ctx)
    await _until(lambda: _pending(series), what="a failure row")

    row = _pending(series)[0]
    assert row["capture_status"] == "failed"
    assert "sidecar exploded" in row["capture_reason"]
    assert "bundle_object_url" not in row  # nothing landed → not gradable
    series._stop.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_failure_after_the_bundle_landed_degrades_to_partial(monkeypatch):
    _install(monkeypatch, trajectory_reason=None)
    series = _series()
    ctx = context()

    def explode(traj, prefix):
        raise RuntimeError("upload died")

    monkeypatch.setattr(mod, "upload_trajectory", explode)
    row = await series._capture_never_raising(ctx, is_final=False)

    # The bundle is what makes a point gradable, so a lost auxiliary read must
    # degrade rather than discard — reporting `failed` here once failed the FINAL
    # capture and made the runner re-run an already-completed agent run.
    assert row["capture_status"] == "partial"
    assert row["capture_reason"] == "trajectory_upload_failed"
    assert row["bundle_object_url"]


@pytest.mark.asyncio
async def test_a_capture_timeout_degrades_the_row(monkeypatch):
    gate = asyncio.Event()  # never set
    _install(monkeypatch, _FakeWorkspace(gate=gate))
    series = _series(timeout_seconds=0.05)
    ctx = context()
    row = await series._capture_never_raising(ctx, is_final=True)
    assert row["capture_status"] == "failed"
    assert "capture_timeout" in row["capture_reason"]


@pytest.mark.asyncio
async def test_an_unsupported_trajectory_read_still_yields_a_gradable_row(monkeypatch):
    _install(monkeypatch, trajectory_reason="trajectory_context_unsupported")
    ctx = context()
    row = await _series()._capture_never_raising(ctx, is_final=False)
    # A card predating the context_id mode costs the trajectory column, not the
    # whole curve.
    assert row["capture_status"] == "partial"
    assert row["capture_reason"] == "trajectory_context_unsupported"
    # Nulls are stripped on append, not here — this is the raw row.
    assert row["bundle_object_url"] and row["trajectory_s3_uri"] is None


@pytest.mark.asyncio
async def test_the_final_row_reads_the_trajectory_cumulatively_like_the_others(monkeypatch):
    """`PromptResponse.agent_trajectory_s3_uri` is only the LAST turn's, so using
    it here would make the final point narrower than its predecessors."""
    _install(monkeypatch)
    ctx = context()
    ctx.prompt_responses.append(
        PromptResponse(prompt_id="p1", response="done", a2a_context_id="cid-1",
                       agent_trajectory_s3_uri="s3://b/last-turn-only.json")
    )
    await _series().finish(ctx)

    (row,) = _rows(ctx)
    assert row["trajectory_s3_uri"] == f"{TRAJ_PREFIX}trajectory-x.json"
    assert row["capture_status"] == "ok"


@pytest.mark.asyncio
async def test_the_final_row_falls_back_to_the_recorded_uri(monkeypatch):
    # On a failed run the live read can be gone while the per-turn upload landed.
    _install(monkeypatch, trajectory_reason="trajectory_session_missing")
    ctx = context()
    ctx.prompt_responses.append(
        PromptResponse(prompt_id="p1", response="done", a2a_context_id="cid-1",
                       agent_trajectory_s3_uri="s3://b/t.json")
    )
    await _series().finish(ctx)

    (row,) = _rows(ctx)
    assert row["trajectory_s3_uri"] == "s3://b/t.json"
    assert row["capture_status"] == "partial"
    assert row["capture_reason"] == "trajectory_session_missing"


@pytest.mark.asyncio
async def test_a_sibling_attempts_trajectory_is_never_borrowed(monkeypatch):
    """A retry can restore a context snapshot taken during an earlier attempt, so
    that attempt's response for the same prompt_id can still be in the list."""
    _install(monkeypatch, trajectory_reason="trajectory_session_missing")
    ctx = context()
    ctx.prompt_responses.append(
        PromptResponse(prompt_id="p1", response="old", a2a_context_id="cid-PREVIOUS",
                       agent_trajectory_s3_uri="s3://b/previous-attempt.json")
    )
    await _series().finish(ctx)

    (row,) = _rows(ctx)
    assert "trajectory_s3_uri" not in row, "matched on prompt_id alone, not the conversation"
    assert row["capture_reason"] == "trajectory_session_missing; trajectory_missing"


@pytest.mark.asyncio
async def test_rows_carry_no_null_keys(monkeypatch):
    _install(monkeypatch, trajectory_reason="trajectory_empty")
    ctx = context()
    series = _series()
    await series.finish(ctx)
    for row in _rows(ctx):
        assert None not in row.values(), f"null-valued key in {row}"
    # Every row still carries the five keys the existing manifest contract has.
    assert {"id", "version", "bundle_object_url", "source_agent_name", "source_context_id"} <= set(
        _rows(ctx)[0]
    )


@pytest.mark.asyncio
async def test_the_reads_share_one_capture_budget(monkeypatch):
    """Each read gets what's LEFT of the budget, not a fresh copy of it —
    handing each the full timeout would leave the outer wait_for unable to bound
    the capture as a whole."""
    seen: list[float] = []

    async def slow_workspace(**kwargs):
        seen.append(kwargs["timeout_seconds"])
        await asyncio.sleep(0.1)
        return mod.WorkspaceCapture(
            universe_id="wsp", universe_version=1,
            bundle_object_url=f"s3://{BUCKET}/b/", capture_prefix=f"s3://{BUCKET}/b/",
        )

    async def traj(**kwargs):
        seen.append(kwargs["timeout_seconds"])
        return mod.TrajectoryCapture(trajectory=[{"role": "user"}])

    monkeypatch.setattr(mod, "capture_workspace", slow_workspace)
    monkeypatch.setattr(mod, "read_partial_trajectory", traj)
    monkeypatch.setattr(mod, "upload_trajectory", lambda t, p: f"{p}t.json")

    # Budget well above the 1s floor `remaining()` clamps to, so the remainder is
    # observable rather than swallowed by that floor.
    row = await _series(timeout_seconds=5)._capture_never_raising(context(), is_final=False)

    assert row["capture_status"] == "ok"
    workspace_budget, trajectory_budget = seen
    assert workspace_budget == pytest.approx(5.0, abs=0.05)
    assert trajectory_budget < workspace_budget - 0.05, (
        f"the trajectory read got {trajectory_budget}s of a {workspace_budget}s budget "
        "with 0.1s already spent — it was handed a fresh copy, not the remainder"
    )


# ============================================================== env capture

def _install_env(monkeypatch, *, fail=False, base_version=4):
    import agent_env.env.env as env_mod
    import agent_env.task_step.task_steps.snapshot_env as snap_env_mod

    class _FakeEnv:
        id = "env-1"
        version = 1

    monkeypatch.setattr(
        env_mod.Env, "get", classmethod(lambda cls, id, version=None: _FakeEnv())
    )
    calls: list[dict] = []

    async def fake_snapshot_env_state(**kwargs):
        calls.append(kwargs)
        if fail:
            # snapshot_env is all-or-nothing: one unreachable service raises.
            raise RuntimeError("1/13 services were not exported (missing=['slack'])")
        return snap_env_mod.EnvSnapshotResult(
            # Echoed, like the real one: it versions into the id it was handed.
            environment_universe_artifact_id=kwargs["snapshot_id"],
            environment_universe_artifact_version=base_version + len(calls) - 1,
            environments_snapshotted=["slack"],
            total=1,
        )

    monkeypatch.setattr(
        snap_env_mod.SnapshotEnvTaskStep,
        "snapshot_env_state",
        staticmethod(fake_snapshot_env_state),
    )
    return calls


@pytest.mark.asyncio
async def test_env_state_lands_as_a_versioned_service_universe_on_the_row(monkeypatch):
    _install(monkeypatch)
    calls = _install_env(monkeypatch)
    ctx = context(env_id="env-1")
    ctx.prompt_responses.append(PromptResponse(prompt_id="p1", response="ok"))
    series = _series(env_id="env-1", interval_seconds=0.01)
    series.start(ctx)
    await _until(lambda: len(calls) >= 2, what="two env captures")
    await series.finish(ctx)

    rows = [r for r in _rows(ctx) if r.get("env_universe_id")]
    assert len(rows) >= 2
    # Two scalars per row, not a per-service URL map a long series would grow
    # without bound.
    assert all(r["env_universe_id"] == f"snapshot-env-1-{DISCRIMINATOR}" for r in rows)
    versions = [r["env_universe_version"] for r in rows]
    assert versions == sorted(versions) and len(set(versions)) == len(versions)
    # One artifact id per rollout, scoped to the instance.
    assert {c["snapshot_id"] for c in calls} == {f"snapshot-env-1-{DISCRIMINATOR}"}


@pytest.mark.asyncio
async def test_the_final_capture_publishes_the_universe_for_load_artifact(monkeypatch):
    _install(monkeypatch)
    _install_env(monkeypatch)
    ctx = context(env_id="env-1")
    ctx.prompt_responses.append(PromptResponse(prompt_id="p1", response="ok"))
    await _series(env_id="env-1").finish(ctx)

    # Keyed on the step id, so a judge step can be wired to this prompt step
    # without knowing it snapshots. `load_artifact` resolves it.
    assert ctx.metadata["env_snapshotted_universes"]["solve"] == {
        "id": f"snapshot-env-1-{DISCRIMINATOR}",
        "version": 4,
    }


@pytest.mark.asyncio
async def test_an_env_export_failure_degrades_the_row_but_keeps_the_bundle(monkeypatch):
    _install(monkeypatch)
    _install_env(monkeypatch, fail=True)
    ctx = context(env_id="env-1")
    ctx.prompt_responses.append(PromptResponse(prompt_id="p1", response="ok"))
    await _series(env_id="env-1").finish(ctx)

    (row,) = _rows(ctx)
    assert row["capture_status"] == "partial"
    assert "env_capture_failed" in row["capture_reason"]
    assert row["bundle_object_url"], "a lost env leg must not cost the workspace point"
    assert "env_universe_id" not in row
    # Still a gradable end state, so not fatal.
    _series(env_id="env-1").raise_if_final_capture_missing()


@pytest.mark.asyncio
async def test_an_undeployed_env_id_is_a_reason_not_a_raise(monkeypatch):
    _install(monkeypatch)
    ctx = context()  # no deployed envs
    ctx.prompt_responses.append(
        PromptResponse(prompt_id="p1", response="ok", agent_trajectory_s3_uri="s3://b/t.json")
    )
    await _series(env_id="env-missing").finish(ctx)

    (row,) = _rows(ctx)
    assert row["capture_status"] == "partial"
    assert row["capture_reason"] == "env_not_deployed"
    assert row["bundle_object_url"]


# ============================================================== fatality

@pytest.mark.asyncio
async def test_a_failed_final_capture_fails_the_step(monkeypatch):
    _install(monkeypatch, _FakeWorkspace(fail_after=0))
    ctx = context()
    series = _series()
    await series.finish(ctx)

    assert _rows(ctx)[-1]["capture_status"] == "failed"
    with pytest.raises(RuntimeError, match="final capture did not land"):
        series.raise_if_final_capture_missing()


@pytest.mark.asyncio
async def test_a_partial_final_row_with_a_bundle_is_not_fatal(monkeypatch):
    _install(monkeypatch, trajectory_reason="trajectory_session_missing")
    ctx = context()
    ctx.prompt_responses.append(PromptResponse(prompt_id="p1", response="ok"))
    series = _series()
    await series.finish(ctx)

    assert _rows(ctx)[-1]["capture_status"] == "partial"
    # Gradability keys on the landed bundle, never on capture_status.
    series.raise_if_final_capture_missing()


@pytest.mark.asyncio
async def test_snapshot_at_end_false_captures_nothing_and_is_never_fatal(monkeypatch):
    ws = _install(monkeypatch)
    ctx = context()
    # Never started, so the cadence the config now requires alongside at_end=False
    # never produces a tick either.
    series = _series(at_end=False, interval_seconds=3600)
    await series.finish(ctx)

    assert ws.calls == 0 and _rows(ctx) == []
    series.raise_if_final_capture_missing()
    # Teardown must not walk away still holding the lock it took to drain.
    assert not series._lock.locked()


@pytest.mark.asyncio
async def test_snapshot_at_end_false_still_drains_the_in_flight_capture(monkeypatch):
    gate = asyncio.Event()
    ws = _install(monkeypatch, _FakeWorkspace(gate=gate))
    series = _series(interval_seconds=0.01, at_end=False)
    ctx = context()
    series.start(ctx)
    await _until(lambda: ws.calls == 1, what="the interior capture to start")

    finishing = asyncio.create_task(series.finish(ctx))
    await asyncio.sleep(0.03)
    assert not finishing.done(), "teardown must drain rather than close under the capture"
    gate.set()
    await finishing

    # Its row survives: closing under it would drop it as a late append and
    # orphan both the S3 object and the universe version it burnt.
    assert [r["is_final"] for r in _rows(ctx)] == [False]


@pytest.mark.asyncio
async def test_a_lock_timeout_records_a_skipped_final_row_and_is_fatal(monkeypatch):
    # A genuinely independent stuck lock — NOT a gated interior capture, whose
    # deadline is the same value and would make this assert the very race that
    # `_FINAL_LOCK_GRACE_SECONDS` exists to prevent.
    _install(monkeypatch)
    monkeypatch.setattr(ps, "_FINAL_LOCK_GRACE_SECONDS", 0.05)
    series = _series(timeout_seconds=1)
    ctx = context()
    await series._lock.acquire()  # held by nobody the series can wait out

    await series.finish(ctx)

    final = [r for r in _rows(ctx) if r["is_final"]]
    assert final and final[0]["capture_reason"] == "final_capture_lock_timeout"
    with pytest.raises(RuntimeError, match="final capture did not land"):
        series.raise_if_final_capture_missing()


@pytest.mark.asyncio
async def test_teardown_outlasts_an_interior_capture_that_times_out(monkeypatch):
    """The interior capture and the final lock wait must not share a deadline: an
    equal bound let both fire in one loop iteration, recording a fatal `skipped`
    final row for a rollout whose agent had already completed."""
    gate = asyncio.Event()  # the INTERIOR capture never returns on its own
    _install(monkeypatch, _FakeWorkspace(gate=gate, gate_only_first=True))
    series = _series(interval_seconds=0.01, timeout_seconds=0.05)
    ctx = context()
    ctx.prompt_responses.append(
        PromptResponse(prompt_id="p1", response="ok", a2a_context_id="cid-1",
                       agent_trajectory_s3_uri="s3://b/t.json")
    )
    series.start(ctx)
    await _until(lambda: series._lock.locked(), what="the interior capture")

    await series.finish(ctx)

    # The interior capture times out, releases, and the final one still lands.
    final = [r for r in _rows(ctx) if r["is_final"]]
    assert final and final[0]["bundle_object_url"], final
    series.raise_if_final_capture_missing()
    gate.set()


@pytest.mark.asyncio
async def test_cancellation_neither_waits_for_the_lock_nor_starts_a_capture(monkeypatch):
    """Both are bounded by `timeout_seconds`, so honouring them under cancellation
    would postpone it by up to twice that — 10 minutes at defaults."""
    gate = asyncio.Event()  # an interior capture holds the lock forever
    ws = _install(monkeypatch, _FakeWorkspace(gate=gate))
    series = _series(interval_seconds=0.01, timeout_seconds=300)
    ctx = context()
    series.start(ctx)
    await _until(lambda: series._lock.locked(), what="the interior capture")

    # `entered` is set INSIDE the try, so cancelling can never land before the
    # finally is armed — polling the lock from out here could.
    entered = asyncio.Event()

    async def run():
        try:
            entered.set()
            await asyncio.sleep(3600)
        finally:
            await series.finish(ctx)

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    started = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"teardown took {elapsed:.1f}s; it waited on the 300s budget"
    assert ws.calls == 1, "no second capture may start once cancellation is in flight"
    final = [r for r in _rows(ctx) if r["is_final"]]
    assert final and final[0]["capture_reason"] == "cancelled"
    gate.set()


@pytest.mark.asyncio
async def test_cancellation_arriving_mid_teardown_still_forces_the_ticker_down(monkeypatch):
    """The cancelling flag is sampled when `finish()` starts, so a cancellation that
    lands during the lock wait leaves it stale — and a stale `False` lets a
    mid-capture ticker outlive teardown, persisting an unreferenced capture."""
    gate = asyncio.Event()  # the interior capture holds the lock indefinitely
    _install(monkeypatch, _FakeWorkspace(gate=gate))
    series = _series(interval_seconds=0.01, timeout_seconds=300)
    ctx = context()
    series.start(ctx)
    await _until(lambda: series._lock.locked(), what="the interior capture")

    entered = asyncio.Event()

    async def run():
        entered.set()
        # No cancellation yet, so `finish` samples False and waits for the lock.
        await series.finish(ctx)

    task = asyncio.create_task(run())
    await entered.wait()
    await asyncio.sleep(0)          # let it reach the lock wait
    task.cancel()                   # cancellation arrives mid-teardown
    with pytest.raises(asyncio.CancelledError):
        await task

    assert series._ticker.done(), "a stale cancelling flag left the ticker alive"
    gate.set()


def _slow_to_die(started: asyncio.Event):
    """A ticker whose cleanup outlives its own cancellation, so a parent awaiting it
    is still parked there and can be cancelled at that await."""

    async def ticker():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            raise

    return ticker()


@pytest.mark.asyncio
async def test_retiring_swallows_the_tickers_own_cancellation(monkeypatch):
    _install(monkeypatch)
    series = _series()
    started = asyncio.Event()
    series._ticker = asyncio.create_task(_slow_to_die(started))
    await started.wait()

    # Nobody cancelled US, so the ticker's cancellation is ours to absorb.
    await series._retire_ticker(force=True)
    assert series._ticker.cancelled()


@pytest.mark.asyncio
async def test_retiring_the_ticker_does_not_swallow_a_parent_cancellation(monkeypatch):
    """Real cancellation propagates INTO the awaited ticker, so `ticker.cancelled()`
    reads True for BOTH sources — two real tasks are the only way to see this."""
    _install(monkeypatch)
    series = _series()
    started = asyncio.Event()
    series._ticker = asyncio.create_task(_slow_to_die(started))
    await started.wait()

    parent = asyncio.create_task(series._retire_ticker(force=True))
    await asyncio.sleep(0.01)  # parent is now parked on `await self._ticker`
    parent.cancel()

    # Must propagate: swallowing it silently completes a cancelled step.
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert series._ticker.cancelled(), "true for both sources — hence the counter"


@pytest.mark.asyncio
async def test_a_cancellation_during_retirement_still_publishes_the_rows(monkeypatch):
    """Retirement re-raises a cancellation aimed at us, so publication must not sit
    after that await in the same `finally` — the rows live nowhere else."""
    _install(monkeypatch)
    series = _series(at_end=False, interval_seconds=3600)
    ctx = context()
    series._append(series._row(is_final=False, status="ok"))
    collected = len(series._rows)

    started = asyncio.Event()
    series._ticker = asyncio.create_task(_slow_to_die(started))
    await started.wait()

    finishing = asyncio.create_task(series.finish(ctx))
    await asyncio.sleep(0.02)  # parked inside `_retire_ticker`'s await
    finishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finishing

    assert len(_rows(ctx)) == collected, "a cancelled teardown must still publish"


@pytest.mark.asyncio
async def test_a_tick_that_wins_the_lock_after_teardown_captures_nothing(monkeypatch):
    """Teardown can set `stop` and take the lock while a tick waits on it. A capture
    started afterwards would burn an S3 object and a universe version for a row
    teardown has already published past — so the tick must recheck inside the lock."""
    ws = _install(monkeypatch)
    series = _series(interval_seconds=0.01)
    ctx = context()
    await series._lock.acquire()  # stand in for teardown holding it

    ticking = asyncio.create_task(series._ticks(ctx))
    await asyncio.sleep(0.05)     # the tick fires and blocks on the lock
    assert ws.calls == 0

    series._stop.set()            # teardown stops the series, then drops the lock
    series._lock.release()
    await ticking

    assert ws.calls == 0, "a tick must recheck stop after winning the lock"
    assert _pending(series) == [], "and records nothing"


@pytest.mark.asyncio
async def test_rows_reach_the_context_only_at_teardown(monkeypatch):
    """The ticker collects onto the series, so a capture racing teardown cannot
    land after the ``is_final`` row — there is no late-append case to detect."""
    _install(monkeypatch)
    ctx = context()
    ctx.prompt_responses.append(PromptResponse(prompt_id="p1", response="ok"))
    series = _series(interval_seconds=0.01)
    series.start(ctx)
    await _until(lambda: _pending(series), what="an interior row on the series")
    assert _rows(ctx) == [], "nothing is published while the run is in flight"

    await series.finish(ctx)
    assert _rows(ctx) == _pending(series), "published verbatim, exactly once"
    assert _rows(ctx)[-1]["is_final"] is True


# ============================================================== execute wiring

@pytest.mark.asyncio
async def test_execute_finishes_the_series_even_when_the_conversation_raises(monkeypatch):
    _install(monkeypatch)
    step = _snapshotting_step(interval_seconds=3600)

    async def exploding_conversation(self, context, *, a2a_context_id=None, series=None):
        raise RuntimeError("A2A task failed (task_id=abc): boom")

    monkeypatch.setattr(PromptAgentTaskStep, "_execute_conversation", exploding_conversation)
    ctx = context()
    ctx.prompt_responses.append(
        PromptResponse(prompt_id=step.prompt_id, response="", agent_trajectory_s3_uri="s3://b/t")
    )

    # A long run that died is the most interesting point for a curve, so the
    # capture still happens — but the agent's error is what propagates.
    with pytest.raises(RuntimeError, match="A2A task failed"):
        await step.execute(ctx)
    (row,) = _rows(ctx)
    assert row["is_final"] is True and row["capture_status"] == "ok"


@pytest.mark.asyncio
async def test_a_failed_final_capture_never_masks_the_agents_own_failure(monkeypatch):
    _install(monkeypatch, _FakeWorkspace(fail_after=0))
    step = _snapshotting_step()

    async def exploding_conversation(self, context, *, a2a_context_id=None, series=None):
        raise RuntimeError("A2A task failed (task_id=abc): boom")

    monkeypatch.setattr(PromptAgentTaskStep, "_execute_conversation", exploding_conversation)
    ctx = context()
    # The final capture ALSO fails here, so the fatality check would raise — but
    # it runs after the finally, so the agent's own error is what surfaces.
    with pytest.raises(RuntimeError, match="A2A task failed"):
        await step.execute(ctx)
    assert _rows(ctx)[-1]["capture_status"] == "failed"


@pytest.mark.asyncio
async def test_execute_raises_when_a_successful_run_captured_nothing(monkeypatch):
    _install(monkeypatch, _FakeWorkspace(fail_after=0))
    step = _snapshotting_step()

    async def ok_conversation(self, context, *, a2a_context_id=None, series=None):
        return context

    monkeypatch.setattr(PromptAgentTaskStep, "_execute_conversation", ok_conversation)
    # Otherwise the instance reports `completed`, consumes a pass@k target slot
    # and contributes no gradable point.
    with pytest.raises(RuntimeError, match="no gradable end state"):
        await step.execute(context())


@pytest.mark.asyncio
async def test_execute_derives_the_workspace_artifact_id_from_the_instance(monkeypatch):
    """Nobody names the artifact: it is derived per rollout, so concurrent
    rollouts never allocate versions of one id."""
    ws = _install(monkeypatch)
    step = _snapshotting_step()

    async def ok_conversation(self, context, *, a2a_context_id=None, series=None):
        return context

    monkeypatch.setattr(PromptAgentTaskStep, "_execute_conversation", ok_conversation)
    ctx = context()
    ctx.instance_id = "solve-0123456789abcdef01"
    await step.execute(ctx)

    # Last dash-segment of the instance id, truncated to 16 — as snapshot_env does.
    assert ws.artifact_ids == ["solve-workspace-0123456789abcdef"]


# ============================================================== concurrency

@pytest.mark.asyncio
async def test_concurrent_rollouts_on_one_step_instance_keep_separate_series(monkeypatch):
    """`agent-env task run --k N` runs k rollouts through ONE step instance, so
    per-rollout state lives on the series (a local), never on the step."""
    ws = _install(monkeypatch)
    step = _snapshotting_step(interval_seconds=0.01)

    async def conversation(self, context, *, a2a_context_id=None, series=None):
        await asyncio.sleep(0.08 if context.instance_id.endswith("slow") else 0.01)
        return context

    monkeypatch.setattr(PromptAgentTaskStep, "_execute_conversation", conversation)

    async def one(instance_id):
        ctx = context()
        ctx.instance_id = instance_id
        ctx.prompt_responses.append(PromptResponse(prompt_id=step.prompt_id, response="ok"))
        await step.execute(ctx)
        return ctx

    fast, slow = await asyncio.gather(one("solve-fast"), one("solve-slow"))

    # One conversation per rollout, and neither bleeds into the other's rows.
    fast_cids = {r["source_context_id"] for r in _rows(fast)}
    slow_cids = {r["source_context_id"] for r in _rows(slow)}
    assert len(fast_cids) == len(slow_cids) == 1
    assert fast_cids.isdisjoint(slow_cids)
    assert [r["is_final"] for r in _rows(fast)][-1] is True
    assert [r["is_final"] for r in _rows(slow)][-1] is True
    # The slow rollout got more ticks; a shared ticker would have interleaved them.
    assert len(_rows(slow)) > len(_rows(fast))
    # And each rollout derived its own artifact, so no two captures across the two
    # rollouts ever allocate a version of the same id.
    assert set(ws.artifact_ids) == {"solve-workspace-fast", "solve-workspace-slow"}


# ============================================================== turn boundaries

def test_per_turn_round_trips_and_is_a_cadence_of_its_own():
    """`per_turn` alone satisfies the "would capture nothing" guard: a turn boundary
    is a capture point, so no interval is required alongside it."""
    cfg = ps.SnapshotConfig(at_end=False, per_turn=True)
    step = PromptAgentTaskStep(
        id="s", version=None, prompt="hi", snapshot_config=cfg
    )
    assert PromptAgentTaskStep.from_dict(step.to_dict()).snapshot_config == cfg
    # A document written before `per_turn` existed keeps the old behaviour.
    assert ps.SnapshotConfig.from_dict({"interval_seconds": 90}).per_turn is False


@pytest.mark.asyncio
async def test_capture_turn_is_inert_unless_per_turn_is_set(monkeypatch):
    ws = _install(monkeypatch)
    series = _series()  # at_end only
    ctx = context()
    await series.capture_turn(ctx, 1)

    assert ws.calls == 0 and _pending(series) == []


@pytest.mark.asyncio
async def test_a_turn_capture_records_the_turn_it_closed(monkeypatch):
    ws = _install(monkeypatch)
    series = _series(per_turn=True)
    ctx = context()
    await series.capture_turn(ctx, 1)
    await series.capture_turn(ctx, 2)
    await series.finish(ctx)

    rows = _rows(ctx)
    assert [r.get("turn") for r in rows] == [1, 2, None], "the final row is not a turn"
    assert [r["is_final"] for r in rows] == [False, False, True]
    assert all(r["capture_status"] == "ok" for r in rows)
    assert ws.calls == 3


@pytest.mark.asyncio
async def test_turn_captures_share_the_interior_budget_and_refuse_once(monkeypatch):
    """Bounded by the same `max_snapshots` as the ticker — a conversation
    can run for hundreds of turns, and each capture with env state mints artifact
    versions per service."""
    ws = _install(monkeypatch)
    series = _series(per_turn=True, max_snapshots=2)
    ctx = context()
    for turn in range(1, 5):
        await series.capture_turn(ctx, turn)
    await series.finish(ctx)

    rows = _rows(ctx)
    captured = [r["turn"] for r in rows if r["capture_status"] == "ok" and not r["is_final"]]
    refused = [r for r in rows if r.get("capture_reason") == "limit_reached"]
    assert captured == [1, 2]
    # One row for the refusal, not one per remaining turn.
    assert len(refused) == 1 and refused[0]["turn"] == 3
    assert ws.calls == 3, "two interior captures plus the exempt final one"


@pytest.mark.asyncio
async def test_a_turn_capture_after_teardown_captures_nothing(monkeypatch):
    """`finish` publishes, so a capture landing after it would be dropped as a late
    append while still having burnt an S3 object and a universe version."""
    ws = _install(monkeypatch)
    series = _series(per_turn=True)
    ctx = context()
    await series.finish(ctx)
    await series.capture_turn(ctx, 1)

    assert ws.calls == 1, "only the final capture ran"
    assert [r["is_final"] for r in _rows(ctx)] == [True]


@pytest.mark.asyncio
async def test_a_failing_turn_capture_costs_a_row_not_the_conversation(monkeypatch):
    _install(monkeypatch, _FakeWorkspace(fail_after=0))
    series = _series(per_turn=True)
    ctx = context()
    await series.capture_turn(ctx, 1)  # must not raise into the turn loop

    (row,) = _pending(series)
    assert row["turn"] == 1 and row["capture_status"] == "failed"


@pytest.mark.asyncio
async def test_a_turn_capture_and_the_ticker_never_overlap(monkeypatch):
    """They share the lock, so the sidecar is never asked to tar one workspace
    twice concurrently."""
    gate = asyncio.Event()
    ws = _install(monkeypatch, _FakeWorkspace(gate=gate))
    series = _series(per_turn=True, interval_seconds=0.01)
    ctx = context()
    series.start(ctx)
    await _until(lambda: ws.calls == 1, what="the ticker's capture to start")

    turning = asyncio.create_task(series.capture_turn(ctx, 1))
    await asyncio.sleep(0.05)
    assert ws.calls == 1, "the turn capture must wait for the in-flight one"
    gate.set()
    await turning
    await series.finish(ctx)


# ================================================ turn boundaries, end to end

class _Conversation:
    """The real turn loop with only its two boundaries stubbed: the A2A wire and the
    conversation store. Where `capture_turn` is called from is the whole feature, so
    stubbing `_execute_conversation` would test nothing."""

    TARGET_URL = "https://agent"
    USER_URL = "https://usersim"

    def __init__(self, monkeypatch, *, user_done_at: int | None = None):
        self.user_done_at = user_done_at
        self.user_turns = 0

        for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task",
                   "mark_closed"):
            # `*a` too: the loop calls some of these positionally.
            monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: {})
        monkeypatch.setattr(
            pa_mod.conversation_store, "get_conversation",
            lambda *a, **kw: {"status": "active"},
        )

        async def send(url, parts, message_id, wire_context_id, timeout_seconds):
            return f"{url}#{message_id}", None

        async def poll(url, task_id, timeout_seconds, poll_interval_seconds):
            if url == self.USER_URL:
                self.user_turns += 1
                done = self.user_done_at == self.user_turns
                text = f'{{"message": "and then?", "done": {str(done).lower()}}}'
            else:
                text = "agent worked"
            return {"status": {"state": "completed",
                               "message": {"parts": [{"kind": "text", "text": text}]}}}

        monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
        monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)

        class _Cfg:
            def get_model_params(self, overrides=None):
                return {}

            # Unreached while a user sim is deployed, but the loop falls back to it
            # when one is not — so the stub tracks the real API.
            def get_default_human_a2a_url(self):
                return _Conversation.USER_URL

        monkeypatch.setattr(pa_mod, "get_config", lambda: _Cfg())

    def context(self) -> TaskStepContext:
        from .capture_stubs import _StubAgent

        ctx = context(agent=_StubAgent(name="solver", a2a_url=self.TARGET_URL))
        ctx.deployed_agents.append(
            _StubAgent(name="human_agent", a2a_url=self.USER_URL)
        )
        ctx.instance_id = INSTANCE_ID
        return ctx


def _turn_step(turns: int, **cfg_overrides) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver",
        max_conversation_turns=turns, trajectory_output_prefix=TRAJ_PREFIX,
        snapshot_config=ps.SnapshotConfig(per_turn=True, **cfg_overrides),
    )


@pytest.mark.asyncio
async def test_every_turn_the_user_sim_answers_is_captured(monkeypatch):
    ws = _install(monkeypatch)
    conv = _Conversation(monkeypatch)
    ctx = conv.context()

    await _turn_step(3).execute(ctx)

    rows = _rows(ctx)
    # Turns 1 and 2 hand off to the user sim; turn 3 ends the conversation and is
    # the final capture — captured once, not twice.
    assert [r.get("turn") for r in rows] == [1, 2, None]
    assert [r["is_final"] for r in rows] == [False, False, True]
    assert ws.calls == 3
    assert conv.user_turns == 2


@pytest.mark.asyncio
async def test_a_turn_capture_precedes_the_user_sims_reply(monkeypatch):
    """The point of a turn boundary is that the agent has stopped writing, so the
    capture must land before the next prompt goes out — otherwise it races the
    agent's next turn and the "quiescent workspace" guarantee is gone."""
    order: list[str] = []

    class _Ordered(_FakeWorkspace):
        async def __call__(self, **kwargs):
            order.append("capture")
            return await super().__call__(**kwargs)

    _install(monkeypatch, _Ordered())
    conv = _Conversation(monkeypatch)
    real_send = pa_mod.protocol.send_a2a_message

    async def send(url, *args, **kwargs):
        order.append("send:user" if url == conv.USER_URL else "send:agent")
        return await real_send(url, *args, **kwargs)

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)

    await _turn_step(2).execute(conv.context())

    assert order == ["send:agent", "capture", "send:user", "send:agent", "capture"]


@pytest.mark.asyncio
async def test_the_turn_that_ends_the_conversation_is_only_the_final_capture(monkeypatch):
    """The user sim signalling done breaks the loop, and teardown's final capture
    covers that point — capturing at the boundary too would double up on it."""
    ws = _install(monkeypatch)
    conv = _Conversation(monkeypatch, user_done_at=1)
    ctx = conv.context()

    await _turn_step(5).execute(ctx)

    assert [r.get("turn") for r in _rows(ctx)] == [1, None]
    assert ws.calls == 2


@pytest.mark.asyncio
async def test_a_single_turn_step_has_no_boundary_to_capture(monkeypatch):
    ws = _install(monkeypatch)
    conv = _Conversation(monkeypatch)
    ctx = conv.context()

    await _turn_step(1).execute(ctx)

    assert [r["is_final"] for r in _rows(ctx)] == [True]
    assert ws.calls == 1 and conv.user_turns == 0


@pytest.mark.asyncio
async def test_per_turn_without_a_multi_turn_conversation_warns(monkeypatch, caplog):
    _install(monkeypatch)
    conv = _Conversation(monkeypatch)
    with caplog.at_level("WARNING"):
        await _turn_step(1).execute(conv.context())

    assert "no turn boundary to capture at" in caplog.text


@pytest.mark.asyncio
async def test_an_unconfigured_multi_turn_step_captures_nothing(monkeypatch):
    """The inertness guarantee, through the real loop: `prompt_agent` is the
    most-used step in the repo and the turn loop now has a call site in it."""
    ws = _install(monkeypatch)
    conv = _Conversation(monkeypatch)
    ctx = conv.context()

    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver",
        max_conversation_turns=3, trajectory_output_prefix=TRAJ_PREFIX,
    )
    await step.execute(ctx)

    assert ws.calls == 0 and "agent_snapshots" not in ctx.metadata


@pytest.mark.asyncio
async def test_a_turn_capture_rechecks_the_budget_under_the_lock(monkeypatch):
    """Both contenders check the budget before waiting for the lock, so the check
    and the increment must not be assumed atomic across that wait — see
    `SnapshotSeries._at_limit`. Forced by consuming the last slot while a turn
    capture is parked on the lock."""
    ws = _install(monkeypatch)
    series = _series(per_turn=True, max_snapshots=1)
    ctx = context()
    await series._lock.acquire()  # stand in for a capture in flight

    turning = asyncio.create_task(series.capture_turn(ctx, 1))
    await asyncio.sleep(0.01)  # past the cheap check, now parked on the lock
    series._attempts = 1       # a tick spent the last slot meanwhile
    series._lock.release()
    await turning

    assert ws.calls == 0, "the turn capture must not overshoot the cap"
    assert [r.get("capture_reason") for r in _pending(series)] == ["limit_reached"]


@pytest.mark.asyncio
async def test_a_tick_rechecks_the_budget_under_the_lock(monkeypatch):
    """The same recheck on the ticker's side, so neither contender can spend a slot
    the other already took."""
    ws = _install(monkeypatch)
    series = _series(per_turn=True, interval_seconds=0.01, max_snapshots=1)
    ctx = context()
    await series._lock.acquire()

    ticking = asyncio.create_task(series._ticks(ctx))
    await asyncio.sleep(0.05)  # the tick fires and blocks on the lock
    series._attempts = 1       # a turn boundary spent the last slot meanwhile
    series._lock.release()
    await ticking

    assert ws.calls == 0
    assert [r.get("capture_reason") for r in _pending(series)] == ["limit_reached"]


# ====================================================== the pre-rename spelling

def test_a_pre_rename_document_is_not_read():
    """The rename is a deliberate break, not a migration: `periodic_snapshot_config`
    and `max_periodic_snapshots` are ignored outright. Documented here because the
    failure is silent — a step written under the old key simply stops capturing."""
    step = PromptAgentTaskStep.from_dict({
        "id": "s", "type": "prompt_agent", "prompt": "hi",
        "periodic_snapshot_config": {"interval_seconds": 90, "max_periodic_snapshots": 5},
    })
    assert step.snapshot_config is None

    # Same inside the config: the old bound does not carry over.
    assert ps.SnapshotConfig.from_dict({"max_periodic_snapshots": 5}).max_snapshots == 24


def test_only_the_new_keys_are_written():
    doc = PromptAgentTaskStep(
        id="s", version=None, prompt="hi",
        snapshot_config=ps.SnapshotConfig(interval_seconds=60, max_snapshots=3),
    ).to_dict()
    assert "periodic_snapshot_config" not in doc
    assert doc["snapshot_config"]["max_snapshots"] == 3


@pytest.mark.asyncio
async def test_a_budget_below_the_turn_count_warns_up_front(monkeypatch, caplog):
    """Both cadences share one budget, so a long conversation stops being captured
    partway through while the run still succeeds. Warned at startup, since by the
    time anyone reads the rows the run is over."""
    _install(monkeypatch)
    conv = _Conversation(monkeypatch)
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver",
        max_conversation_turns=10, trajectory_output_prefix=TRAJ_PREFIX,
        snapshot_config=ps.SnapshotConfig(per_turn=True, max_snapshots=3),
    )
    with caplog.at_level("WARNING"):
        await step.execute(conv.context())

    assert "max_snapshots=3 is below the 9 turn boundaries" in caplog.text
    assert "captures stop after turn 3" in caplog.text


@pytest.mark.asyncio
async def test_a_budget_that_covers_the_conversation_does_not_warn(monkeypatch, caplog):
    _install(monkeypatch)
    conv = _Conversation(monkeypatch)
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver",
        max_conversation_turns=3, trajectory_output_prefix=TRAJ_PREFIX,
        snapshot_config=ps.SnapshotConfig(per_turn=True, max_snapshots=6),
    )
    with caplog.at_level("WARNING"):
        await step.execute(conv.context())

    assert "below the" not in caplog.text
