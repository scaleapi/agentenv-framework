"""Run dispatch + progress — the Task Runner surface.

Every route goes through the configured ``[runner]``. Responses use ``workflow_id`` /
``instance_id`` for the UI's poll loop; under the local runner ``workflow_id`` is a
locally-minted run id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from agent_env.config import get_config, get_runner
from agent_env.explorer.entity_ids import EntityId
from agent_env.explorer.routers.common import PaginatedResponse, docs
from agent_env.runner.runner import RunStatus
from agent_env.store import Filter, Sort
from agent_env.task.store import TaskStepStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["runs"])

TASK_INSTANCES_COLLECTION = "task_instances"

MAX_RUNS_PER_REQUEST = 100   # cap a single start-runs fan-out

RUN_GROUP_STREAM_POLL_SECONDS = 2      # gap between run-group snapshots on the SSE stream
RUN_GROUP_STREAM_MAX_SNAPSHOTS = 600   # ceiling before a `timeout` event (~20 min at the poll interval)


class RunRequest(BaseModel):
    """One run. ``extra="forbid"`` rejects unknown keys so an unsupported field returns a
    422 rather than being silently dropped."""

    model_config = ConfigDict(extra="forbid")

    version: Optional[int] = None
    agent_model: Optional[str] = None
    agent_artifact_id: Optional[str] = None
    metadata: Optional[dict] = None
    # Resume / override controls, threaded into context.metadata for the task step.
    start_step: Optional[int] = None
    context_from_instance_id: Optional[str] = None
    context_json: Optional[dict] = None
    step_overrides: Optional[dict] = None
    overrides: Optional[dict] = None
    project_id: Optional[str] = None
    priority: Optional[int] = None
    # Accepted but ignored: the local explorer gets models from the [model] config, not a
    # per-request key; declared so a client that always sends them isn't rejected.
    litellm_api_key: Optional[str] = None
    litellm_key_id: Optional[str] = None


class RunResponse(BaseModel):
    workflow_id: str
    instance_id: str
    status: RunStatus


# Keys the runner acts on that a caller could also smuggle through the untyped
# `metadata` passthrough. Validating only the typed RunRequest fields is not enough:
# `_run_metadata` copies `body.metadata` verbatim, so every one of these has to be
# checked on the merged dict, whatever route it arrived by.
_UNSUPPORTED_METADATA_KEYS = ("context_from_instance_id", "context_json")


def _validate_run_metadata(metadata: dict) -> None:
    """Reject metadata the runner cannot honour, before a run id is handed back.

    Anything accepted here and then ignored produces the worst outcome available: a run
    that reports its skipped steps as SUCCESS with no data, or one that raises inside
    the worker after submit already returned 200 and so never registers an instance.
    """
    for key in _UNSUPPORTED_METADATA_KEYS:
        if metadata.get(key) is not None:
            raise HTTPException(
                status_code=501,
                detail=f"{key} is not supported by this runner: a resumed run would "
                       "report skipped steps as successful with no data",
            )

    start_step = metadata.get("start_step")
    if start_step is not None:
        if isinstance(start_step, bool) or not isinstance(start_step, int):
            raise HTTPException(
                status_code=422,
                detail=f"start_step must be an integer, got {type(start_step).__name__}",
            )
        if start_step < 0:
            raise HTTPException(status_code=422, detail="start_step must be >= 0")

    overrides = metadata.get("user_overrides")
    if overrides is not None and not isinstance(overrides, dict):
        raise HTTPException(
            status_code=422,
            detail=f"metadata.user_overrides must be an object, got {type(overrides).__name__}",
        )


def _run_metadata(body: "RunRequest") -> dict:
    """Fold the override fields into the metadata Task.run() carries."""
    metadata = dict(body.metadata or {})
    _validate_run_metadata(metadata)          # what the caller smuggled in

    # Seed from any user_overrides already in metadata — a re-run inherits the prior
    # instance's — so the explicit fields merge into them rather than replacing the lot.
    user_overrides = {**(metadata.get("user_overrides") or {}), **(body.overrides or {})}
    for key in ("project_id", "priority"):
        value = getattr(body, key)
        if value is not None:
            user_overrides[key] = value
    # The steps read `step_params` (TaskStep.step_param_overrides); the request field is
    # named differently. Writing the request name through dropped every override.
    if body.step_overrides is not None:
        user_overrides["step_params"] = body.step_overrides
    if user_overrides:
        metadata["user_overrides"] = user_overrides

    for key in ("start_step", *_UNSUPPORTED_METADATA_KEYS):
        value = getattr(body, key)
        if value is not None:
            metadata[key] = value
    _validate_run_metadata(metadata)          # and what the typed fields added
    return metadata


def _resolve_task_version(task_id: str, version: Optional[int]) -> int:
    """The concrete version to run; 404 if the task (at ``version``, else latest) is absent."""
    from agent_env.task import Task

    task = Task.get(task_id, version)
    if task is None:
        label = f"v{version}" if version is not None else "latest"
        raise HTTPException(status_code=404, detail=f"Task {task_id} {label} not found")
    return task.version


@router.post("/{task_id}/run", response_model=RunResponse)
async def start_run(task_id: EntityId, body: RunRequest | None = None) -> RunResponse:
    body = body or RunRequest()
    version = _resolve_task_version(task_id, body.version)
    handle = await get_runner().submit(
        task_id, version,
        agent_model=body.agent_model,
        agent_artifact_id=body.agent_artifact_id,
        metadata=_run_metadata(body),
    )
    return RunResponse(workflow_id=handle.run_id, instance_id=handle.instance_id, status=RunStatus.QUEUED)


class StartRunsRequest(RunRequest):
    """N runs of one task, started as a group (the UI's "Start N Runs")."""

    count: Optional[int] = None
    seeds: Optional[list[dict]] = None
    # Accepted but ignored, like litellm_api_key above. Runs are submitted
    # sequentially and LocalRunner bounds execution with a semaphore sized once at
    # construction from `[runner.config] workers`, so there is no per-request knob to
    # honour. Declared so the inherited Start-Runs panel isn't rejected by
    # `extra="forbid"`; the panel labels it as ignored.
    concurrency: Optional[int] = None


class RunInfo(BaseModel):
    index: int
    workflow_id: Optional[str] = None
    seed: Optional[dict] = None
    error: Optional[str] = None


class StartRunsResponse(BaseModel):
    run_group_id: str
    task_id: str
    task_version: Optional[int] = None
    total: int
    runs: list[RunInfo]


@router.post("/{task_id}/runs", response_model=StartRunsResponse)
async def start_runs(task_id: EntityId, body: StartRunsRequest | None = None) -> StartRunsResponse:
    """Start a group of runs, returning their handles immediately. ``seeds`` wins over
    ``count`` when both are given (one run per seed)."""
    body = body or StartRunsRequest()
    version = _resolve_task_version(task_id, body.version)  # 404 the whole group if the task is absent
    requested = len(body.seeds) if body.seeds is not None else max(body.count or 1, 1)
    if requested > MAX_RUNS_PER_REQUEST:
        raise HTTPException(status_code=422,
                            detail=f"at most {MAX_RUNS_PER_REQUEST} runs per request (got {requested})")
    seeds = body.seeds if body.seeds is not None else [None] * requested
    run_group_id = f"rg-{uuid.uuid4().hex}"
    base_metadata = _run_metadata(body)

    runs: list[RunInfo] = []
    for index, seed in enumerate(seeds):
        metadata = dict(base_metadata)
        metadata["run_group_id"] = run_group_id
        if seed:
            metadata["seed"] = seed
        try:
            handle = await get_runner().submit(
                task_id, version,
                agent_model=body.agent_model,
                agent_artifact_id=body.agent_artifact_id,
                metadata=metadata,
            )
            runs.append(RunInfo(index=index, workflow_id=handle.run_id, seed=seed))
        except Exception as e:  # one bad run must not sink the group
            logger.exception("Run %d of group %s failed to start", index, run_group_id)
            runs.append(RunInfo(index=index, seed=seed, error=f"{type(e).__name__}: {e}"))

    return StartRunsResponse(
        run_group_id=run_group_id, task_id=task_id, task_version=version,
        total=len(runs), runs=runs,
    )


@router.post("/{task_id}/cancel-run")
async def cancel_run(task_id: EntityId, workflow_id: str = Query(...)) -> dict:
    canceled = await get_runner().cancel(workflow_id)
    return {"workflow_id": workflow_id, "canceled": canceled}


@router.get("/{task_id}/runs", response_model=PaginatedResponse)
def list_runs(
    task_id: EntityId,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> PaginatedResponse:
    from agent_env.runner import store as run_store

    records = run_store.list_runs(task_id=task_id, limit=limit, offset=offset)
    total = docs().count("runs", Filter.of(task_id=task_id))
    return PaginatedResponse(
        items=[r.to_dict() for r in records], total=total, limit=limit, offset=offset,
        has_more=offset + len(records) < total,
    )


@router.get("/{task_id}/instances", response_model=PaginatedResponse)
def list_instances(
    task_id: EntityId,
    task_version: Optional[int] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> PaginatedResponse:
    conditions = {"task_id": task_id}
    if task_version is not None:
        conditions["task_version"] = task_version
    filt = Filter.of(**conditions)
    store = docs()
    items = store.query(TASK_INSTANCES_COLLECTION, filt,
                        sort=Sort.by("created_at_utc", descending=True),
                        limit=limit, offset=offset)
    total = store.count(TASK_INSTANCES_COLLECTION, filt)
    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset,
                             has_more=offset + len(items) < total)


@router.get("/{task_id}/instances/{instance_id}")
def get_instance(task_id: EntityId, instance_id: EntityId) -> dict:
    doc = docs().find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"instance {instance_id} not found")
    return doc


# --- run groups -------------------------------------------------------------
#
# A run group is the set of runs sharing a run_group_id (stamped into each run's
# metadata) — a view over the runs collection, not a stored entity.


def _all_task_runs(task_id: str) -> list:
    """Every run of a task, paged so no fixed limit truncates the group aggregation."""
    from agent_env.runner import store as run_store

    out, offset = [], 0
    while True:
        page = run_store.list_runs(task_id=task_id, limit=500, offset=offset)
        out.extend(page)
        if len(page) < 500:
            return out
        offset += 500


def _group_id_of(record) -> str:
    """The group a run belongs to.

    A batch (``POST /runs``) stamps ``metadata.run_group_id`` on every run it starts.
    A single run (``POST /run``, the "Start 1 Run" button) stamps nothing, so it is its
    own group, keyed by its run id. ``list_run_groups`` and ``_instances_for_group``
    MUST agree on this — when only the former applied the fallback, every single run
    produced a group row that the join could never match, so the Rollouts table showed
    the group with "0 runs" and no instance, for every non-batch run.
    """
    return (record.overrides.get("metadata") or {}).get("run_group_id") or record.run_id


def _instances_for_group(run_group_id: str, task_id: str) -> tuple[list[dict], dict]:
    """Every run in the group, joined to its task instance for live status, plus the
    per-step ``{"done", "failed"}`` tally the Batch progress strip draws."""
    out: list[dict] = []
    step_counts: dict[str, dict[str, int]] = {}
    for record in _all_task_runs(task_id):
        if _group_id_of(record) != run_group_id:
            continue
        instance = docs().find_one(
            TASK_INSTANCES_COLLECTION, Filter.of(instance_id=record.instance_id)
        ) or {}
        # One outcome per step per run. A completion callback that raises records a
        # failure beside the step's success, and the failure is the final verdict.
        outcome: dict[str, str] = {}
        for step in instance.get("completed_steps") or []:
            step_id, status = step.get("step_id"), step.get("status")
            if status not in (TaskStepStatus.SUCCESS, TaskStepStatus.FAILURE):
                continue
            if outcome.get(step_id) != TaskStepStatus.FAILURE:
                outcome[step_id] = status
        for step_id, status in outcome.items():
            counts = step_counts.setdefault(step_id, {"done": 0, "failed": 0})
            counts["done" if status == TaskStepStatus.SUCCESS else "failed"] += 1
        out.append({
            "instance_id": record.instance_id,
            "workflow_id": record.run_id,
            "seed": (record.overrides.get("metadata") or {}).get("seed"),
            # The run record is authoritative for lifecycle; the instance may lag or be absent.
            "status": str(record.status).lower(),
            "created_at_utc": record.created_at_utc,
            "completed_at_utc": record.finished_at_utc,
            "current_step": instance.get("current_step"),
            "total_steps": instance.get("total_steps"),
        })
    return out, step_counts


# Run-group funnel buckets keyed by RunStatus, so the tally tracks the enum — adding a
# status is a deliberate edit here, not a silent fall-through. CANCELED folds into
# "failed"; QUEUED is "provisioning" (accepted, not yet picked up by a worker).
_FUNNEL_BUCKET: dict[RunStatus, str] = {
    RunStatus.COMPLETED: "completed",
    RunStatus.FAILED: "failed",
    RunStatus.CANCELED: "failed",
    RunStatus.RUNNING: "running",
    RunStatus.QUEUED: "provisioning",
}


def _group_status(task_id: str, run_group_id: str) -> dict:
    instances, step_counts = _instances_for_group(run_group_id, task_id)
    tally = {"completed": 0, "failed": 0, "running": 0, "provisioning": 0}
    for inst in instances:
        try:
            run_status: Optional[RunStatus] = RunStatus(str(inst["status"]).upper())
        except ValueError:
            run_status = None
        tally[_FUNNEL_BUCKET.get(run_status, "provisioning")] += 1
    return {
        "run_group_id": run_group_id,
        "task_id": task_id,
        "total": len(instances),
        **tally,
        "step_counts": step_counts,
        "instances": instances,
    }


@router.get("/{task_id}/run-groups", response_model=PaginatedResponse)
def list_run_groups(
    task_id: EntityId,
    task_version: Optional[int] = None,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PaginatedResponse:
    """Run groups for a task, newest first — the Rollouts table."""
    groups: dict[str, dict] = {}
    for record in _all_task_runs(task_id):
        if task_version is not None and record.task_version != task_version:
            continue
        gid = _group_id_of(record)
        group = groups.setdefault(gid, {
            "run_group_id": gid,
            "task_id": task_id,
            "task_version": record.task_version,
            "created_at_utc": record.created_at_utc,
            "total": 0,
        })
        group["total"] += 1
        group["created_at_utc"] = record.created_at_utc  # newest-first input: ends on the oldest

    ordered = sorted(groups.values(), key=lambda g: g["created_at_utc"] or "", reverse=True)
    page = ordered[offset: offset + limit]
    items = []
    for group in page:
        status = _group_status(task_id, group["run_group_id"])
        # The list row nests the tally under `counts` and names its timestamp
        # `earliest_created_at_utc` — unlike the flat shape /run-groups/{id} returns.
        items.append({
            "run_group_id": group["run_group_id"],
            "task_id": task_id,
            "task_version": group["task_version"],
            "earliest_created_at_utc": group["created_at_utc"],
            "total": status["total"],
            "counts": {
                "completed": status["completed"],
                "failed": status["failed"],
                "running": status["running"],
                "provisioning": status["provisioning"],
                "waiting": 0,
            },
            "step_counts": status["step_counts"],
            "instances": status["instances"],
        })
    return PaginatedResponse(items=items, total=len(ordered), limit=limit, offset=offset,
                             has_more=offset + len(items) < len(ordered))


@router.get("/{task_id}/run-groups/{run_group_id}")
def get_run_group(task_id: EntityId, run_group_id: str) -> dict:
    return _group_status(task_id, run_group_id)


@router.get("/{task_id}/run-groups/{run_group_id}/stream")
async def stream_run_group(task_id: EntityId, run_group_id: str):
    """Server-sent snapshots until every run in the group is terminal, ending with a
    ``complete`` event."""
    from fastapi.responses import StreamingResponse

    async def events():
        for _ in range(RUN_GROUP_STREAM_MAX_SNAPSHOTS):
            snapshot = _group_status(task_id, run_group_id)
            if snapshot["total"] == 0:
                yield "event: not_found\ndata: {}\n\n"
                return
            yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
            if snapshot["completed"] + snapshot["failed"] == snapshot["total"]:
                yield "event: complete\ndata: {}\n\n"
                return
            await asyncio.sleep(RUN_GROUP_STREAM_POLL_SECONDS)
        yield "event: timeout\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/{task_id}/instances/{instance_id}/progress")
def instance_progress(task_id: EntityId, instance_id: EntityId) -> dict:
    """Step progress for one instance. Polled while a run is in flight."""
    doc = docs().find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"instance {instance_id} not found")
    # The poller stops on `done`. It also reads `live_step_index`, deliberately not
    # sent: `current_step` counts completed steps rather than indexing them, and a
    # `depends_on` DAG completes out of order, so the count would mark un-started
    # steps done. The component's firstPending fallback handles that correctly.
    status = doc.get("status")
    current_step = doc.get("current_step")
    try:
        # The instance store spells it "cancelled"; RunStatus spells it CANCELED.
        terminal = RunStatus("CANCELED" if status == "cancelled" else str(status).upper()).is_terminal
    except ValueError:
        terminal = False   # an unrecognised status keeps polling rather than stalling the view
    return {
        "instance_id": instance_id,
        "status": status,
        "current_step": current_step,
        "total_steps": doc.get("total_steps"),
        "completed_steps": doc.get("completed_steps") or [],
        "done": terminal,
    }
