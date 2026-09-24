"""Unit tests for CombineUniverseVerdictsStep — the single authoritative writer of the merged
UNIVERSE_COMPATIBILITY result. The env-artifact store is mocked so these run offline.
"""

import dataclasses
import json

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.env.env_artifact_store import EnvArtifactType
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.multienv_validator.combine_universe_verdicts import (
    CombineUniverseVerdictsStep,
)
from agent_env.task_step.task_steps.multienv_validator.verify_universe_agent_judge import (
    CROSS_SERVICE_KEY,
)

VID = "task-judge-rubric"


class _FakeStore:
    def __init__(self):
        self.puts = []

    def put(self, **kwargs):
        self.puts.append(kwargs)


def _deployed_env(env_id="multi-x", env_version=3):
    return DeployedEnv(
        env_id=env_id, env_version=env_version, gateway_url="g", mcp_url="m",
        db_web_url=None, sandbox_id="sb-1",
    )


def _ctx(verifications, deployed_envs=None):
    ctx = TaskStepContext()
    ctx.deployed_envs = deployed_envs if deployed_envs is not None else [_deployed_env()]
    ctx.metadata = {"verifications": verifications}
    return ctx


def _step():
    return CombineUniverseVerdictsStep(
        id="task-combine", version=None, env_id="multi-x",
        universe_artifact_id="uni", universe_artifact_version=7, judge_verifier_id=VID,
    )


def _pv():
    return {"compatible": True, "services": {"slack": {"compatible": True, "issues": []}}}


@pytest.fixture
def fake_store(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr("agent_env.env.env_artifact_store.get_env_artifact_store", lambda: store)
    return store


@pytest.mark.asyncio
async def test_merges_judge_verdict_and_persists_once(fake_store):
    verifs = {
        EnvArtifactType.UNIVERSE_COMPATIBILITY: _pv(),
        VID: {"results": [{"id": "service_preserved__slack", "score": 0.0, "justification": "topic dropped"}], "score": 0.0},
    }
    await _step().execute(_ctx(verifs))

    assert len(fake_store.puts) == 1
    data = fake_store.puts[0]["data"]
    assert fake_store.puts[0]["env_version"] == 3  # taken from the deployed env, matching the PV write
    assert data["compatible"] is False  # judge gated it
    assert data["services"]["slack"]["compatible"] is False
    assert any(i["type"] == "agent_judge" for i in data["services"]["slack"]["issues"])
    assert "agent_judge" in data  # raw verdict retained
    assert "agent_judge_skipped" not in data  # gated normally → not marked skipped


@pytest.mark.asyncio
async def test_persists_pv_ungated_when_judge_verdict_missing(fake_store):
    verifs = {EnvArtifactType.UNIVERSE_COMPATIBILITY: _pv()}  # no judge entry
    await _step().execute(_ctx(verifs))

    assert len(fake_store.puts) == 1
    data = fake_store.puts[0]["data"]
    assert data["compatible"] is True  # PV unchanged
    assert "agent_judge" not in data
    assert data["agent_judge_skipped"] is True  # marked so downstream knows it wasn't gated


@pytest.mark.asyncio
async def test_no_write_when_pv_result_missing(fake_store):
    await _step().execute(_ctx({VID: {"results": [], "score": 1.0}}))
    assert fake_store.puts == []


@pytest.mark.asyncio
async def test_no_write_when_env_not_deployed(fake_store):
    verifs = {EnvArtifactType.UNIVERSE_COMPATIBILITY: _pv(), VID: {"results": [], "score": 1.0}}
    await _step().execute(_ctx(verifs, deployed_envs=[]))  # env not in context
    assert fake_store.puts == []


def test_serialization_round_trip():
    s = _step()
    back = CombineUniverseVerdictsStep.from_dict(s.to_dict())
    assert back.env_id == "multi-x"
    assert back.universe_artifact_version == 7
    assert back.judge_verifier_id == VID


# --- The "environments" twin across the Temporal step boundary -----------------------------
#
# VerifyUniverseLoadExportRoundtripStep builds ONE dict and binds it to both "services" and
# "environments" (dual-write during the service->environment rename), so in-process every mutation
# lands in both. That aliasing does NOT survive serialization: dataclasses.asdict recurses per key
# with no shared memo, and JSON has no notion of object identity — so after a Temporal heartbeat or
# an activity hand-off the twins are two independent dicts. apply_judge_verdict writes only
# "services", so the twin would persist stale without the re-point in the combine step.


def _pv_aliased():
    """The programmatic verdict as VerifyUniverseLoadExportRoundtripStep actually builds it:
    ONE dict bound to both keys (see verify_universe_roundtrip.py Phase 7)."""
    per_environment: dict = {}
    pv = {
        "compatible": True,
        "services": per_environment,
        "environments": per_environment,
        "exported_universe_artifact_id": "uni-export1",
    }
    pv["services"]["slack"] = {"compatible": True, "issues": []}
    pv["services"]["email"] = {"compatible": True, "issues": []}
    return pv


def _judge_flags_slack_and_refs():
    """A verdict flagging one per-service criterion and one cross-cutting criterion. The
    cross-cutting flag lands under a service key that does not exist in the PV at all."""
    return {
        "results": [
            {"id": "service_preserved__slack", "result": False, "score": 0.0, "justification": "topic dropped"},
            {"id": "referential_integrity", "result": False, "score": 0.0, "justification": "dangling refs"},
            {"id": "service_preserved__email", "result": True, "score": 1.0, "justification": "ok"},
        ],
        "score": 0.0,
    }


def _cross_temporal_boundary(ctx: TaskStepContext) -> TaskStepContext:
    """Round-trip the context exactly the way the Temporal worker does: _sanitize_context
    (dataclasses.asdict) -> JSON payload/heartbeat -> _context_from_json (TaskStepContext.from_dict).
    default=str mirrors the worker's tolerant encoder."""
    return TaskStepContext.from_dict(json.loads(json.dumps(dataclasses.asdict(ctx), default=str)))


def test_alias_does_not_survive_the_temporal_boundary():
    """Premise pin: aliased in-process, two independent dicts after asdict and after JSON."""
    pv = _pv_aliased()
    assert pv["services"] is pv["environments"]

    ctx = _ctx({EnvArtifactType.UNIVERSE_COMPATIBILITY: pv})

    # dataclasses.asdict alone already splits them — the worker's _sanitize_context is enough.
    as_dict = dataclasses.asdict(ctx)["metadata"]["verifications"][EnvArtifactType.UNIVERSE_COMPATIBILITY]
    assert as_dict["services"] is not as_dict["environments"]
    assert as_dict["services"]["slack"] is not as_dict["environments"]["slack"]

    # ...and so does a full JSON round-trip.
    crossed = _cross_temporal_boundary(ctx).metadata["verifications"][EnvArtifactType.UNIVERSE_COMPATIBILITY]
    assert crossed["services"] is not crossed["environments"]
    assert crossed["services"]["slack"] is not crossed["environments"]["slack"]


@pytest.mark.asyncio
async def test_environments_twin_not_stale_when_judge_gates_after_boundary(fake_store):
    """The judge's False must reach BOTH twins in the persisted doc, even though the boundary broke
    the alias and apply_judge_verdict writes only "services"."""
    ctx = _ctx({EnvArtifactType.UNIVERSE_COMPATIBILITY: _pv_aliased(), VID: _judge_flags_slack_and_refs()})
    crossed = _cross_temporal_boundary(ctx)
    pre = crossed.metadata["verifications"][EnvArtifactType.UNIVERSE_COMPATIBILITY]
    assert pre["services"] is not pre["environments"]  # guard: the boundary really did split them

    await _step().execute(crossed)

    assert len(fake_store.puts) == 1
    data = fake_store.puts[0]["data"]
    assert data["compatible"] is False
    # the whole twin agrees — not just the top-level verdict
    assert data["environments"] == data["services"]
    # judge's False propagated into BOTH twins, not just "services"
    assert data["services"]["slack"]["compatible"] is False
    assert data["environments"]["slack"]["compatible"] is False
    assert any(i["type"] == "agent_judge" for i in data["environments"]["slack"]["issues"])
    # a cross-cutting flag creates a service key absent from the PV; it must exist in the twin too
    assert data["environments"][CROSS_SERVICE_KEY]["compatible"] is False
    # unflagged services stay compatible in both
    assert data["services"]["email"]["compatible"] is True
    assert data["environments"]["email"]["compatible"] is True
