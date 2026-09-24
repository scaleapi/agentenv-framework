"""Unit tests pinning the trajectory check's dependency on the MCP check.

`VerifyA2ATrajectoryStep` retrieves the completed task's id from the MCP step's
verification result. When both steps were anchored on `deploy_agent` they ran
concurrently, so the trajectory step read `verifications` before the ~30s MCP
turn had written it and reported the extension unsupported even when the
agent's `/ext/trajectory` endpoint was healthy.

These tests pin the two halves of that contract: the metadata key the
trajectory step reads, and the dependency edge emitted by the real validator
task builder.
"""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _s3_bucket_backs_prefix_minting(monkeypatch):
    # get_s3_bucket() re-sources from the configured object store; these tests
    # run without one, so pin the bucket the probe prefixes are minted in.
    from agent_env.config import Config

    monkeypatch.setattr(Config, "get_s3_bucket", lambda self: "test-bucket")

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.task import Task
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator import verify_a2a_trajectory
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_mcp import (
    VerifyA2AAgentMCPStep,
)
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_trajectory import (
    VerifyA2ATrajectoryStep,
)


def _context_with_trajectory_agent(mcp_verification: dict | None) -> TaskStepContext:
    """A context whose deployed agent advertises the trajectory extension."""
    context = TaskStepContext()
    context.deployed_agents = [
        DeployedAgent(
            agent_name="agent-under-test",
            api_url="http://agent.invalid",
            a2a_url="http://agent.invalid",
            a2a_card={
                "capabilities": {
                    "extensions": [{"uri": A2AAgent.EXT_TRAJECTORY, "config": {}}]
                }
            },
        )
    ]
    if mcp_verification is not None:
        context.metadata["verifications"] = {"a2a_agent_mcp": mcp_verification}
    return context


class _StubResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _StubAsyncClient:
    """Records the trajectory POSTs and answers them without touching the network."""

    posts: list[dict] = []

    async def __aenter__(self) -> "_StubAsyncClient":
        return self

    async def __aexit__(self, *_exc_info) -> None:
        return None

    async def post(self, _url: str, json: dict, timeout: int) -> _StubResponse:  # noqa: A002
        type(self).posts.append(json)
        if "trajectory_s3_prefix" in json:
            return _StubResponse({"trajectory_s3_prefix": json["trajectory_s3_prefix"]})
        return _StubResponse({"trajectory": [{"type": "tool_call"}]})


class _FakeAgent:
    """Stands in for the Mongo-backed A2AAgent document."""

    def __init__(self) -> None:
        self.metadata: dict = {}

    def update_metadata(self, metadata: dict) -> None:
        self.metadata = metadata


def test_trajectory_step_fails_loudly_without_the_mcp_task_id():
    """Without the MCP step's task_id the check cannot run, and says so.

    This is precisely the failure the old deploy_agent-anchored ordering
    produced on every run that lost the race.
    """
    step = VerifyA2ATrajectoryStep(id="t", version=None, a2a_agent_id="agent-x")

    with pytest.raises(RuntimeError, match="task_id"):
        asyncio.run(step.execute(_context_with_trajectory_agent(None)))


def test_trajectory_step_probes_with_the_task_id_from_the_mcp_verification(monkeypatch):
    """The MCP step's `task_id` is load-bearing — it is what both probes carry.

    Renaming the `verifications["a2a_agent_mcp"]["task_id"]` key, or severing
    the DAG edge that populates it, breaks the check.
    """
    fake_agent = _FakeAgent()
    _StubAsyncClient.posts = []
    monkeypatch.setattr(verify_a2a_trajectory.httpx, "AsyncClient", _StubAsyncClient)
    monkeypatch.setattr(A2AAgent, "get", lambda *_a, **_kw: fake_agent)

    step = VerifyA2ATrajectoryStep(id="t", version=None, a2a_agent_id="agent-x")
    result = asyncio.run(
        step.execute(_context_with_trajectory_agent({"task_id": "task-abc"}))
    )

    assert [p["task_id"] for p in _StubAsyncClient.posts] == ["task-abc", "task-abc"]
    assert result.metadata["verifications"]["a2a_trajectory"] == {
        "inline": True,
        "s3": True,
    }
    assert (
        fake_agent.metadata["validated_a2a_extensions"][A2AAgent.EXT_TRAJECTORY][
            "supported"
        ]
        is True
    )


@pytest.mark.asyncio
async def test_validator_anchors_trajectory_step_on_its_mcp_step(monkeypatch):
    """The real validator task has an MCP → trajectory dependency edge.

    This exercises ``A2AAgentValidator.validate`` through task construction,
    rather than creating an independent miniature DAG that would keep passing
    if the validator wiring regressed.
    """
    captured: dict = {}

    class FakeConfig:
        def get_s3_bucket(self) -> str:
            return "unit-test-bucket"

    class FakeTask:
        id = "validation-task"
        version = 1

        async def run(self, *, context: TaskStepContext, **_kwargs) -> TaskStepContext:
            return context

    class FakeAgent:
        id = "agent-x"
        version = 7
        VALIDATION_ENV_ID = "a2a-validation-env"

    def capture_task_put(**kwargs):
        captured["steps"] = kwargs["steps"]
        return FakeTask()

    async def noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("agent_env.config.get_config", lambda: FakeConfig())
    monkeypatch.setattr(Task, "put", capture_task_put)
    monkeypatch.setattr(
        A2AAgentValidator, "_upload_skill_fixture", lambda *_args: "s3://fixture/skill"
    )
    monkeypatch.setattr(
        A2AAgentValidator, "_upload_probe_fixtures", lambda *_args: object()
    )
    monkeypatch.setattr(
        A2AAgentValidator, "_build_modality_steps", lambda *_args: ([], [])
    )
    monkeypatch.setattr(A2AAgentValidator, "_validate_agent_changelog", noop)
    monkeypatch.setattr(A2AAgentValidator, "_validate_install", noop)
    monkeypatch.setattr(A2AAgentValidator, "_cleanup_sandboxes", noop)

    await A2AAgentValidator.validate(FakeAgent())

    mcp_step = next(
        step for step in captured["steps"] if isinstance(step, VerifyA2AAgentMCPStep)
    )
    trajectory_step = next(
        step for step in captured["steps"] if isinstance(step, VerifyA2ATrajectoryStep)
    )
    assert [dependency.task_step_id for dependency in trajectory_step.depends_on] == [
        mcp_step.id
    ]
