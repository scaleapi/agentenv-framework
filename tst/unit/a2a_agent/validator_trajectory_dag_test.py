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
from datetime import UTC, datetime, timedelta

import pytest
from agentenv_protocol.transfers import HttpPutGrant

from agent_env.a2a_agent import A2AAgent, object_transfer
from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.config import Config, configure, get_config
from agent_env.task import Task
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from tst.unit.event_loop_probe import on_event_loop
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_mcp import (
    VerifyA2AAgentMCPStep,
)
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_trajectory import (
    VerifyA2ATrajectoryStep,
)


def _context_with_trajectory_agent(
    mcp_verification: dict | None, *, get_method: dict | None = None
) -> TaskStepContext:
    """A context whose deployed agent advertises the trajectory extension."""
    context = TaskStepContext()
    context.deployed_agents = [
        DeployedAgent(
            agent_name="agent-under-test",
            api_url="http://agent.invalid",
            a2a_url="http://agent.invalid",
            a2a_card={
                "capabilities": {
                    "extensions": [
                        {
                            "uri": A2AAgent.EXT_TRAJECTORY,
                            "config": (
                                {"methods": {"get": get_method}}
                                if get_method is not None
                                else {}
                            ),
                        }
                    ]
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
        if "objects" in json:
            return _StubResponse(
                {
                    "objects": {
                        "trajectory": {"size_bytes": 42}
                    }
                }
            )
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
    monkeypatch.setattr(object_transfer.httpx, "AsyncClient", _StubAsyncClient)
    monkeypatch.setattr(A2AAgent, "get", lambda *_a, **_kw: fake_agent)

    step = VerifyA2ATrajectoryStep(id="t", version=None, a2a_agent_id="agent-x")
    result = asyncio.run(
        step.execute(_context_with_trajectory_agent({"task_id": "task-abc"}))
    )

    assert [p["task_id"] for p in _StubAsyncClient.posts] == ["task-abc", "task-abc"]
    assert result.metadata["verifications"]["a2a_trajectory"] == {
        "inline": True,
        "objects": False,
        "s3": True,
    }
    assert (
        fake_agent.metadata["validated_a2a_extensions"][A2AAgent.EXT_TRAJECTORY][
            "supported"
        ]
        is True
    )


def test_trajectory_step_treats_method_without_request_as_legacy(monkeypatch):
    fake_agent = _FakeAgent()
    _StubAsyncClient.posts = []
    monkeypatch.setattr(object_transfer.httpx, "AsyncClient", _StubAsyncClient)
    monkeypatch.setattr(A2AAgent, "get", lambda *_a, **_kw: fake_agent)

    result = asyncio.run(
        VerifyA2ATrajectoryStep(
            id="t", version=None, a2a_agent_id="agent-x"
        ).execute(
            _context_with_trajectory_agent(
                {"task_id": "task-abc"}, get_method={"endpoint": "/trajectory"}
            )
        )
    )

    assert [set(post) for post in _StubAsyncClient.posts] == [
        {"task_id"},
        {"task_id", "trajectory_s3_prefix"},
    ]
    assert result.metadata["verifications"]["a2a_trajectory"]["s3"] is True


def test_trajectory_step_records_failed_object_url_construction(monkeypatch):
    class _FailingObjectStore:
        supports_transfer_grants = True

        def grants_reach(self, _sandbox_type: str | None) -> bool:
            return True

        def object_url(self, _key: str) -> str:
            raise RuntimeError("cannot mint object URL")

    fake_agent = _FakeAgent()
    _StubAsyncClient.posts = []
    monkeypatch.setattr(object_transfer.httpx, "AsyncClient", _StubAsyncClient)
    monkeypatch.setattr(A2AAgent, "get", lambda *_a, **_kw: fake_agent)
    monkeypatch.setattr(
        Config, "get_object_store", lambda self: _FailingObjectStore()
    )
    get_method = {
        "request": {
            "required": ["task_id", "objects"],
        }
    }

    result = asyncio.run(
        VerifyA2ATrajectoryStep(
            id="t", version=None, a2a_agent_id="agent-x"
        ).execute(
            _context_with_trajectory_agent(
                {"task_id": "task-abc"}, get_method=get_method
            )
        )
    )

    assert result.metadata["verifications"]["a2a_trajectory"] == {
        "inline": True,
        "objects": False,
        "s3": False,
    }


class _ObjectStore:
    supports_transfer_grants = True
    max_single_upload_bytes = None

    def __init__(self) -> None:
        self.write_grants: list[str] = []

    def object_url(self, key: str) -> str:
        return f"s3://test-bucket/{key}"

    def grants_reach(self, sandbox_type: str | None) -> bool:
        return True

    def get_object_key(self, object_url: str) -> str:
        return object_url.removeprefix("s3://test-bucket/")

    def issue_write_grant(self, object_url, *, media_type, max_bytes, expires_in=None):
        self.write_grants.append(object_url)
        return HttpPutGrant(
            kind="http-put",
            url="https://objects.example.test/write?secret=signed",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )


def test_trajectory_step_probes_the_advertised_object_variant(monkeypatch):
    fake_agent = _FakeAgent()
    _StubAsyncClient.posts = []
    monkeypatch.setattr(object_transfer.httpx, "AsyncClient", _StubAsyncClient)
    monkeypatch.setattr(A2AAgent, "get", lambda *_a, **_kw: fake_agent)
    monkeypatch.setattr(Config, "get_object_store", lambda self: _ObjectStore())
    get_method = {
        "request": {
            "required": ["task_id"],
            "oneOf": [
                {"optional": ["trajectory_s3_prefix"]},
                {"required": ["objects"]},
            ],
        }
    }

    result = asyncio.run(
        VerifyA2ATrajectoryStep(
            id="t", version=None, a2a_agent_id="agent-x"
        ).execute(
            _context_with_trajectory_agent(
                {"task_id": "task-abc"}, get_method=get_method
            )
        )
    )

    assert [set(post) for post in _StubAsyncClient.posts] == [
        {"task_id"},
        {"task_id", "objects"},
        {"task_id", "trajectory_s3_prefix"},
    ]
    assert result.metadata["verifications"]["a2a_trajectory"] == {
        "inline": True,
        "objects": True,
        "s3": True,
    }
    options = fake_agent.metadata["validated_a2a_extensions"][
        A2AAgent.EXT_TRAJECTORY
    ]["methods"]["get"]["options"]
    assert options["objects"] == {"supported": True}


def test_trajectory_step_skips_the_object_variant_on_a_store_without_grants(monkeypatch):
    store = _ObjectStore()
    store.supports_transfer_grants = False
    fake_agent = _FakeAgent()
    _StubAsyncClient.posts = []
    monkeypatch.setattr(object_transfer.httpx, "AsyncClient", _StubAsyncClient)
    monkeypatch.setattr(A2AAgent, "get", lambda *_a, **_kw: fake_agent)
    monkeypatch.setattr(Config, "get_object_store", lambda self: store)
    get_method = {"request": {"required": ["task_id"], "oneOf": [{}, {"required": ["objects"]}]}}

    result = asyncio.run(
        VerifyA2ATrajectoryStep(id="t", version=None, a2a_agent_id="agent-x").execute(
            _context_with_trajectory_agent({"task_id": "task-abc"}, get_method=get_method)
        )
    )

    assert _StubAsyncClient.posts == [{"task_id": "task-abc"}]
    assert store.write_grants == []
    assert result.metadata["verifications"]["a2a_trajectory"]["inline"] is True


@pytest.mark.asyncio
async def test_validator_anchors_trajectory_step_on_its_mcp_step(monkeypatch):
    """The real validator task has an MCP → trajectory dependency edge.

    This exercises ``A2AAgentValidator.validate`` through task construction,
    rather than creating an independent miniature DAG that would keep passing
    if the validator wiring regressed.
    """
    captured: dict = {}

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

    monkeypatch.setattr("agent_env.a2a_agent.validator.get_config", lambda: FakeConfig())
    monkeypatch.setattr(Task, "put", capture_task_put)
    monkeypatch.setattr(
        A2AAgentValidator,
        "_upload_skill_fixture",
        lambda *_args, **_kwargs: "s3://fixture/skill",
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


def test_the_trajectory_probe_prefix_is_under_the_fixture_prefix(local_stores, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_FIXTURE_PREFIX", "fx")
    configure()
    _StubAsyncClient.posts = []
    monkeypatch.setattr(object_transfer.httpx, "AsyncClient", _StubAsyncClient)
    monkeypatch.setattr(A2AAgent, "get", lambda *_a, **_kw: _FakeAgent())

    asyncio.run(
        VerifyA2ATrajectoryStep(id="t", version=None, a2a_agent_id="agent-x").execute(
            _context_with_trajectory_agent({"task_id": "task-abc"})
        )
    )

    (legacy,) = [post for post in _StubAsyncClient.posts if "trajectory_s3_prefix" in post]
    store = get_config().get_object_store()
    assert store.get_object_key(legacy["trajectory_s3_prefix"]).startswith("fx/a2a_validator_trajectories/agent-x")


@pytest.mark.asyncio
async def test_the_skill_fixtures_upload_off_the_event_loop(monkeypatch):
    on_loop: list[bool] = []

    class FakeTask:
        id, version = "validation-task", 1

        async def run(self, *, context: TaskStepContext, **_kwargs) -> TaskStepContext:
            return context

    class FakeAgent:
        id, version, VALIDATION_ENV_ID = "agent-x", 7, "a2a-validation-env"

    async def noop(*_args, **_kwargs) -> None:
        return None

    def upload(*_args, **_kwargs):
        on_loop.append(on_event_loop())
        return "s3://fixture/skill"

    monkeypatch.setattr(Task, "put", lambda **kwargs: FakeTask())
    monkeypatch.setattr(A2AAgentValidator, "_upload_skill_fixture", upload)
    monkeypatch.setattr(A2AAgentValidator, "_upload_probe_fixtures", lambda *_args: object())
    monkeypatch.setattr(A2AAgentValidator, "_build_modality_steps", lambda *_args: ([], []))
    for step in ("_validate_agent_changelog", "_validate_install", "_cleanup_sandboxes"):
        monkeypatch.setattr(A2AAgentValidator, step, noop)

    await A2AAgentValidator.validate(FakeAgent())

    assert on_loop == [False, False, False]


@pytest.mark.asyncio
async def test_the_install_image_fixture_uploads_off_the_event_loop(local_stores, monkeypatch):
    on_loop: list[bool] = []

    def upload(_agent):
        on_loop.append(on_event_loop())
        raise RuntimeError("stop after the upload")

    monkeypatch.setattr(A2AAgentValidator, "_upload_install_test_image_fixture", upload)
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_INSTALL}]}}
    context = TaskStepContext(deployed_agents=[
        DeployedAgent(agent_name=TaskStep.DEFAULT_AGENT_NAME, api_url="http://agent", a2a_card=card),
    ])

    await A2AAgentValidator._validate_install(type("Agent", (), {"id": "agent-x", "version": 1})(), context, "task")

    assert on_loop == [False]
    assert "stop after the upload" in context.metadata["verifications"]["a2a_install"]["error"]


@pytest.mark.asyncio
async def test_cleanup_terminates_the_env_sandboxes_and_leaves_an_env_outside_them(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
    from agent_env.task_step.context import TaskStepContext

    provider = MagicMock(get_sandbox=AsyncMock(return_value=MagicMock(terminate=AsyncMock())))
    monkeypatch.setattr("agent_env.providers.get_env_sandbox_provider", lambda: provider)
    context = TaskStepContext(deployed_envs=[
        DeployedEnv(env_id="hosted", env_version=1, env_provider_type="hosted"),
        DeployedSandboxEnv(env_id="contained", env_version=1, sandbox_id="sb-1"),
    ])

    await A2AAgentValidator._cleanup_sandboxes(context)

    provider.get_sandbox.assert_awaited_once_with("sb-1")
