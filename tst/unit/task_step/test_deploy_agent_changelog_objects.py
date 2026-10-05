from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from agent_env.config import configure
from agent_env.store import GrantUnavailableError
from agent_env.task_step.context import TaskStepContext
from agent_env.a2a_agent import object_transfer
from agent_env.a2a_agent.object_transfer import ObjectLimits, changelog_apply_call
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from tst.util.granting_object_store import GrantingObjectStore

NAMESPACE_KEY = "agent_changelog/run-1/solver"


@pytest.fixture
def store(tmp_path) -> GrantingObjectStore:
    store = GrantingObjectStore(str(tmp_path))
    configure(object_store=store)
    return store


def _increments(store: GrantingObjectStore, *names: str, data: bytes = b"increment") -> str:
    for name in names:
        store.put(f"{NAMESPACE_KEY}/{name}", data)
    return store.object_url(NAMESPACE_KEY)


def _card(
    *,
    enable_objects: bool = True,
    enable_legacy: bool = True,
    apply_objects: bool = True,
) -> dict:
    enable_variants = []
    if enable_legacy:
        enable_variants.append({"required": ["s3_prefix"]})
    if enable_objects:
        enable_variants.append({"required": ["write_namespace"]})
    apply_variants = [{"required": ["s3_prefix"]}]
    if apply_objects:
        apply_variants.append({"required": ["increments"]})
    return {
        "capabilities": {
            "extensions": [
                {
                    "uri": "urn:agentenv:snapshot/v1",
                    "params": {
                        "methods": {
                            "enable-changelog": {
                                "endpoint": "/custom/changelog",
                                "request": {"oneOf": enable_variants},
                            },
                            "apply-changelog": {
                                "endpoint": "/custom/changelog",
                                "request": {"oneOf": apply_variants},
                            },
                        }
                    },
                }
            ]
        }
    }


def _step(**kwargs) -> DeployAgentTaskStep:
    return DeployAgentTaskStep(
        id="deploy",
        version=None,
        agent_name="solver",
        **kwargs,
    )


def _install(monkeypatch, response: dict, *, status_code: int = 200) -> list[dict]:
    requests: list[dict] = []

    async def fake_request(
        self, method, url, *, json=None, timeout=None, **kwargs
    ):  # noqa: A002
        requests.append(
            {"method": method, "url": url, "json": json, "timeout": timeout}
        )
        return httpx.Response(
            status_code, json=response, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    return requests


def _run_context() -> TaskStepContext:
    context = TaskStepContext()
    context.instance_id = "run-1"
    return context


def _granted_names(store: GrantingObjectStore) -> list[str]:
    return [url.rsplit("/", 1)[-1] for url in store.granted]


def test_changelog_grant_lifetime_tracks_resolved_agent_ttl():
    assert _step(ttl_seconds=7_200)._resolved_ttl_seconds(TaskStepContext()) == 7_200
    context = TaskStepContext(metadata={"user_overrides": {"ttl_seconds": 28_800}})
    assert _step(ttl_seconds=7_200)._resolved_ttl_seconds(context) == 28_800
    assert _step()._resolved_ttl_seconds(TaskStepContext()) == 7_200


@pytest.mark.asyncio
async def test_enable_prefers_a_bounded_namespace_grant(monkeypatch, store):
    requests = _install(monkeypatch, {"roots": ["workspace"]})
    context = _run_context()

    await _step(enable_agent_changelog=True)._configure_agent_changelog(
        "https://agent", _card(), context, expires_in=28_800
    )

    sent = requests[0]
    assert sent["url"] == "https://agent/custom/changelog"
    assert set(sent["json"]) == {"write_namespace"}
    grant = sent["json"]["write_namespace"]
    assert grant["root_path"] == NAMESPACE_KEY
    assert grant["max_objects"] == object_transfer.CHANGELOG_LIMITS.max_objects
    expires_at = datetime.fromisoformat(grant["expires_at"])
    assert abs(expires_at - (datetime.now(UTC) + timedelta(seconds=28_800))) < timedelta(
        seconds=5
    )
    assert context.metadata["agent_changelog"] == [
        {
            "agent_name": "solver",
            "roots": ["workspace"],
            "transfer_mode": "objects",
            "object_url": store.object_url(NAMESPACE_KEY),
        }
    ]


@pytest.mark.asyncio
async def test_the_changelog_namespace_is_under_the_fixture_prefix(monkeypatch, store):
    monkeypatch.setenv("AGENT_ENV_FIXTURE_PREFIX", "fx")
    configure(object_store=store)
    requests = _install(monkeypatch, {"roots": ["workspace"]})
    context = _run_context()

    await _step(enable_agent_changelog=True)._configure_agent_changelog(
        "https://agent", _card(), context, expires_in=600
    )

    assert requests[0]["json"]["write_namespace"]["root_path"] == f"fx/{NAMESPACE_KEY}"
    assert context.metadata["agent_changelog"][0]["object_url"] == store.object_url(f"fx/{NAMESPACE_KEY}")


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_legacy", [True, False], ids=["dual", "portable-only"])
async def test_enable_fails_when_the_store_cannot_grant_the_lifetime(
    monkeypatch, store, enable_legacy
):
    def unavailable(*args, **kwargs):
        raise GrantUnavailableError("credentials expire first")

    monkeypatch.setattr(store, "issue_upload_policy", unavailable)
    requests = _install(monkeypatch, {"roots": ["workspace"]})

    with pytest.raises(
        GrantUnavailableError,
        match="'solver': the object store cannot issue its namespace grant: credentials expire first",
    ):
        await _step(enable_agent_changelog=True)._configure_agent_changelog(
            "https://agent",
            _card(enable_legacy=enable_legacy),
            _run_context(),
            expires_in=28_800,
        )

    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_legacy", [True, False], ids=["dual", "portable-only"])
async def test_enable_on_a_store_without_grants_is_refused(monkeypatch, store, enable_legacy):
    store.supports_transfer_grants = False
    requests = _install(monkeypatch, {"roots": ["workspace"]})

    with pytest.raises(RuntimeError, match="does not issue transfer grants"):
        await _step(enable_agent_changelog=True)._configure_agent_changelog(
            "https://agent", _card(enable_legacy=enable_legacy), _run_context(), expires_in=7_200
        )

    assert not requests
    assert store.granted == []


@pytest.mark.asyncio
async def test_enable_is_refused_for_an_agent_without_the_object_form(monkeypatch, store):
    requests = _install(monkeypatch, {"roots": ["workspace"]})

    with pytest.raises(RuntimeError, match="does not advertise the object form"):
        await _step(enable_agent_changelog=True)._configure_agent_changelog(
            "https://agent", _card(enable_objects=False), _run_context(), expires_in=7_200
        )

    assert not requests


@pytest.mark.asyncio
async def test_apply_lists_validates_and_sends_ordered_read_grants(monkeypatch, store):
    namespace = _increments(store, "000002.tar", "000000.tar", "000001.tar")
    requests = _install(monkeypatch, {"count": 2, "context_id": "continued"})
    listed_on: list[int] = []
    list_at = store.list_at
    monkeypatch.setattr(
        store, "list_at", lambda url: (listed_on.append(threading.get_ident()), list_at(url))[1]
    )
    step = _step(
        agent_changelog_object_url=namespace,
        agent_changelog_toolcall_position_exclusive=2,
        agent_snapshot_target_context_id="continued",
    )
    context = TaskStepContext()

    event_loop_thread = threading.get_ident()
    await step._apply_agent_changelog("https://agent", _card(), context)

    sent = requests[0]["json"]
    assert [item["sequence"] for item in sent["increments"]] == [0, 1]
    assert {item["object"]["media_type"] for item in sent["increments"]} == {
        "application/octet-stream"
    }
    assert sent["resume_conversation"] is True
    assert sent["target_context_id"] == "continued"
    assert _granted_names(store) == ["000000.tar", "000001.tar"]
    assert context.metadata["agent_changelog_rewinds"][0]["transfer_mode"] == "objects"
    assert listed_on and listed_on[0] != event_loop_thread


@pytest.mark.asyncio
async def test_apply_uses_absolute_tool_call_positions_for_the_cutoff(monkeypatch, store):
    namespace = _increments(store, "000007.tar", "000002.tar", "000004.tar")
    requests = _install(monkeypatch, {"count": 2})
    step = _step(
        agent_changelog_object_url=namespace,
        agent_changelog_toolcall_position_exclusive=7,
    )

    await step._apply_agent_changelog("https://agent", _card(), TaskStepContext())

    assert [item["sequence"] for item in requests[0]["json"]["increments"]] == [2, 4]
    assert _granted_names(store) == ["000002.tar", "000004.tar"]


@pytest.mark.asyncio
async def test_apply_allows_a_cutoff_beyond_the_last_tool_call(monkeypatch, store):
    namespace = _increments(store, "000002.tar", "000004.tar")
    requests = _install(monkeypatch, {"count": 2})
    step = _step(
        agent_changelog_object_url=namespace,
        agent_changelog_toolcall_position_exclusive=99,
    )

    await step._apply_agent_changelog("https://agent", _card(), TaskStepContext())

    assert [item["sequence"] for item in requests[0]["json"]["increments"]] == [2, 4]


@pytest.mark.asyncio
async def test_apply_rejects_duplicate_tool_call_positions(monkeypatch, store):
    namespace = _increments(store, "000002.tar", "000002.json")
    requests = _install(monkeypatch, {"count": 2})

    with pytest.raises(ValueError, match="strictly increasing"):
        await _step(agent_changelog_object_url=namespace)._apply_agent_changelog(
            "https://agent", _card(), TaskStepContext()
        )
    assert not requests


@pytest.mark.asyncio
async def test_apply_rejects_unsequenced_objects_in_a_portable_namespace(monkeypatch, store):
    namespace = _increments(store, "manifest.json", "workspace.tar.gz")
    requests = _install(monkeypatch, {"count": 2})

    with pytest.raises(ValueError, match="zero-padded sequence names"):
        await _step(agent_changelog_object_url=namespace)._apply_agent_changelog(
            "https://agent", _card(), TaskStepContext()
        )
    assert not requests
    assert store.granted == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names, cutoff",
    [((), None), (("000000.tar", "000001.tar"), 0)],
    ids=["empty-namespace", "cutoff-before-the-first-tool-call"],
)
async def test_apply_sends_the_empty_baseline_when_no_increment_is_selected(
    monkeypatch, store, names, cutoff
):
    namespace = _increments(store, *names)
    requests = _install(monkeypatch, {"count": 0, "context_id": "baseline"})
    context = TaskStepContext()
    step = _step(
        agent_changelog_object_url=namespace,
        agent_changelog_toolcall_position_exclusive=cutoff,
    )

    await step._apply_agent_changelog("https://agent", _card(), context)

    assert requests[0]["json"]["increments"] == []
    assert store.granted == []
    assert context.metadata["agent_changelog_rewinds"][0]["context_id"] == "baseline"


_APPLY_OBJECTS = {"request": {"required": ["increments"]}}


def _apply_call(store, source_url):
    return changelog_apply_call(
        _APPLY_OBJECTS, store, agent_name="solver", source_url=source_url
    )


def _limit_changelog(monkeypatch, **limits):
    monkeypatch.setattr(
        object_transfer,
        "CHANGELOG_LIMITS",
        ObjectLimits(**{**vars(object_transfer.CHANGELOG_LIMITS), **limits}),
    )


def test_apply_rejects_too_many_increments_before_issuing_grants(monkeypatch, store):
    namespace = _increments(store, "000000.tar", "000001.tar")
    _limit_changelog(monkeypatch, max_objects=1)

    with pytest.raises(ValueError, match="the limit is 1"):
        _apply_call(store, namespace)

    assert store.granted == []


def test_apply_rejects_a_namespace_over_the_aggregate_limit(monkeypatch, store):
    namespace = _increments(store, "000000.tar", "000001.tar", data=b"sixsix")
    _limit_changelog(monkeypatch, max_total_bytes=10)

    with pytest.raises(ValueError, match="10-byte limit"):
        _apply_call(store, namespace)


def test_apply_counts_empty_increments_as_capture_does(monkeypatch, store):
    store.put(f"{NAMESPACE_KEY}/000000.tar", b"")
    store.put(f"{NAMESPACE_KEY}/000001.tar", b"ten bytes!")
    _limit_changelog(monkeypatch, max_total_bytes=10)

    call = _apply_call(store, store.object_url(NAMESPACE_KEY))

    assert [item["object"]["size_bytes"] for item in call.payload["increments"]] == [0, 10]


@pytest.mark.asyncio
async def test_a_source_is_not_sent_to_an_agent_without_the_object_apply_form(monkeypatch, store):
    namespace = _increments(store, "000000.tar")
    requests = _install(monkeypatch, {"ok": True, "count": 1})

    with pytest.raises(RuntimeError, match="does not advertise the object form"):
        await _step(agent_changelog_object_url=namespace)._apply_agent_changelog(
            "https://agent", _card(apply_objects=False), TaskStepContext()
        )

    assert not requests
    assert store.granted == []


def test_a_changelog_source_round_trips():
    step = _step(
        agent_changelog_object_url="s3://artifact-bucket/agent_changelog/run-1/solver",
        agent_changelog_toolcall_position_exclusive=2,
    )

    restored = DeployAgentTaskStep.from_dict(step.to_dict())

    assert restored.agent_changelog_object_url == step.agent_changelog_object_url
    assert restored.agent_changelog_toolcall_position_exclusive == 2


def test_a_cutoff_needs_a_changelog_source():
    with pytest.raises(ValueError, match="requires agent_changelog_object_url"):
        _step(agent_changelog_toolcall_position_exclusive=2)
