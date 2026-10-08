"""Portable skill-bundle negotiation in ``A2AAgent.add_skill`` and ``register_skill``."""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent, DeployedA2AAgent, object_transfer
from agent_env.a2a_agent.object_transfer import ObjectLimits, skill_bundle_request
from agent_env.artifact import SkillArtifact
from agent_env.config import configure
from agent_env.task_step.task_steps.add_skills import Skill
from agent_env.task_step.task_steps.deploy_agent import _skill_fields
from tst.unit.event_loop_probe import on_event_loop
from tst.util.granting_object_store import GrantingObjectStore


def _card(
    *, bundle: bool, legacy: bool | None = None, config_key: str = "params"
) -> dict:
    if legacy is None:
        legacy = not bundle
    variants = [{"required": ["skill_md"]}]
    if bundle:
        variants.append({"required": ["skill_bundle"]})
    if legacy:
        variants.append({"required": ["skill_s3_url"]})
    request = {
        "required": ["name", "description"],
        "oneOf": variants,
    }
    return {
        "capabilities": {
            "extensions": [
                {
                    "uri": A2AAgent.EXT_SKILL_CONFIG,
                    config_key: {
                        "endpoint": "/custom/skills",
                        "methods": {
                            "add": {
                                "request": request,
                            }
                        },
                    },
                }
            ]
        }
    }


def _deployed(
    *,
    bundle: bool,
    legacy: bool | None = None,
    config_key: str = "params",
    sandbox_type: str = "local",
) -> DeployedA2AAgent:
    return DeployedA2AAgent(
        agent_id="agent",
        agent_version=1,
        a2a_url="https://agent.example.test",
        sandbox_id="sandbox",
        agent_card=_card(bundle=bundle, legacy=legacy, config_key=config_key),
        sandbox_type=sandbox_type,
    )


@pytest.fixture
def store(tmp_path) -> GrantingObjectStore:
    store = GrantingObjectStore(str(tmp_path))
    store.put(
        "artifacts/skill/review/1/SKILL.md",
        b"# review ok\n",
        content_type="text/markdown; charset=utf-8",
    )
    store.put("artifacts/skill/review/1/scripts/check.py", b"", content_type="text/x-python")
    configure(object_store=store)
    return store


@pytest.fixture
def prefix(store: GrantingObjectStore) -> str:
    return store.object_url("artifacts/skill/review/1") + "/"


class _Agent:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.status_code = 200


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> _Agent:
    monkeypatch.setattr(Skill, "validate", lambda self: None)
    agent = _Agent()

    async def post(self, url, *, json, timeout):
        agent.requests.append({"url": url, "json": json, "timeout": timeout})
        body = (
            {"status": "added", "name": json["name"], "source": "s3"}
            if agent.status_code < 400
            else {"detail": "skill_md has no frontmatter"}
        )
        return httpx.Response(agent.status_code, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return agent


@pytest.mark.asyncio
async def test_add_skill_negotiates_bundle_from_legacy_config_spelling(
    store: GrantingObjectStore, prefix: str, agent: _Agent
) -> None:
    await A2AAgent.add_skill(
        _deployed(bundle=True, config_key="config"),
        Skill(name="review", description="Review work", object_url=prefix),
    )

    assert agent.requests[0]["url"] == "https://agent.example.test/custom/skills"
    assert set(agent.requests[0]["json"]) == {"name", "description", "skill_bundle"}


@pytest.mark.asyncio
async def test_artifact_skill_prefers_portable_bundle_when_advertised(
    store: GrantingObjectStore,
    prefix: str,
    agent: _Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SkillArtifact,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            skill_name="review",
            description="Review work",
            skill_object_url=prefix,
        ),
    )
    listed_on: list[int] = []
    list_at = store.list_at
    monkeypatch.setattr(
        store, "list_at", lambda url: (listed_on.append(threading.get_ident()), list_at(url))[1]
    )

    event_loop_thread = threading.get_ident()
    await A2AAgent.add_skill(
        _deployed(bundle=True),
        Skill(skill_artifact_id="review", skill_artifact_version=1),
    )

    (sent,) = agent.requests
    assert sent["url"] == "https://agent.example.test/custom/skills"
    assert set(sent["json"]) == {"name", "description", "skill_bundle"}
    bundle = sent["json"]["skill_bundle"]
    assert bundle["max_total_bytes"] == 13
    assert [item["path"] for item in bundle["files"]] == [
        "SKILL.md",
        "scripts/check.py",
    ]
    skill_md, script = (item["object"] for item in bundle["files"])
    assert (skill_md["media_type"], skill_md["max_bytes"], skill_md["size_bytes"]) == (
        "text/markdown", 12, 12,
    )
    assert (script["max_bytes"], script["size_bytes"]) == (1, 0)
    assert skill_md["read"]["url"].endswith("/artifacts/skill/review/1/SKILL.md?sig=read")
    assert listed_on and listed_on[0] != event_loop_thread


@pytest.mark.asyncio
async def test_an_object_skill_is_refused_by_an_agent_without_the_bundle_form(
    store: GrantingObjectStore, prefix: str, agent: _Agent
) -> None:
    with pytest.raises(RuntimeError, match="skill add: the agent does not advertise the object form"):
        await A2AAgent.add_skill(
            _deployed(bundle=False),
            Skill(name="review", description="Review work", object_url=prefix),
        )

    assert agent.requests == []
    assert store.granted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [True, False], ids=["dual", "bundle-only"])
async def test_an_object_skill_on_a_store_without_grants_is_refused_before_any_request(
    store: GrantingObjectStore, prefix: str, agent: _Agent, legacy: bool
) -> None:
    store.supports_transfer_grants = False

    with pytest.raises(RuntimeError, match="does not issue transfer grants"):
        await A2AAgent.add_skill(
            _deployed(bundle=True, legacy=legacy),
            Skill(name="review", description="Review work", object_url=prefix),
        )

    assert agent.requests == []


@pytest.mark.asyncio
async def test_an_object_skill_is_refused_when_the_stores_grants_do_not_reach_the_agent(
    store: GrantingObjectStore, prefix: str, agent: _Agent
) -> None:
    with pytest.raises(RuntimeError, match="grants do not reach agents on the 'modal' sandbox provider"):
        await A2AAgent.add_skill(
            _deployed(bundle=True, sandbox_type="modal"),
            Skill(name="review", description="Review work", object_url=prefix),
        )

    assert agent.requests == []
    assert store.granted == []


@pytest.mark.asyncio
async def test_inline_skill_stays_inline_when_bundle_is_advertised(
    store: GrantingObjectStore, agent: _Agent
) -> None:
    await A2AAgent.add_skill(
        _deployed(bundle=True),
        Skill(name="review", description="Review work", body="Use a reviewer."),
    )

    assert set(agent.requests[0]["json"]) == {"name", "description", "skill_md"}
    assert store.granted == []


@pytest.mark.asyncio
async def test_inline_skill_does_not_require_bundle_support(
    store: GrantingObjectStore, agent: _Agent
) -> None:
    await A2AAgent.add_skill(
        _deployed(bundle=False),
        Skill(name="review", description="Review work", body="Use a reviewer."),
    )

    assert set(agent.requests[0]["json"]) == {"name", "description", "skill_md"}
    assert store.granted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["object_url", "skill_s3_url", "s3_uri"])
async def test_a_deploy_skill_dict_naming_an_object_negotiates_a_portable_bundle(
    store: GrantingObjectStore, prefix: str, agent: _Agent, key: str
) -> None:
    """``object_url``, or either of its older names, which stored steps may still carry."""
    await A2AAgent.register_skill(
        _deployed(bundle=True),
        **_skill_fields({"name": "review", "description": "Review work", key: prefix}),
    )

    assert set(agent.requests[0]["json"]) == {"name", "description", "skill_bundle"}


@pytest.mark.asyncio
async def test_legacy_deploy_inline_skill_dict_stays_inline(
    store: GrantingObjectStore, agent: _Agent
) -> None:
    await A2AAgent.register_skill(
        _deployed(bundle=True),
        **_skill_fields(
            {
                "name": "review",
                "description": "Review work",
                "content": "# Existing skill document",
            }
        ),
    )

    assert agent.requests[0]["json"] == {
        "name": "review",
        "description": "Review work",
        "skill_md": "# Existing skill document",
    }
    assert store.granted == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured, sent",
    [
        (
            {"name": "review", "skill_md": "# Inline", "s3_uri": "s3://artifact-bucket/skill/"},
            {"name": "review", "description": "", "skill_md": "# Inline"},
        ),
        ({"name": "review"}, {"name": "review", "description": ""}),
    ],
    ids=["skill-md-wins-over-an-object-source", "no-content-source"],
)
async def test_deploy_skill_dicts_keep_their_established_shapes(
    store: GrantingObjectStore, agent: _Agent, configured: dict, sent: dict
) -> None:
    await A2AAgent.register_skill(_deployed(bundle=False), **_skill_fields(configured))

    assert agent.requests[0]["json"] == sent


@pytest.mark.asyncio
async def test_an_inline_skill_error_keeps_the_agents_detail(
    store: GrantingObjectStore, agent: _Agent
) -> None:
    agent.status_code = 400

    with pytest.raises(httpx.HTTPStatusError, match="skill_md has no frontmatter"):
        await A2AAgent.add_skill(
            _deployed(bundle=False),
            Skill(name="review", description="Review work", body="Check it."),
        )


@pytest.mark.asyncio
async def test_a_bundle_skill_error_keeps_the_agents_body_out_of_logs(
    store: GrantingObjectStore, prefix: str, agent: _Agent, caplog: pytest.LogCaptureFixture
) -> None:
    agent.status_code = 400

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await A2AAgent.add_skill(
            _deployed(bundle=True),
            Skill(name="review", description="Review work", object_url=prefix),
        )

    assert str(raised.value) == "skill add (review) failed with HTTP 400"
    assert "skill_s3_url must be" not in caplog.text


@pytest.mark.parametrize("path", ["missing", "review/1/SKILL.md"], ids=["no-objects", "a-file"])
def test_a_bundle_url_with_no_objects_under_it_is_named(
    store: GrantingObjectStore, path: str
) -> None:
    object_url = store.object_url(f"artifacts/skill/{path}")

    with pytest.raises(ValueError, match=f"no objects under {re.escape(object_url)}"):
        skill_bundle_request(store, name="review", description="Review work", object_url=object_url)


def test_portable_bundle_is_bounded_by_the_skill_limits(
    store: GrantingObjectStore, prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        object_transfer,
        "SKILL_BUNDLE_LIMITS",
        ObjectLimits(max_objects=1, max_object_bytes=1024, max_total_bytes=1024),
    )

    with pytest.raises(ValueError, match="limit is 1"):
        skill_bundle_request(store, name="review", description="Review work", object_url=prefix)

    assert store.granted == []


@pytest.mark.asyncio
async def test_an_agent_without_skill_config_is_refused_before_the_skill_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def validate(self):
        raise AssertionError("validated a skill the agent cannot take")

    monkeypatch.setattr(Skill, "validate", validate)
    deployed = DeployedA2AAgent(
        agent_id="agent", agent_version=1, a2a_url="https://agent", sandbox_id="s", agent_card={}
    )

    with pytest.raises(RuntimeError, match="does not advertise"):
        await A2AAgent.add_skill(deployed, Skill(name="review", description="Review work", body="b"))


@pytest.mark.asyncio
async def test_an_s3_url_skill_s_skill_md_is_read_off_the_event_loop(
    store: GrantingObjectStore, prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.put(
        "artifacts/skill/review/1/SKILL.md",
        Skill(name="review", description="Review work", body="Check it.").to_skill_md().encode(),
        content_type="text/markdown; charset=utf-8",
        allow_overwrite=True,
    )
    on_loop: list[bool] = []
    get = store.get
    monkeypatch.setattr(store, "get", lambda url: (on_loop.append(on_event_loop()), get(url))[1])

    async def post(self, url, *, json, timeout):
        return httpx.Response(200, json={"status": "added", "name": json["name"]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    await A2AAgent.add_skill(_deployed(bundle=True), Skill(name="review", description="Review work", object_url=prefix))

    assert on_loop and not any(on_loop)
