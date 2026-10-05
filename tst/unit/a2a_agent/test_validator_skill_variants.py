from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent, object_transfer
from agent_env.config import configure
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator import verify_a2a_skill_config
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_skill_config import (
    VerifyA2ASkillConfigStep,
)
from tst.util.granting_object_store import GrantingObjectStore


class _Agent:
    def __init__(self) -> None:
        self.metadata: dict = {}

    def update_metadata(self, metadata: dict) -> None:
        self.metadata = metadata


class _Client:
    def __init__(self, posts: list[dict], rejected_variants: set[str]) -> None:
        self.posts = posts
        self.rejected_variants = rejected_variants

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, *, timeout):
        return httpx.Response(
            200,
            json={"skills": {"validator-test": {"description": "test"}}},
            request=httpx.Request("GET", url),
        )

    async def post(self, url, *, json, timeout):
        variant = "bundle" if "skill_bundle" in json else "other"
        self.posts.append({"variant": variant, "json": json})
        status_code = 422 if variant in self.rejected_variants else 200
        return httpx.Response(
            status_code,
            json={"status": "added", "name": json["name"], "source": variant},
            request=httpx.Request("POST", url),
        )


def _context(*, bundle: bool) -> TaskStepContext:
    variants = [
        {"required": ["skill_md"]},
        {"required": ["skill_s3_url"]},
    ]
    if bundle:
        variants.append({"required": ["skill_bundle"]})
    context = TaskStepContext()
    context.deployed_agents = [
        DeployedAgent(
            agent_name="agent",
            api_url="https://agent",
            a2a_url="https://agent",
            a2a_card={
                "capabilities": {
                    "extensions": [
                        {
                            "uri": A2AAgent.EXT_SKILL_CONFIG,
                            "params": {
                                "methods": {
                                    "add": {
                                        "request": {
                                            "required": ["name", "description"],
                                            "oneOf": variants,
                                        }
                                    }
                                }
                            },
                        }
                    ]
                }
            },
        )
    ]
    context.metadata["verifications"] = {
        "rubric": {
            "results": [
                {"id": "secret_code_inline", "score": 1.0},
            ]
        }
    }
    return context


@pytest.mark.parametrize(
    ("bundle", "rejected_variants", "expected"),
    [
        (True, set(), {"bundle": True}),
        (False, set(), {"bundle": False}),
        (True, {"bundle"}, {"bundle": False}),
    ],
    ids=["bundle", "no-bundle-form", "bundle-rejected"],
)
def test_validator_reports_the_artifact_variant_actually_used(
    monkeypatch, tmp_path, bundle, rejected_variants, expected
):
    configure(object_store=GrantingObjectStore(str(tmp_path)))
    agent = _Agent()
    posts: list[dict] = []
    monkeypatch.setattr(
        verify_a2a_skill_config.httpx,
        "AsyncClient",
        lambda: _Client(posts, rejected_variants),
    )
    monkeypatch.setattr(A2AAgent, "get", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(
        object_transfer,
        "skill_bundle_request",
        lambda store, **kwargs: SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "name": kwargs["name"],
                "description": kwargs["description"],
                "skill_bundle": {"files": []},
            }
        ),
    )

    result = asyncio.run(
        VerifyA2ASkillConfigStep(
            id="verify",
            version=None,
            a2a_agent_id="agent-1",
            rubric_verifier_id="rubric",
            skill_bundle_object_url="s3://bucket/bundle/",
        ).execute(_context(bundle=bundle))
    )

    verification = result.metadata["verifications"]["a2a_skill_config"]
    assert verification == {"inline": True, "list": True, **expected}
    options = agent.metadata["validated_a2a_extensions"][
        A2AAgent.EXT_SKILL_CONFIG
    ]["methods"]["add"]["options"]
    assert options["skill_bundle"] == {"supported": expected["bundle"]}
    assert "skill_s3_url" not in options
    assert options["skill_md"] == {"supported": True}
    # An agent without the bundle form is never sent the object-backed probe at all.
    assert [post["variant"] for post in posts] == (["bundle"] if bundle else [])
