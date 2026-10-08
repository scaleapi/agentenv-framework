from __future__ import annotations

from types import SimpleNamespace

import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.config import set_object_store
from agent_env.store import routing
from agent_env.store.object_store import s3_object_store
from agent_env.store.object_store.s3_object_store import S3ObjectStore
from agent_env.task_step.task_steps.snapshot_env import _push_s3_credentials

_CARD = {"capabilities": {"extensions": [{"uri": "urn:agentenv:add-s3-credentials/v1", "params": {"endpoint": "/x"}}]}}


@pytest.fixture
def pushed(monkeypatch):
    calls: list[dict] = []

    async def fake_invoke(base_url, card, uri, params=None, timeout=30):
        calls.append(params)

    monkeypatch.setattr(protocol_v1, "invoke_extension", fake_invoke)
    return calls


@pytest.fixture
def host_credentials(monkeypatch):
    frozen = SimpleNamespace(access_key="AKIA-HOST", secret_key="host-secret", token="tok")
    creds = SimpleNamespace(get_frozen_credentials=lambda: frozen)
    monkeypatch.setattr(s3_object_store.boto3, "Session", lambda: SimpleNamespace(get_credentials=lambda: creds))


def _merged_env(caller_env: dict[str, str] | None = None) -> dict[str, str]:
    return A2AAgent._build_merged_env(SimpleNamespace(default_env_vars={}), caller_env or {}, 8000)


@pytest.mark.asyncio
@pytest.mark.parametrize("s3", [False, True], ids=["local-store", "s3-store"])
async def test_nothing_is_shared_by_default(s3, host_credentials, pushed):
    if s3:
        set_object_store(S3ObjectStore(object(), "bucket", region="us-west-2"))
    await _push_s3_credentials("https://svc", _CARD, 30)
    assert not any(k.startswith("AWS_") for k in _merged_env())
    assert pushed == []


@pytest.mark.asyncio
async def test_an_opted_in_s3_store_shares_the_frozen_chain(host_credentials, pushed):
    set_object_store(S3ObjectStore(object(), "bucket", region="us-west-2", share_credentials=True))
    await _push_s3_credentials("https://svc", _CARD, 30)
    assert {k: v for k, v in _merged_env().items() if k.startswith("AWS_")} == {
        "AWS_ACCESS_KEY_ID": "AKIA-HOST",
        "AWS_SECRET_ACCESS_KEY": "host-secret",
        "AWS_SESSION_TOKEN": "tok",
        "AWS_DEFAULT_REGION": "us-west-2",
    }
    assert pushed == [{
        "aws_access_key_id": "AKIA-HOST",
        "aws_secret_access_key": "host-secret",
        "aws_session_token": "tok",
        "region_name": "us-west-2",
        "bucket": "bucket",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner, shared", [("@local/~/b/tasks/t", False), ("registry-task", True)])
async def test_an_local_run_hands_a_service_no_credentials_to_upload_a_snapshot_with(owner, shared, host_credentials, pushed, cli_routing):
    set_object_store(S3ObjectStore(object(), "bucket", region="us-west-2", share_credentials=True))
    with routing.run_scope(owner):
        await _push_s3_credentials("https://svc", _CARD, 30)
    assert [call["bucket"] for call in pushed] == (["bucket"] if shared else [])


def test_a_caller_supplied_key_suppresses_the_whole_shared_set(host_credentials):
    set_object_store(S3ObjectStore(object(), "bucket", region="us-west-2", share_credentials=True))
    merged = _merged_env({"AWS_ACCESS_KEY_ID": "AKIA-CALLER", "AWS_SECRET_ACCESS_KEY": "caller-secret"})
    assert merged["AWS_ACCESS_KEY_ID"] == "AKIA-CALLER"
    assert "AWS_SESSION_TOKEN" not in merged
