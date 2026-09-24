"""``_capture_universe_state`` publishes unsigned ``s3://`` refs to
``context.metadata['snapshot_json_url']``, not presigned URLs."""
from __future__ import annotations

import json

import pytest

from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import snapshot_agent_state as mod
from agentenv_protocol import client as protocol_v1

_PREFIX = "s3://artifact-bucket/agent_snapshots/oc_post_run_workspace_T/3-deadbeef/"


class _StubS3:
    def __init__(self):
        self.puts: list[dict] = []

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803
        self.puts.append({"Bucket": Bucket, "Key": Key})

    def generate_presigned_url(self, *a, **k):
        raise AssertionError(
            "snapshot_agent_state must not presign — publish s3:// and let the "
            "run_openclaw_unit_test consumer re-sign"
        )


class _StubServiceArtifact:
    def __init__(self, name):
        self.environment_name = name


class _StubUniverse:
    def get_environment_artifacts(self):
        return [_StubServiceArtifact("calendar"), _StubServiceArtifact("contacts")]


def _step():
    return mod.SnapshotAgentStateTaskStep(
        id="t-capture",
        version=None,
        artifact_id="oc_post_run_workspace_T",
        prompt_id="main",
        agent_name="openclaw-cli",
        env_id="env-1",
        universe_artifact_id="uni-1",
    )


def _ctx():
    ctx = TaskStepContext()
    ctx.deployed_envs = [
        DeployedEnv(
            env_id="env-1",
            env_version=1,
            gateway_url="https://gw",
            mcp_url="m",
            db_web_url=None,
            sandbox_id="sb-1",
        )
    ]
    return ctx


@pytest.mark.asyncio
async def test_capture_universe_state_publishes_unsigned_s3_urls(monkeypatch):
    stub_s3 = _StubS3()

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: stub_s3)

    import agent_env.artifact as artifact_mod

    class _UniClass:
        @staticmethod
        def get(_id):
            return _StubUniverse()

    monkeypatch.setattr(artifact_mod, "EnvironmentUniverseArtifact", _UniClass)

    import agent_env.config as config_mod

    class _Cfg:
        def get_s3_region(self):
            return "us-west-2"

    monkeypatch.setattr(config_mod, "get_config", lambda: _Cfg())

    monkeypatch.setattr(
        legacy_protocol, "environment_base_url", lambda gw, name, mcp=True: f"{gw}/svc/mcp-{name}"
    )

    async def _supports_v1(_base):
        return False

    monkeypatch.setattr(protocol_v1, "supports_v1", _supports_v1)

    async def _export_state(_gw, name):
        return {"service": name, "rows": 1}

    monkeypatch.setattr(legacy_protocol, "export_state", _export_state)

    ctx = _ctx()
    await _step()._capture_universe_state(ctx, _PREFIX)

    published = json.loads(ctx.metadata["snapshot_json_url"])
    assert published == {
        "calendar": f"{_PREFIX}services/calendar.json",
        "contacts": f"{_PREFIX}services/contacts.json",
    }
    assert all(url.startswith("s3://") and "?" not in url for url in published.values())
    assert {p["Key"].rsplit("/", 1)[-1] for p in stub_s3.puts} == {"calendar.json", "contacts.json"}
