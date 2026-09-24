"""Custom Env types via .agentenv/config.toml [envs].impls, against the configured document store.

The unit tests (tst/unit/env/test_registry.py) prove the registry merge with a
FakeDocumentStore whose query()/count() raise NotImplementedError. This is the
first real-service coverage of the config.toml -> registry -> store path: a custom
Env registered only via [envs].impls round-trips through the real store
(serialization, versioning, query-by-type); a genuinely-persisted doc is unreadable
without the config ("Unknown env type"); and the custom deploy() runs through the
real DeployEnvTaskStep. Fast tier (store only, no sandbox).
"""

from __future__ import annotations

import asyncio
import textwrap
import uuid

import pytest

from agent_env.env.env import DeployedEnv, Env
from agent_env.env.store import reset_env_store
from agent_env.config import Config, configure, get_config, reset_config
from agent_env.store.document_store import Filter
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep

pytestmark = pytest.mark.integration

_HERE = "tst.integration.env.custom_env_config_test"


class _CustomEnv(Env):
    type = "custom_env_integ_test"

    def __init__(self, id, version=None, metadata=None, flavor="plain"):
        super().__init__(id=id, version=version, metadata=metadata)
        self.flavor = flavor

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["flavor"] = self.flavor
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "_CustomEnv":
        return cls(id=data["id"], version=data.get("version"),
                   metadata=data.get("metadata"), flavor=data.get("flavor", "plain"))

    async def deploy(self, **kwargs) -> DeployedEnv:
        return DeployedEnv(
            env_id=self.id, env_version=self.version or 1,
            gateway_url=f"https://gw.example/{self.id}",
            mcp_url=f"https://mcp.example/{self.id}",
            db_web_url=None, sandbox_id=f"sbx-{self.id}",
            metadata={"deployed_by": type(self).__name__, "flavor": self.flavor},
        )


def _unique_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _write_env_config(tmp_path) -> str:
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(textwrap.dedent(f"""
        [envs]
        impls = ["{_HERE}:_CustomEnv"]
    """))
    return str(cfg)


def _delete_env(env_id: str) -> None:
    get_config().get_document_store().delete("envs", Filter.of(id=env_id))


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_config()
    reset_env_store()
    yield
    reset_config()
    reset_env_store()


@pytest.fixture(autouse=True)
def _mongodb_uri_for_local_secret_store(monkeypatch):
    # The tests' tmp configs carry no [stores.secret]; the local default store resolves
    # secret:mongodb_uri from this env var.
    uri = Config().get_secret_store().get("mongodb_uri")
    reset_config()
    configure()
    if uri:
        monkeypatch.setenv("mongodb_uri", uri)


def test_custom_env_round_trips_and_versions_through_dev_mongo(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ENV_CONFIG", _write_env_config(tmp_path))
    env_id = _unique_id("custom_env")
    try:
        saved = _CustomEnv.put(id=env_id, flavor="spicy")
        assert saved.version == 1

        loaded = Env.get(env_id)
        assert type(loaded) is _CustomEnv
        assert loaded.flavor == "spicy"

        saved2 = _CustomEnv.put(id=env_id, flavor="mild")
        assert saved2.version == 2
        assert Env.get(env_id, version=1).flavor == "spicy"
        assert Env.get(env_id).version == 2

        results = Env.query().type("custom_env_integ_test").execute()
        mine = [e for e in results if e.id == env_id]
        assert mine and type(mine[0]) is _CustomEnv
    finally:
        _delete_env(env_id)


def test_persisted_doc_is_unknown_without_config(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    env_id = _unique_id("custom_env")
    try:
        _CustomEnv.put(id=env_id, flavor="spicy")
        with pytest.raises(ValueError, match="Unknown env type"):
            Env.get(env_id)
    finally:
        _delete_env(env_id)


def test_custom_env_deploys_through_deploy_env_step(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ENV_CONFIG", _write_env_config(tmp_path))
    env_id = _unique_id("custom_env")
    try:
        _CustomEnv.put(id=env_id, flavor="spicy")
        step = DeployEnvTaskStep(id=_unique_id("deploy-step"), version=None, env_id=env_id)
        ctx = asyncio.run(step.execute(TaskStepContext()))
        assert len(ctx.deployed_envs) == 1
        dep = ctx.deployed_envs[0]
        assert dep.env_id == env_id
        assert dep.metadata["deployed_by"] == "_CustomEnv"
    finally:
        _delete_env(env_id)
