"""Two concurrent deploy_env steps in one task, on the local backend.

Without per-deployment host ports this fails at the second `docker compose up` with
"port is already allocated": both env stacks try to publish the same reserved ports.
Per-VM backends are unaffected -- each deploy gets its own network namespace.

Note the explicit ``depends_on=[]``: a missing ``depends_on`` means "depends on every
prior step", which would serialise the branches and prove nothing.
"""

import json
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

from agent_env.artifact import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config, set_image_store
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.task import Task
from agent_env.task.store import get_task_instance_store
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.store.object_store.local.grant_server import grant_server
from tst.task import journal_invariants as journal

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

_TST_DATA = Path(__file__).resolve().parents[2] / "data"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _wait_ready(url: str, codes: tuple[int, ...], timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code in codes:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture
def local_stack(monkeypatch, tmp_path):
    if _docker("info").returncode != 0:
        pytest.skip("docker daemon not available")

    reg_port = _free_port()
    reg_name = f"par-e2e-registry-{uuid.uuid4().hex[:8]}"
    if _docker("run", "-d", "--rm", "--name", reg_name,
               "-p", f"127.0.0.1:{reg_port}:5000", "registry:2").returncode != 0:
        pytest.skip("could not start registry:2")
    host = f"localhost:{reg_port}"
    if not _wait_ready(f"http://{host}/v2/", (200, 401)):
        _docker("rm", "-f", reg_name)
        pytest.skip("local registry did not become ready")

    monkeypatch.chdir(tmp_path)
    for var in ("DOCUMENT", "OBJECT", "IMAGE"):
        monkeypatch.setenv(f"AGENT_ENV_{var}_STORE", "local")
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield host
    finally:
        for cid in _docker("ps", "-aq", "--filter", "name=agent-env-local-").stdout.split():
            _docker("rm", "-f", cid)
        # Grants handed out here came from this process's grant server, whose certificate is from this test's state
        # root; a later test's containers trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()
        _docker("rm", "-f", reg_name)


def _seed_bootstrap_envs():
    """deploy_env resolves `default-db` + `default` by id; a fresh store has neither."""
    # Import the constants directly: `agent_env.cli.env.gateway` resolves to the click
    # Group re-exported by the package __init__, not to the module.
    from agent_env.cli.env.gateway import GATEWAY_CONTEXT, GATEWAY_DOCKERFILE
    from agent_env.cli.env.service_db import (
        DB_MCP_DOCKERFILE,
        DB_WEB_DOCKERFILE,
        SERVICE_DB_DOCKERFILE,
    )
    from agent_env.env import GatewayEnv
    from agent_env.env.envs.service_db import ServiceDBEnv

    def build(dockerfile: Path, context: Path, tag: str) -> DockerImageArtifact:
        result = _docker("build", "-f", str(dockerfile), "-t", tag, str(context))
        assert result.returncode == 0, f"build {tag} failed: {result.stderr[-2000:]}"
        return DockerImageArtifact.put(id=tag, description="bootstrap", image_name=tag)

    ServiceDBEnv.put(
        id="default-db",
        db_docker_image_artifact=build(SERVICE_DB_DOCKERFILE, SERVICE_DB_DOCKERFILE.parent, "service-db-default-db"),
        db_web_docker_image_artifact=build(DB_WEB_DOCKERFILE, DB_WEB_DOCKERFILE.parent, "db-web-default-db"),
        db_mcp_docker_image_artifact=build(DB_MCP_DOCKERFILE, DB_MCP_DOCKERFILE.parent, "db-mcp-default-db"),
    )
    GatewayEnv.put(id="default", docker_image_artifact=build(GATEWAY_DOCKERFILE, GATEWAY_CONTEXT, "gateway-default"))


@pytest.mark.asyncio
async def test_two_deploy_env_steps_run_concurrently(local_stack):
    _seed_bootstrap_envs()

    uid = uuid.uuid4().hex[:8]
    tag = f"par-slack-{uid}"
    build = _docker("build", "-t", tag, "-f", str(_TST_DATA / "slack_mcp/Dockerfile"), str(_TST_DATA))
    assert build.returncode == 0, f"slack_mcp build failed: {build.stderr[-2000:]}"

    artifact = DockerImageArtifact.put(id=tag, description="parallel e2e", image_name=tag)
    env = MCPServerEnv.put(id=f"par-env-{uid}", docker_image_artifact=artifact,
                           environment_name="slack")

    task = Task.put(id=f"par-task-{uid}", steps=[
        DeployEnvTaskStep(id="deploy_a", version=None, env_id=env.id, env_version=env.version,
                          sandbox_type="local", ttl_seconds=900, depends_on=[]),
        DeployEnvTaskStep(id="deploy_b", version=None, env_id=env.id, env_version=env.version,
                          sandbox_type="local", ttl_seconds=900, depends_on=[]),
    ])
    ctx = await task.run()

    assert not ctx.metadata.get("failed_steps"), ctx.metadata.get("failed_steps")
    assert len(ctx.deployed_envs) == 2, "both branches must produce a deployment"

    # Distinct host ports is the property under test; same-port would have failed above.
    urls = [d.mcp_url for d in ctx.deployed_envs]
    assert len(set(urls)) == 2, f"both envs published the same URL: {urls}"

    # Both are independently reachable, so neither deploy cannibalised the other.
    for url in urls:
        base = url.rsplit("/mcp", 1)[0]
        assert _wait_ready(f"{base}/.well-known/agent-env.json", (200,)), f"{base} not serving"

    # The step journal under real fan-out. `live_context=ctx` is the independent check: nothing writes
    # `context` wholesale on the success path, so an under-reporting diff shows up only here.
    doc = journal.assert_journal_replays(
        ctx.instance_id, expect_steps={"deploy_a", "deploy_b"}, live_context=ctx,
    )
    assert doc["status"] == "completed" and doc["current_step"] == 2

    # Check the survivor against ground truth, not the journal: each DeployedEnv carries its
    # producing step id. The undo touches only the database; the fixture reaps the containers.
    url_of = {
        (d.metadata or {}).get("deploy_step_id"): d.mcp_url for d in ctx.deployed_envs
    }
    store = get_task_instance_store()
    after = store.undo_steps_sync(ctx.instance_id, {"deploy_a"})
    assert [c["step_id"] for c in after["completed_steps"]] == ["deploy_b"]
    assert after["current_step"] == 1
    assert after["status"] == "running" and after["completed_at_utc"] is None
    surviving_urls = [d["mcp_url"] for d in after["context"]["deployed_envs"]]
    assert url_of["deploy_b"] in surviving_urls, "the surviving branch's deployment was rolled back"
    doc = journal.assert_journal_replays(ctx.instance_id, expect_steps={"deploy_b"})
    assert doc["status"] == "running"

    # deploy_a's deployment may survive for exactly one reason: the branch that completed
    # second can have recorded the other's append as its own. Tie survival to that.
    b_ops = json.dumps(next(
        e for e in store.journal_entries_sync(ctx.instance_id) if e["step_id"] == "deploy_b"
    )["ops"])
    if url_of["deploy_a"] in surviving_urls:
        assert url_of["deploy_a"] in b_ops, "deploy_a's deployment survived an undo nothing recorded"
    else:
        assert url_of["deploy_a"] not in b_ops
