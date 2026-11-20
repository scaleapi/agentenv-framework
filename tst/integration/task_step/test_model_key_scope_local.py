"""The configured model key reaches only the configured endpoint. Two stand-in model endpoints, the configured one
and another, record the key each call brings: the direct judge calls them over HTTP, and an echo agent
(tst/data/a2a_agent) is deployed on a local Docker sandbox and its container's environment read back. A URL other
than the configured one with no key of its own is refused before anything is called or started. Builds the echo
agent; needs Docker."""

import json
import shutil
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.sandbox_provider import (
    reset_agent_sandbox_provider,
    reset_env_sandbox_provider,
    reset_sandbox_provider,
)
from agent_env.task import Task
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import PromptResponse, TaskStepContext
from tst.util.a2a_test_agent import build_test_agent_image
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

CONFIGURED_KEY = "configured-key"
REFUSED = "isn't the configured model endpoint, so the configured key isn't sent there"
VERDICT = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "it answered"}]})


class _Endpoint(BaseHTTPRequestHandler):
    """An OpenAI-style chat completions endpoint that records the Authorization each call brings."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.server.calls.append(self.headers.get("Authorization"))
        reply = json.dumps({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "judge",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": VERDICT}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


def _serve() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    server.calls = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _reset():
    for reset in (reset_config, reset_artifact_store, reset_sandbox_provider, reset_env_sandbox_provider,
                  reset_agent_sandbox_provider):
        reset()


@pytest.fixture
def endpoints(monkeypatch, tmp_path):
    """The configured endpoint and another, in local stores whose ``[model]`` is the first and its key."""
    configured, other = _serve(), _serve()
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text(f'[model]\nbase_url = "http://127.0.0.1:{configured.server_port}/v1"\n'
                      f'api_key = "{CONFIGURED_KEY}"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    _reset()
    configure()
    try:
        yield configured, other, sandboxes
    finally:
        configured.shutdown()
        other.shutdown()
        _reset()


def _judge_run(overrides: dict) -> tuple[TaskStepContext, Task]:
    task = Task.from_dict({"id": f"key-scope-{uuid.uuid4().hex[:8]}", "steps": [{
        "id": "judge", "type": "rubrics_verifier", "verifier_id": "judge", "prompt_id": "hello",
        "use_agent_judge": False, "use_trajectory": False, "default_model": "openai/judge",
        "criteria": [{"id": "c1", "description": "it says hello"}]}]})
    return TaskStepContext(prompt_responses=[PromptResponse(prompt_id="hello", response="hello", prompt_text="Hi.")],
                           metadata={"user_overrides": overrides}), task


@pytest.mark.asyncio
async def test_the_judge_sends_the_configured_key_only_to_the_configured_endpoint(endpoints):
    configured, other, _ = endpoints
    elsewhere = f"http://127.0.0.1:{other.server_port}"

    context, task = _judge_run({"judge_litellm_base_url": elsewhere})
    with pytest.raises(Exception, match=REFUSED):
        await task.run(context=context)
    assert configured.calls == [] and other.calls == []

    context, task = _judge_run({"judge_litellm_base_url": elsewhere, "judge_litellm_api_key": "theirs"})
    await task.run(context=context)
    assert other.calls == ["Bearer theirs"] and configured.calls == []

    context, task = _judge_run({"judge_litellm_base_url": f"http://127.0.0.1:{configured.server_port}/"})
    await task.run(context=context)
    assert configured.calls == [f"Bearer {CONFIGURED_KEY}"] and other.calls == ["Bearer theirs"]


def _container_env(sandbox_id: str) -> dict[str, str]:
    """The environment the agent's container behind ``sandbox_id`` was started with: the one named after it."""
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True,
                           check=True).stdout.split()
    [container] = [name for name in names if sandbox_id in name]
    inspected = subprocess.run(["docker", "inspect", "--format", "{{json .Config.Env}}", container],
                               capture_output=True, text=True, check=True).stdout
    return dict(pair.split("=", 1) for pair in json.loads(inspected))


@pytest.mark.asyncio
async def test_an_agent_sent_elsewhere_gets_its_own_key_or_is_refused_before_its_container_starts(endpoints):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    _, other, sandboxes = endpoints
    elsewhere = f"http://127.0.0.1:{other.server_port}"
    agent_id = f"key-scope-{uuid.uuid4().hex[:8]}"
    A2AAgent.put(id=agent_id, docker_image_artifact=build_test_agent_image(f"{agent_id}-image"))
    task = Task.from_dict({"id": agent_id, "steps": [{
        "id": "solver", "type": "deploy_agent", "agent_name": "solver", "env_ids": [], "a2a_agent_id": agent_id,
        "sandbox_type": "local", "litellm_base_url": elsewhere}]})

    refused = TaskStepContext()
    with pytest.raises(Exception, match=REFUSED):
        await task.run(context=refused)
    assert refused.deployed_agents == [] and not (sandboxes.exists() and any(sandboxes.iterdir()))

    context = TaskStepContext(metadata={"user_overrides": {"litellm_api_key": "theirs"}})
    try:
        await task.run(context=context)
        [agent] = context.deployed_agents
        env = _container_env(agent.sandbox_id)
        assert (env["LITELLM_API_KEY"], env["LITELLM_BASE_URL"]) == (
            "theirs", f"http://host.docker.internal:{other.server_port}")
        assert CONFIGURED_KEY not in env.values() and "LITELLM_EXTRA_HEADERS" not in env
    finally:
        await teardown_run(context)
    assert other.calls == []
