"""``agent-env run`` runs a bundle's solver, user simulator and judge on the models its flags give, leaving the task as
written. The solver and user simulator are echo agents (tst/data/a2a_agent) on local sandboxes, read back for the model
each was configured with; the judge is the direct LLM judge, calling a stand-in model endpoint that records the model
each call names. A rerun on other models writes nothing. Builds the echo agent; needs Docker."""

import json
import logging
import shutil
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import Outcome, run_bundle
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.sandbox_provider import (
    reset_agent_sandbox_provider,
    reset_env_sandbox_provider,
    reset_sandbox_provider,
)
from tst.util.a2a_test_agent import put_test_agent
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

VERDICT = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "it answered"}]})


class _ModelEndpoint(BaseHTTPRequestHandler):
    """An OpenAI-style chat completions endpoint that records each call's model and passes every criterion."""

    models: list[str] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).models.append(body["model"])
        reply = json.dumps({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
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


def _steps(agent_id: str) -> list[dict]:
    return [
        {"id": "solver", "type": "deploy_agent", "agent_name": "solver", "env_ids": [], "a2a_agent_id": agent_id},
        {"id": "user-sim", "type": "deploy_agent", "agent_name": "user-sim", "env_ids": [], "a2a_agent_id": agent_id},
        {"id": "prompt", "type": "prompt_agent", "agent_name": "solver", "prompt_id": "hello", "prompt": "Say hello.",
         "model": "task-solver", "user_agent_name": "user-sim", "user_model": "task-user", "max_conversation_turns": 2,
         "depends_on": [{"task_step_id": "solver"}, {"task_step_id": "user-sim"}]},
        {"id": "judge", "type": "rubrics_verifier", "verifier_id": "judge", "prompt_id": "hello",
         "use_agent_judge": False, "use_trajectory": False, "default_model": "openai/task-judge",
         "criteria": [{"id": "c1", "description": "it says hello"}], "depends_on": [{"task_step_id": "prompt"}]},
    ]


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    """A bundle whose one task names its own solver, user simulator and judge models, in a local store whose model
    endpoint is the stand-in. Whatever ran, no sandbox work folder may be left behind."""
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), _ModelEndpoint)
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    monkeypatch.setattr(_ModelEndpoint, "models", [])
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text(f'[model]\nbase_url = "http://127.0.0.1:{endpoint.server_port}/v1"\napi_key = "unused"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    _reset()
    configure()
    agent = put_test_agent(f"role-models-{uuid.uuid4().hex[:8]}")
    root = tmp_path / "roles"
    (root / "tasks").mkdir(parents=True)
    (root / "tasks" / "roles.json").write_text(json.dumps(_steps(agent.id), indent=2))
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield root
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        endpoint.shutdown()
        _reset()


def _reset():
    for reset in (reset_config, reset_artifact_store, reset_sandbox_provider, reset_env_sandbox_provider,
                  reset_agent_sandbox_provider):
        reset()


def _configured_model(url: str) -> str | None:
    return httpx.get(f"{url}/ext/agent-config", timeout=30).json()["config"].get("model")


def test_each_role_runs_on_the_model_its_flag_gives_and_a_rerun_on_others_writes_nothing(bundle):
    written = (bundle / "tasks" / "roles.json").read_text()

    result = run_bundle(bundle, model="flag-solver", user_model="flag-user", judge_model="openai/flag-judge",
                        keep=True)
    try:
        [run] = result.runs
        assert run.outcome is Outcome.PASSED, run.error or run.failed_steps
        agents = {agent.agent_name: agent.a2a_url for agent in run.context.deployed_agents}
        assert {name: _configured_model(url) for name, url in agents.items()} == {
            "solver": "flag-solver", "user-sim": "flag-user"}
        assert run.models == {"solver": {"prompt": "flag-solver"}, "user simulator": {"prompt": "flag-user"},
                              "judge": {"judge": "openai/flag-judge"}}
    finally:
        result.teardown()
    assert _ModelEndpoint.models == ["flag-judge"]

    rerun = CliRunner().invoke(cli, ["run", str(bundle), "--model", "other-solver", "--user-model", "other-user",
                                     "--judge-model", "openai/other-judge"])

    assert rerun.exit_code == 0, rerun.output
    assert "tasks/roles.json: v1, unchanged" in rerun.output
    assert "    models: solver other-solver; user simulator other-user; judge openai/other-judge" in (
        rerun.output.splitlines())
    assert _ModelEndpoint.models == ["flag-judge", "other-judge"]
    assert (bundle / "tasks" / "roles.json").read_text() == written
