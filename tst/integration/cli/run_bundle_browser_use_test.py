"""The agents bundle's Browser Use agent driving agentenv's chromium, deployed alone as a browser computer, through the
address ``browser_connect_cdp`` gives it: its image built from the shipped folder with real docker, real Browser Use
Pi, and a fake OpenAI Chat Completions endpoint as the configured model endpoint, which scripts its JavaScript, so no
model key is needed. Needs a Docker daemon and network access for the images' downloads."""

import asyncio
import json
import logging
import urllib.parse
import uuid
from typing import Final

import httpx
import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import protocol
from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_docker.provider import _host_ips
from agent_env.providers.sandbox_providers.sandbox_provider import reachable_url
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.browser_connect_cdp import BrowserConnectCdpTaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from tst.util.fake_openai import FakeOpenAI, Text, ToolCall

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

AGENT: Final[str] = "@local/agentenv-framework/agents/browser-use"
BROWSER: Final[str] = "@local/agentenv-framework/browsers/chromium"
MODEL: Final[str] = "fake-model-1"
# The page names itself only once it runs, so its title in an answer came from the browser, not the prompt.
PAGE: Final[str] = "data:text/html," + urllib.parse.quote("<script>document.title = 'Lantern ' + (3 + 4)</script>")
OPEN: Final[str] = (f"page = await tabs.open({json.dumps(PAGE)});\n"
                    "await page.waitFor(() => document.title.startsWith('Lantern'));\n"
                    "(await page.info()).title")
PROMPT: Final[str] = "Open the page and reply with its title."


@pytest.fixture
def endpoint():
    """The fake endpoint, where this machine's containers reach it, answering each turn as a model that runs
    ``OPEN`` and finishes with what it printed."""
    with FakeOpenAI(host=_host_ips()[-1]) as fake:
        fake.respond = _browses
        yield fake


@pytest.fixture
def state(monkeypatch, tmp_path, endpoint):
    """Local stores under this test's folder, and [model] pointed at the fake endpoint. Whatever ran, no sandbox work
    folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    config = tmp_path / "config.toml"
    config.write_text(f'[model]\nbase_url = "{endpoint.url}"\napi_key = "fake-key"\ndefault = "{MODEL}"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    for name in ("LITELLM_API_KEY", "LITELLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def test_the_smoke_tasks_shape_passes_with_the_agent_on_the_browsers_ws_url(state, endpoint):
    root = state / "smoke"
    (root / "tasks").mkdir(parents=True)
    (root / "tasks/browse.json").write_text(json.dumps([
        {"id": "browser", "type": "deploy_env", "env_id": "chromium", "sandbox_type": "local"},
        {"id": "deploy", "type": "deploy_agent", "a2a_agent_id": "browser-use", "env_ids": [],
         "sandbox_type": "local"},
        {"id": "connect", "type": "browser_connect_cdp", "env_id": "chromium", "depends_on": ["browser", "deploy"]},
        {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": PROMPT, "depends_on": ["connect"]},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "answer",
         "criteria": [{"type": "response_contains", "needles": ["Lantern 7"]}], "depends_on": ["ask"]},
    ]))

    result = CliRunner().invoke(cli, ["run", str(root)])

    seen = (result.output, [_tool_results(call.body) for call in endpoint.completions()])
    assert result.exit_code == 0 and "tasks/browse.json v1: passed" in result.output, seen


def test_it_connects_by_http_url_too_refuses_a_wrong_credential_and_without_an_address_fails_closed(
        state, endpoint):
    outcomes = {}

    async def steps(context):
        await DeployEnvTaskStep(id="browser", version=None, env_id=BROWSER, sandbox_type="local").execute(context)
        await DeployAgentTaskStep(id="deploy", version=None, env_ids=[], a2a_agent_id=AGENT,
                                  sandbox_type="local").execute(context)
        (env,), (agent,) = context.deployed_envs, context.deployed_agents
        base_url = agent.a2a_url
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"model": MODEL})
        outcomes["none"] = await _send(base_url, PROMPT)
        outcomes["none_asked"] = len(endpoint.completions())

        await BrowserConnectCdpTaskStep(id="connect", version=None, env_id=BROWSER).execute(context)
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"connect_by": "http"})
        outcomes["http"] = await _send(base_url, PROMPT)

        forged = reachable_url(env.environment_url.rstrip("/"), from_sandbox_type=env.sandbox_type,
                               to_sandbox_type=agent.sandbox_type).rstrip("/") + f"/svc/chromium/cdp/{uuid.uuid4().hex}"
        async with httpx.AsyncClient(timeout=60) as client:
            added = await client.post(f"{base_url}/ext/cdp-config", json={
                "name": "forged", "http_url": f"{forged}/", "ws_url": "ws" + forged.removeprefix("http")
                + "/devtools/browser"})
        added.raise_for_status()
        for connect_by in ("ws", "http"):
            await protocol.post_agent_config(f"{base_url}/ext/agent-config",
                                             {"browser": "forged", "connect_by": connect_by})
            outcomes[f"forged_{connect_by}"] = await _send(base_url, PROMPT)

    written = CliRunner().invoke(cli, ["run", str(_writes_the_agent_and_browser(state))])
    assert written.exit_code == 0, written.output
    context = TaskStepContext()

    async def run():
        try:
            await steps(context)
        finally:
            await teardown_run(context)

    with namespace_routing():
        asyncio.run(run())

    seen = (outcomes, [_tool_results(call.body) for call in endpoint.completions()])
    assert (outcomes["none"]["state"], outcomes["none"]["data"].get("error_code")) == ("failed", "cdp.no_endpoint"), seen
    assert outcomes["none_asked"] == 0, seen
    assert outcomes["http"]["state"] == "completed" and "Lantern 7" in outcomes["http"]["text"], seen
    # The relay refuses the forged token's WebSocket upgrade, and its discovery with a 403.
    assert "Could not connect to CDP endpoint" in outcomes["forged_ws"]["text"], seen
    assert "CDP discovery failed (403)" in outcomes["forged_http"]["text"], seen


def _browses(body: dict) -> Text | ToolCall:
    """Run ``OPEN``, then finish with what it printed, or the error it raised."""
    results = _tool_results(body)
    if not results:
        return ToolCall("javascript", {"code": OPEN})
    if results[-1] == "Result accepted.":
        return Text("done")
    return ToolCall("finish", {"result": results[-1]})


def _tool_results(body: dict) -> list[str]:
    """The text of each tool result in the request's messages, in order."""
    texts = []
    for message in body.get("messages") or []:
        if message.get("role") == "tool":
            content = message.get("content")
            texts.append("".join(part.get("text", "") for part in content if isinstance(part, dict))
                         if isinstance(content, list) else content or "")
    return texts


def _writes_the_agent_and_browser(state):
    """A bundle whose one task deploys the agent and the browser, so a run of it writes both to this test's stores."""
    root = state / "setup"
    (root / "tasks").mkdir(parents=True)
    (root / "tasks/setup.json").write_text(json.dumps([
        {"id": "browser", "type": "deploy_env", "env_id": "chromium", "sandbox_type": "local"},
        {"id": "deploy", "type": "deploy_agent", "a2a_agent_id": "browser-use", "env_ids": [],
         "sandbox_type": "local"},
    ]))
    return root


async def _send(base_url: str, text: str) -> dict:
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{base_url}/a2a", json={
            "jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
                "message": {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user",
                            "contextId": uuid.uuid4().hex, "parts": [{"kind": "text", "text": text}]},
                "configuration": {"blocking": True}}})
    task = response.json()["result"]
    parts = task["status"]["message"]["parts"]
    return {"state": task["status"]["state"],
            "text": "".join(part.get("text", "") for part in parts if part["kind"] == "text"),
            "data": next((part["data"] for part in parts if part["kind"] == "data"), {})}
