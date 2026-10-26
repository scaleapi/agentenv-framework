"""``agent-env run`` and the task steps on the agents bundle's OpenHands: its image built from the shipped folder with
real docker, the real OpenHands SDK, and a fake OpenAI Chat Completions endpoint as the configured model endpoint, so
no model key is needed. Needs a Docker daemon and network access for the image's downloads."""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from typing import Final

import httpx
import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import protocol
from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import _host_ips
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.util.fake_openai import USAGE, Fault, FakeOpenAI, Text, ToolCall, answer, tool_result_text, tool_then_answer

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

AGENT: Final[str] = "@local/agentenv-framework/agents/openhands"
MODEL: Final[str] = "fake-model-1"
ECHO_ENV: Final[str] = "@local/agentenv-framework/echo-mcp/echo"


@pytest.fixture
def endpoint():
    """The fake endpoint, where this machine's containers reach it: on Linux, the Docker bridge address."""
    with FakeOpenAI(host=_host_ips()[-1]) as fake:
        yield fake


@pytest.fixture
def state(monkeypatch, tmp_path, endpoint):
    """Local stores under this test's folder, and [model] pointed at the fake endpoint with a header of its own.
    Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    config = tmp_path / "config.toml"
    config.write_text(f'[model]\nbase_url = "{endpoint.url}"\napi_key = "fake-key"\ndefault = "{MODEL}"\n\n'
                      '[model.headers]\nX-Agentenv-Test = "header-check"\n')
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


def _bundle(root, name, steps):
    (root / "tasks").mkdir(parents=True)
    (root / f"tasks/{name}.json").write_text(json.dumps(steps))
    return root


def _ask(prompt, needle, env_ids=(), depends_on=()):
    return [
        {"id": "agent", "type": "deploy_agent", "a2a_agent_id": AGENT, "env_ids": list(env_ids),
         "sandbox_type": "local", **({"depends_on": list(depends_on)} if depends_on else {})},
        {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": prompt, "depends_on": ["agent"]},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "answer",
         "criteria": [{"type": "response_contains", "needles": [needle]}], "depends_on": ["ask"]},
    ]


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def _with_agent(state, steps):
    """Run ``steps`` against the shipped agent, after a bundle run has written it and its image to this test's
    stores, and tear down what they deployed."""
    assert _run(_bundle(state / "setup", "ask", _ask("Say pong.", "pong"))).exit_code == 0
    context = TaskStepContext()

    async def run():
        try:
            await steps(context)
        finally:
            await teardown_run(context)

    with namespace_routing():
        asyncio.run(run())


async def _send(base_url: str, context_id: str, text: str) -> dict:
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(f"{base_url}/a2a", json={
            "jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
                "message": {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user",
                            "contextId": context_id, "parts": [{"kind": "text", "text": text}]},
                "configuration": {"blocking": True}}})
    task = response.json()["result"]
    parts = task["status"]["message"]["parts"]
    return {"state": task["status"]["state"],
            "text": "".join(part.get("text", "") for part in parts if part["kind"] == "text"),
            "data": next((part["data"] for part in parts if part["kind"] == "data"), {})}


def test_the_shipped_openhands_answers_through_the_configured_endpoint(state, endpoint):
    endpoint.respond = answer("pong")

    result = _run(_bundle(state / "one", "ask", _ask("Say pong.", "pong")))

    assert result.exit_code == 0, result.output
    assert "tasks/ask.json v1: passed" in result.output, result.output
    calls = endpoint.completions()
    assert calls, "the agent never called the endpoint"
    assert {call.headers["authorization"] for call in calls} == {"Bearer fake-key"}
    assert {call.headers.get("x-agentenv-test") for call in calls} == {"header-check"}
    assert {call.body.get("model") for call in calls} == {MODEL}


def test_the_shipped_openhands_runs_a_shell_command_with_its_terminal_tool(state, endpoint):
    endpoint.respond = tool_then_answer("terminal", {"command": "echo $((6 * 7))", "security_risk": "LOW"},
                                        prefix="the shell said: ")

    result = _run(_bundle(state / "shell", "ask", _ask("What is six times seven? Use the shell.", "42")))

    seen = (result.output, endpoint.tool_results())
    assert result.exit_code == 0 and "tasks/ask.json v1: passed" in result.output, seen
    # The fake never computes, so 42 reached the answer through the container's shell.
    assert [result.startswith("42") for result in endpoint.tool_results()] == [True], seen


def test_the_shipped_openhands_calls_the_tools_of_the_env_it_is_deployed_with(state, endpoint):
    endpoint.respond = tool_then_answer("echo", {"text": "ping-from-tool"}, prefix="the tool said: ")
    root = _bundle(state / "tools", "echo", [
        {"id": "env", "type": "deploy_env", "env_id": ECHO_ENV, "sandbox_type": "local"},
        *_ask("Call the echo tool with the text ping-from-tool.", "ping-from-tool", env_ids=[ECHO_ENV],
              depends_on=["env"]),
    ])

    result = _run(root)

    offered = sorted({tool["function"]["name"] for call in endpoint.completions() for tool in call.body.get("tools") or []})
    seen = (result.output, offered, endpoint.tool_results())
    assert result.exit_code == 0 and "tasks/echo.json v1: passed" in result.output, seen
    assert any(name.endswith("echo") for name in offered), seen
    assert ["ping-from-tool" in result for result in endpoint.tool_results()] == [True], seen


def test_the_conversation_the_agent_keeps_condenses_and_holds_neither_the_key_nor_the_headers(state, endpoint):
    # Spelled so the command, which the conversation records, holds neither value itself.
    command = ("cd ~/.openhands-a2a/conversations && "
               "(grep -rlF -e \"$(printf 'header-%s' check)\" -e \"$(printf 'fake-%s' key)\" . || echo no-secrets) && "
               "grep -rl LLMSummarizingCondenser .")
    endpoint.respond = tool_then_answer("terminal", {"command": command, "security_risk": "LOW"}, prefix="seen: ")

    result = _run(_bundle(state / "kept", "ask", _ask("Look at your saved conversation.", "seen")))

    (seen,) = endpoint.tool_results()
    assert result.exit_code == 0 and "tasks/ask.json v1: passed" in result.output, (result.output, seen)
    assert "no-secrets" in seen and "base_state.json" in seen, seen
    # Sent all the same.
    assert {call.headers.get("x-agentenv-test") for call in endpoint.completions()} == {"header-check"}


def test_an_mcp_header_naming_a_variable_reaches_the_server_as_written(state, endpoint):
    endpoint.respond = answer("pong")

    async def steps(context):
        await DeployAgentTaskStep(id="agent", version=None, sandbox_type="local", a2a_agent_id=AGENT).execute(context)
        (agent,) = context.deployed_agents
        base_url = agent.a2a_url or agent.api_url
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"model": MODEL})
        async with httpx.AsyncClient() as client:
            registered = await client.post(f"{base_url}/ext/mcp-config", json={
                "name": "probe", "url": f"{endpoint.url}/mcp-probe", "headers": {"X-Probe": "${LITELLM_API_KEY}"}})
            registered.raise_for_status()
        await _send(base_url, "ctx", "Say pong.")

    _with_agent(state, steps)

    probes = [seen for seen in endpoint.seen if seen.path.startswith("/mcp-probe")]
    assert probes, "the agent never called the MCP server"
    assert {seen.headers.get("x-probe") for seen in probes} == {"${LITELLM_API_KEY}"}


def test_a_conversation_the_sdk_cannot_resume_fails_the_next_task_as_an_infra_error(state, endpoint):
    outcomes = {}
    endpoint.respond = answer("pong")

    async def steps(context):
        await DeployAgentTaskStep(id="agent", version=None, sandbox_type="local", a2a_agent_id=AGENT).execute(context)
        (agent,) = context.deployed_agents
        base_url = agent.a2a_url or agent.api_url
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"model": MODEL})
        outcomes["first"] = await _send(base_url, "ctx", "Say pong.")
        subprocess.run(["docker", "exec", f"agent-{agent.sandbox_id}", "sh", "-c",
                        "for f in ~/.openhands-a2a/conversations/*/base_state.json; do printf '{' > \"$f\"; done"],
                       check=True)
        outcomes["second"] = await _send(base_url, "ctx", "Say pong again.")

    _with_agent(state, steps)

    assert outcomes["first"]["state"] == "completed", outcomes
    second = outcomes["second"]
    assert second["state"] == "failed", outcomes
    assert (second["data"]["error_code"], second["data"]["error_type"]) == ("openhands.start", "infra_error"), outcomes
    assert second["text"].startswith("OpenHands could not start the turn: "), outcomes


def test_a_model_endpoint_that_refuses_the_key_fails_the_task_as_an_infra_error(state, endpoint):
    endpoint.respond = answer("pong")

    async def steps(context):
        await DeployAgentTaskStep(id="agent", version=None, sandbox_type="local", a2a_agent_id=AGENT).execute(context)
        endpoint.fail_next(Fault("status", 401), times=100)
        with pytest.raises(Exception, match="error_type=infra_error, error_code=openhands.LLMAuthenticationError"):
            await PromptAgentTaskStep(id="ask", version=None, prompt_id="ask", prompt="Say pong.").execute(context)

    _with_agent(state, steps)


def test_a_run_past_its_timeout_is_interrupted_and_its_context_keeps_working(state, endpoint):
    outcomes = {}

    def respond(body):
        # The first task's model call is the one that stalls past the timeout, and answers with a tool call that a
        # run left going would carry out and report back on, calling the endpoint again.
        if "Say pong again." in json.dumps(body["messages"][-1]):
            return Text("pong")
        return ToolCall("terminal", {"command": "echo late", "security_risk": "LOW"})

    endpoint.respond = answer("pong")

    async def steps(context):
        await DeployAgentTaskStep(id="agent", version=None, sandbox_type="local", a2a_agent_id=AGENT).execute(context)
        (agent,) = context.deployed_agents
        base_url = agent.a2a_url or agent.api_url
        endpoint.respond = respond
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"model": MODEL, "timeout_seconds": 5})
        endpoint.fail_next(Fault("stall", seconds=15))
        started = time.monotonic()
        outcomes["timed_out"] = await _send(base_url, "ctx", "Say pong.")
        outcomes["seconds"] = time.monotonic() - started
        calls = len(endpoint.completions())
        await asyncio.sleep(25)  # past the stall, so a run left going would have called the endpoint again
        outcomes["calls_after_the_timeout"] = len(endpoint.completions()) - calls
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"timeout_seconds": 120})
        outcomes["after"] = await _send(base_url, "ctx", "Say pong again.")

    _with_agent(state, steps)

    assert outcomes["timed_out"]["state"] == "failed", outcomes
    assert outcomes["timed_out"]["data"]["error_code"] == "openhands.timeout", outcomes
    assert outcomes["seconds"] < 15, outcomes
    assert outcomes["calls_after_the_timeout"] == 0, outcomes
    assert (outcomes["after"]["state"], outcomes["after"]["text"]) == ("completed", "pong"), outcomes
    assert outcomes["after"]["data"]["usage"]["input_tokens"] == USAGE["prompt_tokens"], outcomes
    # The interrupted turn stays in the conversation the next task resumes.
    last = json.dumps(endpoint.completions()[-1].body["messages"])
    assert "Say pong." in last and "Say pong again." in last, last


def test_a_run_past_its_timeout_mid_command_leaves_no_process_behind(state, endpoint):
    outcomes = {}

    def respond(body):
        result = tool_result_text(body)
        if result is not None:
            return Text(f"processes: {result}")
        if "List the processes." in json.dumps(body["messages"][-1]):
            return ToolCall("terminal", {"command": "ps -eo args", "security_risk": "LOW"})
        return ToolCall("terminal", {"command": "sleep 120", "security_risk": "LOW"})

    endpoint.respond = answer("pong")

    async def steps(context):
        await DeployAgentTaskStep(id="agent", version=None, sandbox_type="local", a2a_agent_id=AGENT).execute(context)
        (agent,) = context.deployed_agents
        base_url = agent.a2a_url or agent.api_url
        endpoint.respond = respond
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"model": MODEL, "timeout_seconds": 5})
        started = time.monotonic()
        outcomes["timed_out"] = await _send(base_url, "ctx", "Sleep.")
        outcomes["seconds"] = time.monotonic() - started
        await protocol.post_agent_config(f"{base_url}/ext/agent-config", {"timeout_seconds": 120})
        outcomes["after"] = await _send(base_url, "ctx", "List the processes.")

    _with_agent(state, steps)

    assert outcomes["timed_out"]["data"]["error_code"] == "openhands.timeout", outcomes
    assert outcomes["seconds"] < 15, outcomes
    assert outcomes["after"]["state"] == "completed", outcomes
    assert "ps -eo args" in outcomes["after"]["text"] and "sleep 120" not in outcomes["after"]["text"], outcomes
