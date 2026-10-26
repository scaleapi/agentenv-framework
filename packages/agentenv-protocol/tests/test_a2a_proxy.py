from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from typing import Any

import pytest
import uvicorn
from starlette.testclient import TestClient

from agentenv_protocol import EnvironmentCard, ToolContext, ToolSession, create_fastmcp_app, tool
from agentenv_protocol.a2a_agent import ProxyActionState, proxy
from agentenv_protocol.a2a_agent.proxy import ProxyAgent

TOKEN = "t" * 32


class _Items:
    def __init__(self) -> None:
        self.items: list[str] = []

    @tool()
    async def add_item(self, item: str) -> str:
        """Add an item."""
        self.items.append(item)
        return json.dumps({"count": len(self.items)})

    @tool()
    def explode(self) -> str:
        """Always fails."""
        raise RuntimeError("the tool blew up")

    @tool()
    def count(self, n: int) -> str:
        """Echo a number."""
        return str(n)

    @tool()
    def whoami(self, ctx: ToolContext) -> str:
        """Who is calling."""
        return json.dumps({"role": ctx.caller.role})


class _Terminal:
    @tool()
    def bash(self, command: str, cwd: str = "/") -> str:
        """Run a command in the env's own terminal."""
        return f"{cwd}$ {command}"


@pytest.fixture(scope="module")
def env():
    """An items env's MCP server on loopback, as a task's env serves one."""
    yield from _served(_Items(), "items")


@pytest.fixture(scope="module")
def terminal():
    """An env whose own tool is named ``bash``."""
    yield from _served(_Terminal(), "terminal")


def _served(handler: Any, name: str):
    app = create_fastmcp_app(handler, card=EnvironmentCard(name=name)).streamable_http_app()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield handler, f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
    server.should_exit = True
    thread.join(5)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.chdir(tmp_path)
    with TestClient(ProxyAgent(long_poll_seconds=0.2).create_app()) as client:
        assert client.post("/ext/proxy/claim", json={"token": TOKEN}).status_code == 200
        yield client


def _send(client: TestClient, text: str, *, context_id: str = "context-1", parts: list | None = None) -> str:
    response = client.post("/a2a", json={
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {
            "message": {"kind": "message", "messageId": f"message-{text}", "role": "user", "contextId": context_id,
                        "parts": parts or [{"kind": "text", "text": text}]},
            "configuration": {"blocking": False},
        },
    })
    return response.json()["result"]["id"]


def _task(client: TestClient, task_id: str) -> dict[str, Any]:
    return client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "tasks/get",
                                     "params": {"id": task_id}}).json()["result"]


def _settled(client: TestClient, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while (task := _task(client, task_id))["status"]["state"] not in ("completed", "failed", "canceled"):
        assert time.monotonic() < deadline, task
        time.sleep(0.02)
    return task


def _proxy(client: TestClient, operation: str, **body: Any):
    return client.post(f"/ext/proxy/{operation}", json={"token": TOKEN, **body})


def _turns(client: TestClient, after: int = 0) -> dict[str, Any]:
    response = _proxy(client, "next", after=after)
    assert response.status_code == 200, response.text
    return response.json()


def _done(client: TestClient, task_id: str, action_id: str, action: dict[str, Any]) -> dict[str, Any]:
    response = _proxy(client, "act", task_id=task_id, action_id=action_id, action=action)
    assert response.status_code == 200, response.text
    state = response.json()
    while not state["done"]:
        state = _proxy(client, "result", action_id=action_id).json()
    return state["result"]


def _bash(command: str, **options: Any) -> dict[str, Any]:
    return {"kind": "bash", "command": command, **options}


def _register(client: TestClient, url: str, name: str = "items", role: str | None = None) -> None:
    headers = {"AgentEnv-Role": role} if role else None
    assert client.post("/ext/mcp-config", json={"url": url, "name": name, "headers": headers}).status_code == 200


def _trajectory(client: TestClient, task_id: str) -> list[dict[str, Any]]:
    return client.post("/ext/trajectory", json={"task_id": task_id}).json()["trajectory"]


def _attrs(span: dict[str, Any], key: str) -> Any:
    return json.loads(span["attributes"][key])


def _stopped(pid: int) -> None:
    """Until ``pid`` is gone, or a zombie: a killed child whose parent has gone stays one until what adopted it reaps it."""
    deadline = time.monotonic() + 10
    while (state := _state(pid)) and not state.startswith("Z"):
        assert time.monotonic() < deadline, state
        time.sleep(0.02)


def _state(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def test_the_card_offers_the_proxy_env_servers_and_a_system_prompt():
    card = ProxyAgent().create_app().state.agentenv_a2a.card.model_dump(mode="json")
    extensions = {extension["uri"]: extension for extension in card["capabilities"]["extensions"]}

    assert sorted(extensions["urn:agentenv:proxy/v1"]["params"]["methods"]) == [
        "act", "claim", "next", "record", "reply", "result", "tools"]
    assert "urn:agentenv:mcp-config/v1" in extensions
    assert "system_prompt" in extensions["urn:agentenv:agent-config/v1"]["params"]["methods"]["set"]["request"][
        "supported"]


def test_the_first_claim_sets_the_token_that_every_driver_call_needs():
    with TestClient(ProxyAgent(long_poll_seconds=0.2).create_app()) as client:
        assert _proxy(client, "next").status_code == 401
        assert client.post("/ext/proxy/claim", json={"token": "short"}).status_code == 400
        assert client.post("/ext/proxy/claim", json={"token": "é" * 22}).status_code == 200
        assert client.post("/ext/proxy/claim", json={"token": "é" * 22}).status_code == 200
        assert client.post("/ext/proxy/claim", json={"token": TOKEN}).status_code == 409
        assert _proxy(client, "next").status_code == 401
        assert client.post("/ext/proxy/next", json={"token": "é" * 22}).status_code == 200


def test_a_message_is_an_open_turn_until_the_driver_replies(client):
    client.post("/ext/agent-config", json={"system_prompt": "be brief"})
    task_id = _send(client, "fix the build")

    seen = _turns(client)
    assert seen["turns"] == [{"task_id": task_id, "context_id": "context-1",
                              "parts": [{"kind": "text", "text": "fix the build"}], "system_prompt": "be brief",
                              "skills": []}]
    assert _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "fixed"}]).status_code == 200

    task = _settled(client, task_id)
    assert task["status"]["state"] == "completed"
    assert task["status"]["message"]["parts"][0] == {"kind": "text", "text": "fixed"}
    [turn] = _trajectory(client, task_id)
    assert (turn["name"], _attrs(turn, "gen_ai.prompt"), _attrs(turn, "gen_ai.completion")) == (
        "turn", {"prompt": "fix the build", "system_prompt": "be brief"},
        {"content": [{"type": "text", "text": "fixed"}]})
    assert _turns(client, after=seen["version"])["turns"] == []


def test_a_message_with_a_part_that_isnt_text_fails_before_the_driver_sees_it(client):
    task_id = _send(client, "look", parts=[{"kind": "text", "text": "look"}, {"kind": "data", "data": {"a": 1}}])

    task = _settled(client, task_id)
    assert task["status"]["state"] == "failed"
    assert "text parts only" in json.dumps(task["status"]["message"])
    assert _turns(client)["turns"] == []


def test_an_error_reply_fails_the_turn(client):
    task_id = _send(client, "do it")

    _proxy(client, "reply", task_id=task_id, error="the driver gave up")

    task = _settled(client, task_id)
    assert task["status"]["state"] == "failed"
    assert "the driver gave up" in task["status"]["message"]["parts"][0]["text"]
    assert _trajectory(client, task_id)[-1]["status"] == {"status_code": "ERROR"}


def test_a_reply_names_exactly_one_open_turn_and_carries_parts_or_an_error(client):
    task_id = _send(client, "do it")

    assert _proxy(client, "reply", task_id=task_id).status_code == 400
    assert _proxy(client, "reply", task_id=task_id, parts=[]).status_code == 400
    assert _proxy(client, "reply", task_id="nope", parts=[{"kind": "text", "text": "done"}]).status_code == 404
    _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "done"}])
    _settled(client, task_id)
    assert _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "again"}]).status_code == 404


def test_a_turn_its_sender_cancels_is_withdrawn(client):
    task_id = _send(client, "do it")
    assert [turn["task_id"] for turn in _turns(client)["turns"]] == [task_id]

    client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "tasks/cancel", "params": {"id": task_id}})

    assert _settled(client, task_id)["status"]["state"] == "canceled"
    assert _turns(client)["turns"] == []


def test_next_waits_for_a_change_after_the_version_it_was_given(client):
    version = _turns(client)["version"]

    started = time.monotonic()
    unchanged = _turns(client, after=version)

    assert unchanged["version"] == version
    assert time.monotonic() - started >= 0.2


def test_a_turn_offers_its_env_servers_tools_and_bash(client, env):
    _register(client, env[1])
    task_id = _send(client, "go")

    tools = {item["name"]: item for item in _proxy(client, "tools", task_id=task_id).json()["tools"]}

    assert set(tools) == {"add_item", "count", "explode", "whoami", "bash"}
    assert (tools["add_item"]["server"], "server" in tools["bash"]) == ("items", False)
    assert tools["bash"] == {
        "name": "bash",
        "description": "Run a bash command where the agent works, as its user, in its folder; returns its stdout, "
                       "then its stderr.",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string", "description": "The command."}},
                         "required": ["command"]},
    }
    assert tools["add_item"]["input_schema"]["properties"]["item"]["type"] == "string"


def test_two_env_servers_offering_one_tool_name_are_refused(client, env):
    _register(client, env[1], name="items")
    _register(client, env[1].replace("127.0.0.1", "localhost"), name="pantry")
    task_id = _send(client, "go")

    response = _proxy(client, "tools", task_id=task_id)

    assert response.status_code == 409
    assert "offered by both 'items' and 'pantry'" in response.text


def test_a_tool_action_calls_the_env_with_the_headers_registered_for_the_proxy(client, env):
    handler, url = env
    _register(client, url, role="ops")
    task_id = _send(client, "go")

    added = _done(client, task_id, "a1", {"kind": "tool", "server": "items", "name": "add_item",
                                          "arguments": {"item": "crate"}})
    who = _done(client, task_id, "a2", {"kind": "tool", "server": "items", "name": "whoami"})
    failed = _done(client, task_id, "a3", {"kind": "tool", "server": "items", "name": "explode"})

    assert (added["kind"], added["is_error"], "crate" in handler.items) == ("tool", False, True)
    assert json.loads(who["content"][0]["text"]) == {"role": "ops"}
    assert failed["is_error"] is True


def test_a_tool_actions_deadline_covers_connecting_to_its_server(client):
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        _register(client, f"http://127.0.0.1:{silent.getsockname()[1]}/mcp")
        task_id = _send(client, "go")

        started = time.monotonic()
        result = _done(client, task_id, "t", {"kind": "tool", "server": "items", "name": "add_item",
                                              "timeout_seconds": 0.5})

    assert result["is_error"] and "timed out after 0.5s" in result["content"][0]["text"]
    assert time.monotonic() - started < 5


def test_a_server_that_never_lists_its_tools_is_a_bad_gateway(client, monkeypatch):
    monkeypatch.setattr(proxy, "TOOL_TIMEOUT_SECONDS", 0.5)
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        _register(client, f"http://127.0.0.1:{silent.getsockname()[1]}/mcp")
        task_id = _send(client, "go")

        response = _proxy(client, "tools", task_id=task_id)

    assert response.status_code == 502
    assert "didn't list its tools within 0.5s" in response.text


def test_a_turns_servers_list_their_tools_within_one_deadline_for_them_all(client, monkeypatch):
    monkeypatch.setattr(proxy, "TOOL_TIMEOUT_SECONDS", 1.0)

    async def slowly(self, turn, server):
        await asyncio.sleep(0.6)
        return []

    monkeypatch.setattr(proxy.ProxyAgent, "_listed", slowly)
    _register(client, "http://127.0.0.1:9/mcp", name="items")
    _register(client, "http://localhost:9/mcp", name="pantry")
    task_id = _send(client, "go")

    response = _proxy(client, "tools", task_id=task_id)

    assert response.status_code == 502
    assert "'pantry' didn't list its tools within 1.0s" in response.text


def test_a_turns_actions_share_one_connection_to_each_env_server_which_closes_with_the_turn(client, env, monkeypatch):
    opened, closed, results = [], [], []
    open_session, close_session = ToolSession.open, ToolSession.close

    async def opening(self):
        opened.append(self)
        await asyncio.sleep(0.2)
        return await open_session(self)

    async def closing(self):
        if self._client is not None:
            closed.append(self)
        await close_session(self)

    monkeypatch.setattr(ToolSession, "open", opening)
    monkeypatch.setattr(ToolSession, "close", closing)
    _register(client, env[1])
    task_id = _send(client, "go")

    calls = [threading.Thread(target=lambda n=n: results.append(_done(
        client, task_id, f"w{n}", {"kind": "tool", "server": "items", "name": "whoami"}))) for n in range(3)]
    for call in calls:
        call.start()
    for call in calls:
        call.join()
    _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "done"}])
    _settled(client, task_id)

    assert [result["is_error"] for result in results] == [False] * 3
    assert len(opened) == 1 and closed == opened


def test_a_bash_action_runs_in_the_proxys_folder_and_returns_a_structured_result(client, tmp_path):
    (tmp_path / "inputs").mkdir()
    task_id = _send(client, "go")

    result = _done(client, task_id, "b1", _bash("ls; echo $0; echo out; echo err >&2; exit 3"))

    assert result == {
        "kind": "bash", "exit_code": 3, "stdout": "inputs\nbash\nout\n", "stderr": "err\n", "timed_out": False}


def test_where_there_is_no_bash_the_bash_tool_runs_sh_and_says_so(monkeypatch, tmp_path):
    which = shutil.which
    monkeypatch.setattr(proxy.shutil, "which", lambda name, *args, **kwargs: None if name == "bash" else which(name, *args, **kwargs))
    monkeypatch.chdir(tmp_path)
    with TestClient(ProxyAgent(long_poll_seconds=0.2).create_app()) as client:
        assert client.post("/ext/proxy/claim", json={"token": TOKEN}).status_code == 200
        task_id = _send(client, "go")

        tools = {item["name"]: item for item in _proxy(client, "tools", task_id=task_id).json()["tools"]}
        result = _done(client, task_id, "s1", _bash("echo $0"))

    assert "with POSIX sh: this place has no bash" in tools["bash"]["description"]
    assert (result["exit_code"], result["stdout"]) == (0, "sh\n")


def test_an_action_done_within_the_long_poll_answers_at_once_and_a_slow_one_is_polled(client):
    task_id = _send(client, "go")

    quick = _proxy(client, "act", task_id=task_id, action_id="q", action=_bash("echo hi")).json()
    slow = _proxy(client, "act", task_id=task_id, action_id="s", action=_bash("sleep 0.5; echo late")).json()

    assert (quick["done"], quick["result"]["stdout"]) == (True, "hi\n")
    assert slow == {"done": False}
    assert _done(client, task_id, "s", _bash("sleep 0.5; echo late"))["stdout"] == "late\n"


def test_an_action_retried_with_its_id_runs_once(client, tmp_path):
    task_id = _send(client, "go")

    first = _done(client, task_id, "once", _bash("echo x >> ran; wc -l < ran"))
    again = _done(client, task_id, "once", _bash("echo x >> ran; wc -l < ran"))

    assert first == again
    assert (tmp_path / "ran").read_text() == "x\n"


def test_a_command_that_outlives_its_timeout_is_stopped_with_what_it_started(client, tmp_path):
    task_id = _send(client, "go")

    waited = _done(client, task_id, "w", _bash("sleep 60 & echo $! > waited; wait", timeout_seconds=0.5))

    assert waited == {"kind": "bash", "stdout": "", "stderr": "", "timed_out": True}
    assert ProxyActionState.model_validate({"done": True, "result": waited}).result.exit_code is None
    _stopped(int((tmp_path / "waited").read_text()))


def test_a_command_is_done_when_its_shell_exits_and_what_it_left_in_the_background_runs_on(client, tmp_path):
    task_id = _send(client, "go")

    started = time.monotonic()
    left = _done(client, task_id, "l", _bash("sleep 60 & echo $! > left; echo started"))

    assert left == {"kind": "bash", "exit_code": 0, "stdout": "started\n", "stderr": "", "timed_out": False}
    assert time.monotonic() - started < proxy.OUTPUT_GRACE_SECONDS + 2
    pid = int((tmp_path / "left").read_text())
    assert _state(pid) and not _state(pid).startswith("Z")
    os.kill(pid, signal.SIGKILL)


def test_what_a_background_process_writes_after_its_command_is_done_never_blocks_it(client, monkeypatch, tmp_path):
    monkeypatch.setattr(proxy, "OUTPUT_GRACE_SECONDS", 0.1)
    task_id = _send(client, "go")
    writes = "import sys, time; time.sleep(0.5); sys.stdout.write('x' * 200_000); sys.stdout.flush(); open('done', 'w')"

    result = _done(client, task_id, "b", _bash(f"python3 -c \"{writes}\" & echo started"))

    assert (result["exit_code"], result["stdout"]) == (0, "started\n")
    deadline = time.monotonic() + 10
    while not (tmp_path / "done").exists():
        assert time.monotonic() < deadline, "the background process blocked on its output"
        time.sleep(0.05)


def test_a_command_too_long_to_start_fails_as_a_shell_reports_it(client):
    task_id = _send(client, "go")

    result = _done(client, task_id, "huge", _bash("true " + "x" * 3_000_000))

    assert (result["exit_code"], result["timed_out"], result["stdout"]) == (126, False, "")
    assert "couldn't start" in result["stderr"]
    _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "done"}])
    _settled(client, task_id)
    [action, _] = _trajectory(client, task_id)
    assert (action["name"], action["status"]) == ("bash", {"status_code": "ERROR"})


def test_an_action_polled_while_its_turn_closes_is_answered_404(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    agent = ProxyAgent(long_poll_seconds=10)
    with TestClient(agent.create_app()) as client:
        client.post("/ext/proxy/claim", json={"token": TOKEN})
        task_id = _send(client, "go")
        polled = {}
        poll = threading.Thread(target=lambda: polled.update(response=_proxy(
            client, "act", task_id=task_id, action_id="long", action=_bash("sleep 30"))))
        poll.start()
        while "long" not in agent._actions:
            time.sleep(0.01)

        _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "done"}])
        poll.join(10)

    assert polled["response"].status_code == 404
    assert "'long' was stopped: its turn is over" in polled["response"].text


def test_a_turn_that_closes_stops_its_running_actions_and_forgets_them(client, tmp_path):
    task_id = _send(client, "go")
    assert _proxy(client, "act", task_id=task_id, action_id="long",
                  action=_bash("sleep 60 & echo $! > child; wait")).json()["done"] is False

    _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "done"}])
    _settled(client, task_id)

    _stopped(int((tmp_path / "child").read_text()))
    assert _proxy(client, "result", action_id="long").status_code == 404
    assert _proxy(client, "act", task_id=task_id, action_id="new", action=_bash("true")).status_code == 404


def test_a_turn_that_closes_while_a_done_commands_output_is_awaited_stops_what_it_left(client, monkeypatch, tmp_path):
    monkeypatch.setattr(proxy, "OUTPUT_GRACE_SECONDS", 30)
    task_id = _send(client, "go")
    assert _proxy(client, "act", task_id=task_id, action_id="left",
                  action=_bash("sleep 60 & echo $! > child")).json()["done"] is False

    _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "done"}])
    _settled(client, task_id)

    _stopped(int((tmp_path / "child").read_text()))
    assert _proxy(client, "result", action_id="left").status_code == 404


def test_a_proxy_without_a_shell_offers_no_bash_and_keeps_no_skills():
    with TestClient(ProxyAgent(shell=False, long_poll_seconds=0.2).create_app()) as client:
        client.post("/ext/proxy/claim", json={"token": TOKEN})
        task_id = _send(client, "go")

        assert _proxy(client, "tools", task_id=task_id).json()["tools"] == []
        assert _proxy(client, "act", task_id=task_id, action_id="b", action=_bash("true")).status_code == 409
        assert client.post("/ext/skill-config", json={"name": "x", "description": "d", "skill_md": "#"}).status_code \
            == 409


def test_recorded_model_calls_are_checked_against_the_actions_that_run(client, env):
    _register(client, env[1])
    task_id = _send(client, "go")
    message = {"role": "assistant", "content": "Adding it.", "reasoning_content": "The user wants a crate.",
               "tool_calls": [
                   {"id": "c1", "type": "function", "function": {"name": "add_item", "arguments": '{"item": "crate"}'}},
                   {"id": "c2", "type": "function", "function": {"name": "bash",
                                                                 "arguments": '{"command": "true", "timeout": 30}'}},
                   {"id": "c3", "type": "function", "function": {"name": "my_own_tool", "arguments": "{}"}},
                   {"id": "c4", "type": "function", "function": {"name": "count", "arguments": '{"n": 2.0}'}}]}

    assert _proxy(client, "record", task_id=task_id, message=message, model="qwen",
                  usage={"prompt_tokens": 12, "completion_tokens": 7}).status_code == 200
    unmade = _proxy(client, "act", task_id=task_id, action_id="x",
                    action={"kind": "tool", "server": "items", "name": "add_item", "arguments": {"item": "box"}})
    assert unmade.status_code == 409
    assert _proxy(client, "record", task_id=task_id, message="Done.").status_code == 409
    assert _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "x"}]).status_code == 409
    _done(client, task_id, "b", _bash("true"))
    _done(client, task_id, "a", {"kind": "tool", "server": "items", "name": "add_item", "arguments": {"item": "crate"}})
    _done(client, task_id, "n", {"kind": "tool", "server": "items", "name": "count", "arguments": {"n": 2}})
    assert _proxy(client, "record", task_id=task_id, message="Done.").status_code == 200
    _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "Done."}])

    task = _settled(client, task_id)
    spans = _trajectory(client, task_id)
    assert [(span["name"], span["attributes"]["gen_ai.operation.name"]) for span in spans] == [
        ("chat", "chat"), ("bash", "execute_tool"), ("add_item", "execute_tool"), ("count", "execute_tool"),
        ("chat", "chat"), ("turn", "chain")]
    assert {span["status"]["status_code"] for span in spans} == {"OK"}
    assert _attrs(spans[0], "gen_ai.completion")["content"] == [
        {"type": "thinking", "thinking": "The user wants a crate."}, {"type": "text", "text": "Adding it."},
        {"type": "tool_use", "id": "c1", "name": "add_item", "input": {"item": "crate"}},
        {"type": "tool_use", "id": "c2", "name": "bash", "input": {"command": "true", "timeout": 30}},
        {"type": "tool_use", "id": "c3", "name": "my_own_tool", "input": {}},
        {"type": "tool_use", "id": "c4", "name": "count", "input": {"n": 2.0}}]
    assert (spans[0]["attributes"]["gen_ai.request.model"], spans[0]["attributes"]["gen_ai.usage.input_tokens"],
            spans[0]["attributes"]["gen_ai.usage.output_tokens"]) == ("qwen", 12, 7)
    assert _attrs(spans[2], "gen_ai.prompt") == {"input": {"item": "crate"}}
    assert task["status"]["message"]["parts"][1]["data"]["usage"]["tool_call_count"] == 3


def test_an_env_tool_named_bash_is_held_to_all_its_recorded_arguments(terminal, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with TestClient(ProxyAgent(shell=False, long_poll_seconds=0.2).create_app()) as client:
        client.post("/ext/proxy/claim", json={"token": TOKEN})
        _register(client, terminal[1], name="terminal")
        task_id = _send(client, "go")
        message = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "bash", "arguments": '{"command": "ls", "cwd": "/a"}'}}]}

        assert _proxy(client, "record", task_id=task_id, message=message).status_code == 200
        moved = _proxy(client, "act", task_id=task_id, action_id="b", action={
            "kind": "tool", "server": "terminal", "name": "bash", "arguments": {"command": "ls", "cwd": "/b"}})
        ran = _done(client, task_id, "a", {"kind": "tool", "server": "terminal", "name": "bash",
                                          "arguments": {"command": "ls", "cwd": "/a"}})

    assert moved.status_code == 409
    assert ran["content"][0]["text"] == "/a$ ls"


def test_two_model_calls_recorded_at_once_are_held_one_after_the_other(client, monkeypatch):
    listing, statuses = ProxyAgent._tools, []

    async def slow(self, turn):
        await asyncio.sleep(0.3)
        return await listing(self, turn)

    def record(command: str) -> None:
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": command, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]}
        statuses.append(_proxy(client, "record", task_id=task_id, message=message).status_code)

    monkeypatch.setattr(ProxyAgent, "_tools", slow)
    task_id = _send(client, "go")
    records = [threading.Thread(target=record, args=(command,)) for command in ("true", "false")]
    for thread in records:
        thread.start()
    for thread in records:
        thread.join()

    assert sorted(statuses) == [200, 409]


@pytest.mark.parametrize(("message", "content"), [
    ("Plain words.", [{"type": "text", "text": "Plain words."}]),
    ({"role": "assistant", "content": [{"type": "thinking", "thinking": "Hmm."}, {"type": "text", "text": "Hi."},
                                       {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]},
     [{"type": "thinking", "thinking": "Hmm."}, {"type": "text", "text": "Hi."},
      {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]),
    ({"role": "assistant", "content": [{"type": "text", "text": "Parts."}]}, [{"type": "text", "text": "Parts."}]),
])
def test_a_model_call_is_recorded_from_an_openai_or_anthropic_message_or_a_string(client, message, content):
    task_id = _send(client, "go")

    assert _proxy(client, "record", task_id=task_id, message=message).status_code == 200

    tools = [block for block in content if block["type"] == "tool_use"]
    if tools:
        _done(client, task_id, "b", _bash("ls"))
    _proxy(client, "reply", task_id=task_id, parts=[{"kind": "text", "text": "ok"}])
    _settled(client, task_id)
    assert _attrs(_trajectory(client, task_id)[0], "gen_ai.completion") == {"content": content}


@pytest.mark.parametrize("message", [
    {"role": "user", "content": "Not the model's."},
    {"role": "assistant", "content": {"text": "a dict"}},
    {"role": "assistant", "content": [{"type": "image", "source": {}}]},
    {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "bash", "arguments": "{broken"}}]},
])
def test_a_model_call_in_any_other_shape_is_refused(client, message):
    task_id = _send(client, "go")

    assert _proxy(client, "record", task_id=task_id, message=message).status_code == 400


def test_a_skill_is_written_where_its_turns_say_it_is_and_only_once(client, tmp_path):
    skill = {"name": "deploy", "description": "How to deploy.", "skill_md": "# Deploy\n"}
    assert client.post("/ext/skill-config", json=skill).status_code == 200
    assert client.post("/ext/skill-config", json=skill).status_code == 409
    _send(client, "go")

    assert _turns(client)["turns"][0]["skills"] == [
        {"name": "deploy", "description": "How to deploy.", "path": str(tmp_path / "skills" / "deploy")}]
    assert (tmp_path / "skills" / "deploy" / "SKILL.md").read_text() == "# Deploy\n"


def test_a_skill_whose_files_fail_to_download_leaves_no_folder(client, monkeypatch, tmp_path):
    async def download(source, target):
        if target.name == "notes.md":
            raise OSError("the grant expired")
        target.write_text("# Review\n")

    monkeypatch.setattr(proxy, "download", download)
    read = {"media_type": "text/markdown", "max_bytes": 1024,
            "read": {"kind": "http-get", "url": "https://objects.example.test/r", "expires_at": "2099-01-01T00:00:00Z"}}
    bundle = {"name": "review", "description": "Review code", "skill_bundle": {"max_total_bytes": 2048, "files": [
        {"path": "SKILL.md", "object": read}, {"path": "notes.md", "object": read}]}}

    assert client.post("/ext/skill-config", json=bundle).status_code >= 400
    assert not (tmp_path / "skills" / "review").exists()


def test_what_a_turn_still_runs_is_stopped_when_the_app_stops(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with TestClient(ProxyAgent(long_poll_seconds=0.2).create_app()) as client:
        client.post("/ext/proxy/claim", json={"token": TOKEN})
        task_id = _send(client, "go")
        _proxy(client, "act", task_id=task_id, action_id="long", action=_bash("sleep 60 & echo $! > child; wait"))
        while not (tmp_path / "child").exists():
            time.sleep(0.02)

    _stopped(int((tmp_path / "child").read_text()))
