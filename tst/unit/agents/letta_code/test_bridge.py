"""Exercise state restore, system instructions, event parsing and cancellation without model calls."""
import asyncio
import json
import shutil
import signal
import subprocess
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tst.unit.agents.letta_code.agent import AGENT_DIR, a2a_server, letta_code_bridge as bridge, model_relay


@pytest.mark.asyncio
async def test_snapshot_restores_memory_and_conversation_in_another_context(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    source = bridge.context_data_dir("source")
    source.mkdir(parents=True)
    (source / "context.json").write_text(json.dumps({"home": str(source)}))
    (source / "agent-id.json").write_text('{"id":"agent-local-1"}')
    memory = source / ".letta/lc-local-backend/memfs/agent-local-1/memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("Remember violet otters")
    conversation = source / "conversation.json"
    conversation.write_text(json.dumps({"id": "local-conv-7", "memory": str(memory), "messages": ["secret"]}))
    snapshot = tmp_path / "state.tgz"
    snapshot.write_bytes(await bridge.export_session("source", "local-conv-7"))
    target = bridge.context_data_dir("target")
    target.mkdir()
    (target / "stale").write_text("old target")
    assert await bridge.import_session("target", str(snapshot)) == "local-conv-7"
    assert not (target / "stale").exists()
    assert (target / memory.relative_to(source) / "MEMORY.md").read_text() == "Remember violet otters"
    restored = json.loads((target / "conversation.json").read_text())
    assert restored["messages"] == ["secret"]
    assert restored["memory"] == str(target / memory.relative_to(source))
    assert (source / "conversation.json").exists()


def test_system_prompt_is_committed_and_servers_keep_role_headers(tmp_path):
    memory = tmp_path / ".letta/lc-local-backend/memfs/agent-local-1/memory"
    memory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(memory)], check=True)
    (memory / "MEMORY.md").write_text("index")
    bridge._configure_agent(tmp_path, "agent-local-1", {
        "mail": {"url": "https://mail.example/mcp", "headers": {"Authorization": "token", "AgentEnv-Role": "old"}},
    }, "reader", "End every answer with ZEBRA")
    committed = subprocess.check_output(["git", "show", "HEAD:task_instructions.md"], cwd=memory).decode()
    assert "End every answer with ZEBRA" in committed
    assert "mcp__<server>__<tool>" in committed
    settings = json.loads((tmp_path / ".letta/settings.json").read_text())
    server = settings["agents"][0]["mcpServers"][0]
    assert server["headers"] == {"Authorization": "token", "AgentEnv-Role": "reader"}
    bridge._configure_agent(tmp_path, "agent-local-1", {}, None, "Use TIGER now")
    committed = subprocess.check_output(["git", "show", "HEAD:task_instructions.md"], cwd=memory).decode()
    assert "TIGER" in committed and "ZEBRA" not in committed


def test_native_events_track_conversation_tools_and_stderr():
    events = []
    parser = bridge._EventParser(events.append)
    for event in [
        {"type": "system", "conversation_id": "local-conv-2"},
        {"type": "message", "message_type": "tool_call_message", "tool_call": {"tool_call_id": "call-1"}},
        {"type": "message", "message_type": "tool_call_message", "tool_call": {"tool_call_id": "call-1"}},
        {"type": "result", "subtype": "success", "result": "42"},
    ]:
        parser.on_line(json.dumps(event))
    result = parser.result(b"", 0)
    assert result.response == "42" and result.session_id == "local-conv-2"
    assert result.tool_call_count == 1 and result.error is None
    assert len(events) == 4
    assert "provider failed" in bridge._EventParser().result(b"provider failed", 1).error


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "local-conv-7"])
async def test_cancel_kills_the_turn_process(monkeypatch, tmp_path, session_id):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    agent = a2a_server.LettaCodeAgent()
    app = a2a_server.build_app(agent)
    assert app.state.agentenv_a2a.card.name == "Letta Code"
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def command(*args, **kwargs):
        return b'{"id":"agent-local-1"}'

    async def stream(proc, callback):
        started.set()
        await asyncio.Event().wait()

    class Proc:
        returncode = None
        def kill(self):
            self.returncode = -9
            stopped.set()
        async def wait(self):
            return self.returncode

    @asynccontextmanager
    async def relay(*args, **kwargs):
        yield "http://relay/v1"

    monkeypatch.setattr(bridge, "model_relay", relay)

    async def spawn(*args, **kwargs):
        if session_id:
            assert "--conversation" in args and "--agent" not in args
        else:
            assert "--agent" in args and "--new" in args
        return Proc()

    monkeypatch.setattr(bridge, "_command", command)
    monkeypatch.setattr(bridge, "_configure_agent", lambda *args: None)
    monkeypatch.setattr(bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bridge, "_stream_cli", stream)
    turn = asyncio.create_task(bridge.run_letta_code("wait", "ctx", session_id=session_id, workspace=str(tmp_path)))
    await asyncio.wait_for(started.wait(), 5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert stopped.is_set() and not bridge._RUNNING


def test_direct_mcp_catalog_replaces_previous_context_tools(tmp_path):
    bridge._install_mcp_tools(tmp_path, "agent-local-1", [{
        "name": "mcp__mail__search_emails", "description": "Search mail", "inputSchema": {"type": "object"},
    }])
    catalog = tmp_path / ".letta/agentenv-mcp-tools.json"
    assert json.loads(catalog.read_text())["tools"][0]["name"] == "mcp__mail__search_emails"
    assert (tmp_path / ".letta/mods/agentenv-mcp.ts").exists()
    bridge._install_mcp_tools(tmp_path, "agent-local-1", [])
    assert json.loads(catalog.read_text())["tools"] == []


def test_failed_workspace_swap_keeps_the_previous_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(a2a_server, "WORKSPACE_DIR", tmp_path / "app")
    target = bridge.context_data_dir("target")
    target.mkdir(parents=True)
    (target / "conversation.json").write_text("previous")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "kept.txt").write_text("previous workspace")
    staged_state = tmp_path / "staged-state"
    staged_state.mkdir()
    (staged_state / "conversation.json").write_text("restored")
    staged_workspace = tmp_path / "staged-workspace"
    staged_workspace.mkdir()
    (staged_workspace / "new.txt").write_text("restored workspace")

    def broken_move(*args, **kwargs):
        if str(args[0]) == str(staged_state):
            raise OSError("conversation move failed")
        return real_move(*args, **kwargs)

    real_move = bridge.shutil.move
    monkeypatch.setattr(bridge.shutil, "move", broken_move)
    with pytest.raises(OSError):
        a2a_server._replace_workspace(staged_workspace, lambda: bridge.commit_staged_session("target", staged_state))
    assert (tmp_path / "app" / "kept.txt").read_text() == "previous workspace"
    assert not (tmp_path / "app" / "new.txt").exists()
    assert (target / "conversation.json").read_text() == "previous"


def test_commit_staged_session_restores_previous_state_when_the_move_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    target = bridge.context_data_dir("target")
    target.mkdir(parents=True)
    (target / "conversation.json").write_text("previous")
    staged = tmp_path / "staged"
    staged.mkdir()

    def broken_move(*args, **kwargs):
        raise OSError("move failed")

    monkeypatch.setattr(bridge.shutil, "move", broken_move)
    with pytest.raises(OSError):
        bridge.commit_staged_session("target", staged)
    assert (target / "conversation.json").read_text() == "previous"
    assert not target.with_name(target.name + ".previous").exists()


@pytest.mark.asyncio
async def test_cancelled_caller_waits_for_the_file_thread():
    finished = []
    release = __import__("threading").Event()

    def work():
        release.wait(5)
        finished.append(True)

    task = asyncio.create_task(bridge.shielded_thread(work))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("forwarded", [False, True])
async def test_attribution_is_recorded_only_when_the_relay_forwards(monkeypatch, forwarded):
    monkeypatch.setitem(bridge._last_attribution, "project_id", None)
    monkeypatch.setitem(bridge._last_attribution, "task_id", None)

    @asynccontextmanager
    async def relay(upstream, calls, **kwargs):
        if forwarded:
            calls.add(200, model_relay.answer_usage(b'{"usage": {"prompt_tokens": 9, "completion_tokens": 1}}'))
        yield "http://relay"

    monkeypatch.setattr(bridge, "model_relay", relay)

    async def command(*args, **kwargs):
        return b'{"id":"agent-local-1"}'

    async def spawn(*args, **kwargs):
        raise OSError("cli did not start")

    monkeypatch.setattr(bridge, "_command", command)
    monkeypatch.setattr(bridge, "_configure_agent", lambda *args: None)
    monkeypatch.setattr(bridge.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(OSError):
        await bridge.run_letta_code("hi", "ctx-attr", project_id="p1", task_id="t1", workspace="/tmp")
    snapshot = bridge.get_last_attribution_snapshot()
    assert (snapshot["project_id"], snapshot["task_id"]) == (("p1", "t1") if forwarded else (None, None))


def test_cleanup_failure_after_a_successful_swap_keeps_the_new_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    target = bridge.context_data_dir("target")
    target.mkdir(parents=True)
    (target / "conversation.json").write_text("previous")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "conversation.json").write_text("restored")
    real_rmtree = bridge.shutil.rmtree

    def rmtree(path, ignore_errors=False, **kwargs):
        if str(path).endswith(".previous") and not ignore_errors:
            raise PermissionError("old tree is read-only")
        if str(path).endswith(".previous"):
            return None
        return real_rmtree(path, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(bridge.shutil, "rmtree", rmtree)
    bridge.commit_staged_session("target", staged)
    assert (target / "conversation.json").read_text() == "restored"


@pytest.mark.asyncio
async def test_a_load_cancelled_during_its_swap_still_binds_the_restored_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    source = bridge.context_data_dir("source")
    source.mkdir(parents=True)
    (source / "context.json").write_text(json.dumps({"home": str(source)}))
    snapshot = tmp_path / "state.tgz"
    snapshot.write_bytes(await bridge.export_session("source", "local-conv-7"))

    async def download(_object, target):
        shutil.copyfile(snapshot, target)

    swapping, release = threading.Event(), threading.Event()
    real_commit = a2a_server.commit_staged_session

    def commit(context_id, staged):
        swapping.set()
        release.wait(5)
        real_commit(context_id, staged)

    monkeypatch.setattr(a2a_server, "download", download)
    monkeypatch.setattr(a2a_server, "commit_staged_session", commit)
    agent = a2a_server.LettaCodeAgent()
    a2a_server.build_app(agent)
    request = SimpleNamespace(target_context_id="target", objects=SimpleNamespace(trajectory=None, workspace=None))
    load = asyncio.create_task(agent.load_snapshot(request))
    assert await asyncio.to_thread(swapping.wait, 5)
    load.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await load
    # The swap finished, so the context resumes the conversation it restored, not the one it had before.
    assert agent.session_ref_for_context("target") == "local-conv-7"


@pytest.mark.asyncio
async def test_a_cancelled_changelog_apply_holds_the_workspace_until_its_extract_ends(monkeypatch):
    applying, release = threading.Event(), threading.Event()

    def apply(_archive, _sequence):
        applying.set()
        release.wait(5)

    async def download(_object, target):
        target.write_bytes(b"")

    monkeypatch.setattr(a2a_server, "_apply_increment", apply)
    monkeypatch.setattr(a2a_server, "download", download)
    agent = a2a_server.LettaCodeChangelogAgent()
    a2a_server.build_app(agent)
    request = SimpleNamespace(target_context_id="ctx", increments=[SimpleNamespace(sequence=0, object=None)])
    task = asyncio.create_task(agent.apply_changelog(request))
    assert await asyncio.to_thread(applying.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    # No turn may start in the workspace while the extract still writes to it.
    assert not task.done() and agent._workspace._writer
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not agent._workspace._writer


# letta-code-baseline-gpt on gpt-4.1-mini, recorded on 2026-11-21 through Scale's LiteLLM proxy: each call's usage as the
# proxy answered the relay.
GPT_TURN = [
    {"prompt_tokens": 22860, "completion_tokens": 72, "total_tokens": 22932,
     "prompt_tokens_details": {"cached_tokens": 0}},
    {"prompt_tokens": 22993, "completion_tokens": 2, "total_tokens": 22995,
     "prompt_tokens_details": {"cached_tokens": 22784}},
]


@pytest.mark.asyncio
async def test_a_turn_runs_through_a_relay_with_the_endpoints_key_and_reports_what_it_counted(monkeypatch, tmp_path):
    """The relay sends the endpoint's key and the run's headers, lists the turn's model alone, and counts every call
    the CLI makes; the CLI is connected to it with a placeholder key."""
    monkeypatch.setattr(bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-endpoint")
    relays, commands = [], []

    @asynccontextmanager
    async def relay(upstream, calls, **kwargs):
        relays.append((upstream, kwargs))
        for usage in GPT_TURN:
            calls.add(200, model_relay.answer_usage(json.dumps({"usage": usage}).encode()))
        yield "http://127.0.0.1:41000/token"

    async def command(env, *args):
        commands.append((args, env[bridge._TURN_MARKER]))
        return b'{"id":"agent-local-1"}'

    class Proc:
        returncode = 0

        async def wait(self):
            return 0

    async def spawn(*args, **kwargs):
        return Proc()

    async def stream(proc, on_line):
        on_line(json.dumps({"type": "result", "subtype": "success", "result": "42", "conversation_id": "c-1"}))
        return b""

    monkeypatch.setattr(bridge, "model_relay", relay)
    monkeypatch.setattr(bridge, "_command", command)
    monkeypatch.setattr(bridge, "_configure_agent", lambda *args: None)
    monkeypatch.setattr(bridge.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bridge, "_stream_cli", stream)

    result = await bridge.run_letta_code("hi", "ctx-relay", model="gpt-4.1-mini", project_id="p1",
                                         workspace=str(tmp_path))

    ((upstream, options),) = relays
    assert upstream == "https://gateway.example"
    assert options["models"] == ["gpt-4.1-mini"]
    assert options["headers"]["authorization"] == "Bearer sk-endpoint"
    assert options["headers"]["x-litellm-customer-id"] == "p1"
    connect, = [args for args, _ in commands if args[0] == "connect"]
    assert connect == ("connect", "openai-compatible", "--base-url", "http://127.0.0.1:41000/token/v1",
                       "--api-key", "relay")
    assert len({marker for _, marker in commands}) == 1
    # The proxy's own usage for the turn: 45,853 in (22,784 cached) and 74 out.
    assert result.usage.to_dict() == {
        "tool_call_count": 0, "input_tokens": 45853, "output_tokens": 74, "total_tokens": 45927,
        "cache_read_tokens": 22784, "input_includes_cache": True,
        "models": [{"model": "gpt-4.1-mini", "calls": 2, "input_tokens": 45853, "output_tokens": 74,
                    "cache_read_tokens": 22784, "input_includes_cache": True}],
        "provider_details": {"model_calls": 2}}


def test_a_turns_leftover_letta_processes_are_stopped_and_nothing_else(monkeypatch, tmp_path):
    """Found by the turn's marker in a process's environment and Letta as its executable or script: another turn's,
    and the model's own background jobs, even one whose arguments name Letta, keep running."""
    def process(pid: int, environ: dict[str, str], *argv: str) -> None:
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in environ.items()) + b"\0")
        (folder / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")

    mine, theirs = {bridge._TURN_MARKER: "turn-1"}, {bridge._TURN_MARKER: "turn-2"}
    process(101, mine, "node", "/opt/agent/node_modules/@letta-ai/letta-code/letta.js", "server", "--listen")
    process(102, {**mine, "LETTA_SUBAGENT_LAUNCH": "1"}, "node", "/usr/local/bin/letta", "-p", "subtask")
    process(103, mine, "python3", "-m", "http.server", "8080")
    process(105, mine, "python", "/app/letta-demo/server.py", "--name", "letta")
    process(104, theirs, "node", "/usr/local/bin/letta", "server")
    (tmp_path / "self").mkdir()
    killed = []
    monkeypatch.setattr(bridge, "_PROC", tmp_path)
    monkeypatch.setattr(bridge.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    assert sorted(bridge._stop_leftovers("turn-1")) == [101, 102]
    assert sorted(killed) == [(101, signal.SIGKILL), (102, signal.SIGKILL)]


@pytest.mark.asyncio
async def test_a_caller_cancelled_twice_still_waits_for_the_work_and_its_last_step():
    release = threading.Event()
    finished = []

    async def work():
        await asyncio.to_thread(release.wait, 5)
        finished.append(True)

    task = asyncio.create_task(bridge.shielded(work()))
    await asyncio.sleep(0.05)
    for _ in range(2):
        task.cancel()
        await asyncio.sleep(0.05)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [True]


def test_web_tools_is_a_no_op_checked_on_the_cli_version_and_backend_the_bridge_runs():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["@letta-ai/letta-code"]

    assert bridge.NO_WEB_TOOLS_CLI_VERSION == pinned, (
        "the CLI version changed: check that a local-backend agent still has no web tools of its own, then move "
        "NO_WEB_TOOLS_CLI_VERSION")
    assert '"--backend", "local"' in (AGENT_DIR / "letta_code_bridge.py").read_text(), (
        "the bridge left the local backend: a cloud-backend agent has web_search and fetch_webpage, which web_tools "
        "False must then turn off")
