"""What a turn hands the SDK and when it runs: turns of different contexts at once and a context's own one at a time,
each resuming the conversation the last one left; a snapshot alone; the task's config, attachments and MCP servers;
and how a turn's answer, usage and failure reach the A2A task."""

from __future__ import annotations

import asyncio

import pytest
from agentenv_protocol.a2a_agent import Uploaded

from tst.unit.agents.antigravity import fake_sdk
from tst.unit.agents.antigravity.agent import a2a_server, antigravity_bridge, antigravity_kit
from tst.unit.agents.antigravity.fake_sdk import Text, ToolCall, Turn


def result(**fields) -> antigravity_bridge.AntigravityResult:
    """What run_antigravity returns, a finished turn's by default."""
    return antigravity_bridge.AntigravityResult(**{"response": "done", "session_id": "conv-1", **fields})


class Turns:
    """Stand in for run_antigravity: each call's arguments, held until ``release`` when it is set."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.release: asyncio.Event | None = None
        self.answer = lambda kwargs: result(session_id=kwargs["session_id"] or f"conv-{len(self.calls)}")

    async def __call__(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.release is not None:
            await self.release.wait()
        return self.answer(kwargs)


@pytest.fixture
def turns(monkeypatch) -> Turns:
    stand_in = Turns()
    monkeypatch.setattr(a2a_server, "run_antigravity", stand_in)
    return stand_in


async def _until(condition, seconds: float = 5) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.01)


def _status(task: dict) -> tuple[str, list[dict]]:
    return task["status"]["state"], task["status"]["message"]["parts"]


@pytest.mark.asyncio
async def test_a_turns_answer_and_usage_are_the_tasks(ag):
    fake_sdk.Agent.turns.append(Turn(steps=[
        ("chunk", Text(step_index=0, text="checking")),
        ("call", ToolCall(id="h1", name="run_command")),
        ("chunk", ToolCall(id="s1", name="run_command", step_index=0)),
        ("chunk", Text(step_index=1, text="42")),
    ]))
    async with ag.client() as client:
        task = await ag.send(client, "what is 6 * 7?")

    state, parts = _status(task)
    assert (state, parts[0]["text"]) == ("completed", "42")
    assert parts[1]["data"]["usage"]["tool_call_count"] == 1
    assert fake_sdk.Agent.prompts == ["what is 6 * 7?"]


@pytest.mark.asyncio
async def test_a_contexts_next_turn_resumes_its_conversation_and_another_contexts_starts_its_own(ag, folders):
    fake_sdk.Agent.turns.extend([Turn(conversation_id="conv-a"), Turn(conversation_id="conv-a"),
                                 Turn(conversation_id="conv-b")])
    async with ag.client() as client:
        await ag.send(client, "one", "ctx-a")
        (antigravity_bridge.SAVE_DIR / "conv-a.db").write_bytes(b"")
        await ag.send(client, "two", "ctx-a")
        await ag.send(client, "three", "ctx-b")

    assert [config.__dict__.get("conversation_id") for config in fake_sdk.Agent.configs] == [None, "conv-a", None]
    assert ag.agent.session_ref_for_context("ctx-b") == "conv-b"


@pytest.mark.asyncio
@pytest.mark.parametrize("raised, code, error_type", [
    (fake_sdk.AntigravityExecutionError("MCP load failed: env: 502"), "mcp_load_failed", "infra_error"),
    (fake_sdk.AntigravityExecutionError("localharness exited 2"), "antigravity_error", "agent_error"),
])
async def test_a_failed_turn_fails_the_task_with_its_code(ag, raised, code, error_type):
    fake_sdk.Agent.turns.append(Turn(steps=[("raise", raised)]))
    async with ag.client() as client:
        task = await ag.send(client, "go")

    state, parts = _status(task)
    assert (state, parts[0]["text"]) == ("failed", str(raised))
    errors = [part["data"] for part in parts if part.get("kind") == "data" and "error_code" in part["data"]]
    assert {"error_type": "antigravity_error", "error_code": code, "error_message": str(raised)} in errors
    assert any(part["data"].get("error_type") == error_type for part in parts if part.get("kind") == "data")


@pytest.mark.asyncio
async def test_a_turn_the_bridge_cant_run_is_an_infra_failure(monkeypatch, ag):
    async def broken(_prompt, **_kwargs):
        raise RuntimeError("relay port in use")

    monkeypatch.setattr(a2a_server, "run_antigravity", broken)
    async with ag.client() as client:
        task = await ag.send(client, "go")

    state, parts = _status(task)
    assert (state, parts[0]["text"]) == ("failed", "Antigravity invocation failed")
    assert any(part["data"].get("error_type") == "infra_error" for part in parts if part.get("kind") == "data")


@pytest.mark.asyncio
async def test_turns_of_different_contexts_run_at_once(ag, turns):
    turns.release = asyncio.Event()
    async with ag.client() as client:
        sent = [asyncio.create_task(ag.send(client, "go", context_id)) for context_id in ("ctx-a", "ctx-b")]
        await _until(lambda: len(turns.calls) == 2)
        turns.release.set()
        tasks = await asyncio.gather(*sent)

    assert [task["status"]["state"] for task in tasks] == ["completed", "completed"]


@pytest.mark.asyncio
async def test_a_contexts_turns_run_one_at_a_time_each_resuming_the_last(ag, turns):
    turns.release = asyncio.Event()
    async with ag.client() as client:
        first = asyncio.create_task(ag.send(client, "one"))
        await _until(lambda: len(turns.calls) == 1)
        second = asyncio.create_task(ag.send(client, "two"))
        await asyncio.sleep(0.1)
        running_together = len(turns.calls)
        turns.release.set()
        await asyncio.gather(first, second)

    assert running_together == 1
    assert [call["session_id"] for call in turns.calls] == [None, "conv-1"]


@pytest.mark.asyncio
async def test_a_snapshot_waits_for_running_turns_and_holds_new_ones_back(monkeypatch, ag, turns):
    antigravity_bridge.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (antigravity_bridge.SAVE_DIR / "conv-a.db").write_bytes(b"db")
    ag.agent.set_session_ref_for_context("ctx-a", "conv-a")
    archived = asyncio.Event()
    archive = a2a_server._archive_conversation

    def archiving(conversation_id, path):
        archived.set()
        archive(conversation_id, path)

    async def upload(_target, source):
        return Uploaded(size_bytes=1)

    monkeypatch.setattr(a2a_server, "_archive_conversation", archiving)
    monkeypatch.setattr(a2a_server, "upload", upload)
    turns.release = asyncio.Event()
    async with ag.client() as client:
        running = asyncio.create_task(ag.send(client, "go", "ctx-b"))
        await _until(lambda: len(turns.calls) == 1)
        save = asyncio.create_task(client.post("/ext/snapshot", json={"context_id": "ctx-a", "objects": {
            "trajectory": _WRITE}}))
        await asyncio.sleep(0.1)
        held = asyncio.create_task(ag.send(client, "go", "ctx-c"))
        await asyncio.sleep(0.1)
        while_running = (archived.is_set(), len(turns.calls))
        turns.release.set()
        saved = await save
        await asyncio.gather(running, held)

    assert while_running == (False, 1)
    assert saved.status_code == 200, saved.text
    assert len(turns.calls) == 2


@pytest.mark.asyncio
async def test_the_turns_config_and_workspace_reach_the_sdk(ag, turns, folders):
    config = {"model": "provider/model", "effort": "high", "system_prompt": "be brief", "project_id": "p-1",
              "task_id": "t-1", "role": "solver", "timeout_seconds": 42, "max_turns": 45,
              "builtin_tools": {"disable": ["search_web", "read_url_content"]}}
    async with ag.client() as client:
        assert (await client.post("/ext/agent-config", json=config)).status_code == 200
        await ag.send(client, "go")

    (call,) = turns.calls
    assert {key: call[key] for key in ("prompt", "model", "effort", "system_prompt", "project_id", "task_id", "role",
                                       "timeout_s", "workspace", "max_turns")} == {
        "prompt": "go", "model": "provider/model", "effort": "high", "system_prompt": "be brief",
        "project_id": "p-1", "task_id": "t-1", "role": "solver", "timeout_s": 42, "workspace": str(folders),
        "max_turns": 45}
    assert call["builtin_tools"].disable == ["search_web", "read_url_content"]


@pytest.mark.asyncio
async def test_a_turn_gets_the_sdks_peer_tools_with_its_registered_servers(monkeypatch, ag, turns):
    monkeypatch.setenv("A2A_PORT", "8000")
    async with ag.client() as client:
        added = await client.post("/ext/mcp-config", json={"name": "env", "url": "https://mcp.example/mcp",
                                                           "headers": {"X-Key": "k"}})
        await ag.send(client, "go")

    assert added.status_code == 200, added.text
    (call,) = turns.calls
    assert call["mcp_servers"]["env"] == {"url": "https://mcp.example/mcp", "headers": {"X-Key": "k"}}
    assert call["mcp_servers"]["peer_agents"] == {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "headers": None}


@pytest.mark.asyncio
async def test_a_timed_out_turn_keeps_the_conversation_it_started_for_the_next(ag, turns):
    turns.answer = lambda kwargs: (
        result(response="", error="the turn ran past its 1 s", error_code="timeout", session_id="conv-started")
        if not kwargs["session_id"] else result(session_id=kwargs["session_id"]))
    async with ag.client() as client:
        timed_out = await ag.send(client, "one")
        await ag.send(client, "two")

    assert timed_out["status"]["state"] == "failed"
    assert [call["session_id"] for call in turns.calls] == [None, "conv-started"]


@pytest.mark.asyncio
async def test_a_file_part_is_written_under_the_task_and_attached(ag, turns, folders):
    parts = [{"kind": "text", "text": "describe"},
             {"kind": "file", "file": {"name": "../a b.png", "mimeType": "image/png", "bytes": "cG5n"}}]
    async with ag.client() as client:
        task = await ag.send(client, parts)

    (call,) = turns.calls
    (attached,) = call["attachment_paths"]
    assert attached == str(folders / "inputs" / task["id"] / "a_b.png")
    assert open(attached, "rb").read() == b"png"


@pytest.mark.asyncio
async def test_an_unreadable_file_part_fails_the_turn_before_the_sdk_runs(ag, turns):
    parts = [{"kind": "file", "file": {"name": "a.png", "mimeType": "image/png", "bytes": "not base64!"}}]
    async with ag.client() as client:
        task = await ag.send(client, parts)

    assert task["status"]["state"] == "failed" and turns.calls == []


@pytest.mark.asyncio
async def test_an_inline_skill_without_frontmatter_gets_one_from_its_name(ag):
    async with ag.client() as client:
        added = await client.post("/ext/skill-config", json={"name": "report", "description": "Write the\nreport.",
                                                             "skill_md": "# Report\nSteps."})
        listed = await client.get("/ext/skill-config")

    assert added.status_code == 200, added.text
    assert (antigravity_bridge.SKILLS_DIR / "report" / "SKILL.md").read_text() == (
        '---\nname: "report"\ndescription: "Write the report."\n---\n\n# Report\nSteps.')
    assert "report" in listed.json()["skills"]


def test_a_skill_with_its_own_frontmatter_is_written_as_given():
    written = "---\nname: mine\ndescription: d\n---\nbody"

    assert antigravity_kit.with_frontmatter(written, "other", "x") == written


@pytest.mark.parametrize("skill_md", ["---\n\n# Report\nSteps.", "---\n\n# Report\n\n---\n\nMore steps."])
def test_a_skill_that_opens_with_a_horizontal_rule_gets_frontmatter(skill_md):
    assert antigravity_kit.with_frontmatter(skill_md, "report", "Write it.") == (
        f'---\nname: "report"\ndescription: "Write it."\n---\n\n{skill_md}')
    assert antigravity_kit.with_frontmatter("body", 'a "quoted" name', None).startswith(
        '---\nname: "a \\"quoted\\" name"\ndescription: "a \\"quoted\\" name"\n---')


def test_conversation_references_outlive_the_frameworks_bounded_record():
    assert issubclass(a2a_server.AntigravityAgent, antigravity_kit.DurableSessionRefs)


_GRANT = {"expires_at": "2099-01-01T00:00:00Z", "url": "https://objects.example/o"}
_WRITE = {"media_type": "application/octet-stream", "max_bytes": 1024, "write": {"kind": "http-put", **_GRANT}}
