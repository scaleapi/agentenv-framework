"""Pins which codex NDJSON events fail a turn.

Codex emits item-level `error` items and top-level `error` events for advisory
notices and transient retries on turns that then succeed; treating those as fatal
failed every such turn. Only `turn.failed` / `thread.failed` or a non-zero exit do.

Every event literal below is verbatim from a real codex 0.147 `exec --json` run.
"""
import asyncio
import json
from pathlib import Path

import pytest

from tst.unit.agents.codex.agent import codex_bridge

THREAD_STARTED = {"type": "thread.started", "thread_id": "019feca9-6ad2-7753-bbc4-0f4b402b3af3"}
TURN_STARTED = {"type": "turn.started"}
TURN_COMPLETED = {"type": "turn.completed"}

# The turn proceeds normally after these two.
ADVISORY_MODEL_METADATA = {
    "type": "item.completed",
    "item": {
        "id": "item_0",
        "type": "error",
        "message": (
            "Model metadata for `gpt-5-codex` not found. Defaulting to fallback "
            "metadata; this can degrade performance and cause issues."
        ),
    },
}
ADVISORY_UNSTABLE_FEATURE = {
    "type": "item.completed",
    "item": {
        "id": "item_1",
        "type": "error",
        "message": "Under-development features enabled: code_mode. Under-development features are incomplete and may behave unpredictably.",
    },
}
# Retries arrive as *top-level* error events; the turn can still complete.
TRANSIENT_RETRY = {
    "type": "error",
    "message": "Reconnecting... 1/2 (We're currently experiencing high demand, which may cause temporary errors.)",
}
TURN_FAILED = {
    "type": "turn.failed",
    "error": {"message": "stream disconnected before completion: error sending request for url"},
}


def agent_message(text):
    return {"type": "item.completed", "item": {"id": "item_9", "type": "agent_message", "text": text}}


def tool_item(item_type):
    return {"type": "item.completed", "item": {"id": "item_t", "type": item_type}}


def ndjson(*events):
    return ("".join(json.dumps(e) + "\n" for e in events)).encode()


class _FakeStream:
    """Minimal asyncio.StreamReader stand-in."""

    def __init__(self, data: bytes):
        self._data = data

    async def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            data, self._data = self._data, b""
            return data
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk


class _FakeStdin:
    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, data: bytes):
        self.data += data

    async def drain(self):
        return None

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.returncode = returncode

    async def wait(self):
        return self.returncode

    def kill(self):
        pass


@pytest.fixture
def spawn(monkeypatch, tmp_path):
    """Install a fake Codex subprocess."""
    monkeypatch.chdir(tmp_path)
    capture: dict = {}

    def _install(stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        async def fake_exec(*cmd, stdin=None, stdout=None, stderr=None, env=None, **kw):
            capture["cmd"] = list(cmd)
            capture["stdin"] = stdin
            capture["env"] = env
            capture["start_new_session"] = kw.get("start_new_session")
            capture["proc"] = _FakeProc(
                _install.stdout, _install.stderr, _install.returncode
            )
            return capture["proc"]

        _install.stdout, _install.stderr, _install.returncode = stdout, stderr, returncode
        monkeypatch.setattr(codex_bridge.asyncio, "create_subprocess_exec", fake_exec)
        return capture

    return _install


def run(spawn, events, returncode=0, stderr=b"", **kwargs):
    spawn(ndjson(*events), stderr=stderr, returncode=returncode)
    return asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None, **kwargs))


@pytest.mark.parametrize("noise, expected_warnings", [
    ([ADVISORY_MODEL_METADATA], 1),
    ([ADVISORY_UNSTABLE_FEATURE], 1),
    ([TRANSIENT_RETRY], 1),
    ([ADVISORY_MODEL_METADATA, TRANSIENT_RETRY], 2),
])
def test_advisories_and_retries_do_not_fail_a_completed_turn(spawn, noise, expected_warnings):
    events = [THREAD_STARTED, *noise, TURN_STARTED, tool_item("mcp_tool_call"),
              agent_message("done"), TURN_COMPLETED]
    res = run(spawn, events, returncode=0)
    assert (res.error, res.response, len(res.warnings), res.tool_call_count) == (
        None, "done", expected_warnings, 1)
    # The trajectory extension depends on every event surviving, noise included.
    assert [e.get("type") for e in res.trajectory] == [e["type"] for e in events]


@pytest.mark.parametrize("events, returncode", [
    ([THREAD_STARTED, TURN_STARTED, TURN_FAILED], 1),
    ([THREAD_STARTED, TURN_STARTED, TURN_FAILED], 0),  # turn.failed alone is enough
    ([], 1),                                           # so is a bare non-zero exit
])
def test_real_failures_are_fatal(spawn, events, returncode):
    assert run(spawn, events, returncode=returncode).error is not None


def test_error_message_carries_stderr_not_advisories(spawn):
    # No turn.failed, so stderr explains the exit; the advisory is on every run.
    res = run(spawn, [THREAD_STARTED, ADVISORY_MODEL_METADATA], returncode=1,
              stderr=b"boom from stderr")
    assert res.error == "codex CLI exited with code 1: boom from stderr"


def test_tool_call_count_counts_only_tool_items(spawn):
    res = run(spawn, [
        THREAD_STARTED, TURN_STARTED,
        tool_item("command_execution"), tool_item("mcp_tool_call"),
        tool_item("file_change"), tool_item("web_search"),
        tool_item("todo_list"), tool_item("reasoning"),
        agent_message("x"), TURN_COMPLETED,
    ])
    assert res.tool_call_count == 4, "todo_list/reasoning must not count as tool calls"


def test_response_is_only_the_last_agent_message(spawn):
    res = run(spawn, [
        THREAD_STARTED, TURN_STARTED,
        agent_message("Let me check the files."), tool_item("command_execution"),
        agent_message("Found it."), tool_item("mcp_tool_call"),
        agent_message("final answer"), TURN_COMPLETED,
    ])
    assert res.response == "final answer"
    assert [e["item"]["text"] for e in res.trajectory
            if e.get("item", {}).get("type") == "agent_message"] == [
        "Let me check the files.", "Found it.", "final answer"]


def test_schema_shaped_interim_message_does_not_leak_into_response(spawn):
    # Joined, these were `{"results":[]}{...}`, which json.loads rejects.
    verdict = {"results": [{"id": "c1", "score": 1.0, "justification": "ok"}]}
    res = run(spawn, [
        THREAD_STARTED, TURN_STARTED,
        agent_message('{"results":[]}'), tool_item("mcp_tool_call"),
        agent_message(json.dumps(verdict)), TURN_COMPLETED,
    ], output_format={"type": "json_schema", "schema": {"type": "object"}})
    assert json.loads(res.response) == verdict


def test_turn_without_agent_message_has_empty_response(spawn):
    res = run(spawn, [THREAD_STARTED, TURN_STARTED, tool_item("mcp_tool_call"), TURN_COMPLETED])
    assert (res.error, res.response) == (None, "")


def test_malformed_json_line_is_skipped_not_fatal(spawn):
    spawn(json.dumps(THREAD_STARTED).encode() + b"\nnot json\n"
          + json.dumps(agent_message("x")).encode() + b"\n"
          + json.dumps(TURN_COMPLETED).encode() + b"\n")
    res = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert (res.error, res.response) == (None, "x")


def test_event_callback_receives_streamed_events(spawn):
    cap = spawn(ndjson(THREAD_STARTED, TURN_STARTED, TURN_COMPLETED))
    seen = []
    asyncio.run(
        codex_bridge.run_codex(
            prompt="p", thread_id=None, event_callback=seen.append
        )
    )
    assert [event["type"] for event in seen] == [
        "thread.started",
        "turn.started",
        "turn.completed",
    ]
    assert cap["proc"].stdin.data == b"p"


@pytest.mark.parametrize("thread_id, expected_argv", [
    (None, ["codex", "exec", "--cd", "/app", "-"]),
    ("tid-1", ["codex", "exec", "--cd", "/app", "resume", "tid-1", "-"]),
])
def test_stdin_and_session_binding(spawn, thread_id, expected_argv):
    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    res = asyncio.run(codex_bridge.run_codex(prompt="hello", thread_id=thread_id,
                                             model="gpt-5.3-codex"))
    assert cap["cmd"][:len(expected_argv)] == expected_argv
    assert {"--json", "--dangerously-bypass-approvals-and-sandbox",
            'model="gpt-5.3-codex"'} <= set(cap["cmd"])
    assert "hello" not in cap["cmd"]
    assert cap["stdin"] is asyncio.subprocess.PIPE
    assert cap["start_new_session"] is True
    assert cap["proc"].stdin.data == b"hello"
    assert cap["proc"].stdin.closed is True
    assert res.session_id == THREAD_STARTED["thread_id"]


def test_oversized_prompt_never_lands_on_argv(spawn):
    prompt = "criteria " * 45_000
    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    asyncio.run(codex_bridge.run_codex(prompt=prompt, thread_id=None))
    assert not any(prompt in arg for arg in cap["cmd"])
    assert cap["proc"].stdin.data == prompt.encode()


def test_empty_prompt_sends_explicit_stdin_record(spawn):
    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    asyncio.run(codex_bridge.run_codex(prompt="", thread_id=None))
    assert cap["proc"].stdin.data == b"\n"


def test_thread_id_survives_a_missing_thread_started_event(spawn):
    spawn(ndjson(agent_message("x"), TURN_COMPLETED))
    res = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id="pre-existing"))
    assert res.session_id == "pre-existing"


def test_usage_is_incremental_from_existing_rollout(
    spawn, monkeypatch
):
    completed = {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 180,
            "cached_input_tokens": 130,
            "cache_write_input_tokens": 40,
            "output_tokens": 15,
        },
    }
    monkeypatch.setattr(
        codex_bridge,
        "read_rollout_usage",
        lambda _: {
            "input_tokens": 100,
            "cached_input_tokens": 60,
            "cache_write_input_tokens": 10,
            "output_tokens": 10,
        },
    )
    spawn(ndjson(agent_message("x"), completed))
    result = asyncio.run(
        codex_bridge.run_codex(prompt="p", thread_id="pre-existing")
    )
    assert result.usage == completed["usage"]
    assert result.token_usage == {
        "input_tokens": 80,
        "cached_input_tokens": 70,
        "cache_write_input_tokens": 30,
        "output_tokens": 5,
    }


def test_native_rollout_usage_is_read_for_snapshot_resume(monkeypatch, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}})
        + "\n"
        + json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 60,
                            "cache_write_input_tokens": 20,
                            "output_tokens": 10,
                        }
                    },
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: rollout)
    assert codex_bridge.read_rollout_usage("thread-1") == {
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "cache_write_input_tokens": 20,
        "output_tokens": 10,
    }


def test_system_prompt_is_per_invocation_config_not_agents_md(spawn, tmp_path):
    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    system_prompt = 'BE "TERSE"\nUse tools. 🚀'
    asyncio.run(
        codex_bridge.run_codex(
            prompt="p",
            thread_id=None,
            system_prompt=system_prompt,
        )
    )
    expected = (
        "developer_instructions="
        f"{json.dumps(system_prompt, ensure_ascii=False)}"
    )
    assert expected in cap["cmd"]
    assert "\\ud83d" not in expected
    assert not (tmp_path / "AGENTS.md").exists()

    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert not any("developer_instructions=" in arg for arg in cap["cmd"])
    assert not (tmp_path / "AGENTS.md").exists()


def test_attribution_env_is_set_only_when_ids_are_given(spawn, monkeypatch):
    monkeypatch.delenv("LITELLM_CUSTOMER_ID", raising=False)
    monkeypatch.delenv("LITELLM_TAGS", raising=False)
    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert not {"LITELLM_CUSTOMER_ID", "LITELLM_TAGS"} & set(cap["env"])

    asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None,
                                       project_id="proj-1", task_id="task-2"))
    snap = codex_bridge.get_last_attribution_snapshot()
    assert (cap["env"]["LITELLM_CUSTOMER_ID"], cap["env"]["LITELLM_TAGS"],
            snap["project_id"], snap["task_id"]) == (
        "proj-1", "projectId:proj-1,taskId:task-2", "proj-1", "task-2")


def test_each_override_reaches_the_cli_as_its_own_c_flag(spawn):
    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None, config_overrides=['a.b="x"', "c=1"]))

    at = cap["cmd"].index('a.b="x"')
    assert cap["cmd"][at - 1:at + 3] == ["-c", 'a.b="x"', "-c", "c=1"]


def test_a_cli_is_tracked_while_it_runs_a_turn(spawn):
    running = []
    spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None,
                                       event_callback=lambda event: running.append(len(codex_bridge._RUNNING))))

    assert running == [1, 1] and codex_bridge._RUNNING == set()


def test_the_runs_tags_join_the_ones_model_headers_name():
    headers = json.dumps({"X-LiteLLM-Tags": "useCase:evals,taskId:own"})

    assert codex_bridge.cli_env({"LITELLM_EXTRA_HEADERS": headers}, project_id="p-1", task_id="t-1")[
        "LITELLM_TAGS"] == "useCase:evals,taskId:own,projectId:p-1"
    assert codex_bridge.cli_env({"LITELLM_EXTRA_HEADERS": headers})["LITELLM_TAGS"] == "useCase:evals,taskId:own"


def test_the_cli_gets_the_endpoint_key_under_the_name_codex_prechecks():
    assert codex_bridge.cli_env({"LITELLM_API_KEY": "k"})["OPENAI_API_KEY"] == "k"
    assert codex_bridge.cli_env({"LITELLM_API_KEY": "k", "OPENAI_API_KEY": "own"})["OPENAI_API_KEY"] == "own"
    assert "OPENAI_API_KEY" not in codex_bridge.cli_env({})


def test_output_schema_is_written_to_a_temp_file_and_cleaned_up(spawn):
    cap = spawn(ndjson(THREAD_STARTED, TURN_COMPLETED))
    asyncio.run(codex_bridge.run_codex(
        prompt="p", thread_id=None,
        output_format={"type": "json_schema", "schema": {"type": "object"}}))
    path = Path(cap["cmd"][cap["cmd"].index("--output-schema") + 1])
    assert not path.exists(), "schema temp file should be removed after the run"

    with pytest.raises(ValueError):
        asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None,
                                           output_format={"schema": {}}))
