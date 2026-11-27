"""The Antigravity SDK the agent's requirements.txt pins, against what its bridge relies on: the harness opens each turn's
user message with the local-time line the turn's relay rewrites on an armed clock, and writes the machine's date
nowhere else, so the model is sent the env's time alone, and a resumed conversation's earlier turn keeps the time it
was given; the harness is a static binary, which libfaketime can't reach; the n-th tool call the SDK's hooks see is the
n-th it streams, so a result is tied to its call; and the SDK's tools are those AntigravityTool and the two always off
name. Bumping the SDK runs this against the new version, so a version that moves any of these fails here."""

import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tst.unit.agents.antigravity.agent import AGENT_DIR, PYTHONPATH, antigravity_bridge, antigravity_date
from tst.util import fake_openai

pytestmark = [pytest.mark.integration]

ENV_TIMES = ("1950-01-01T09:30:00Z", "1950-01-02T08:15:00Z", "1950-01-03T07:00:00Z")
# The env's time from the first turn's second model call on: a day past the first turn's line.
MOVED_TO = "1950-01-02T06:00:00Z"
# How long the endpoint holds the stalled turn's second model call: past that turn's 20 s.
_STALL_S = 40
# The tools the first turn calls, in order, by a word of the name the model is offered each by: a command, the
# harness's own skill listing, which its hooks never decide, and the workspace's file.
TOOLS = ("command", "list_skills", "view_file")
_DAY_S = 86400
_PT_INTERP = 3

# Two turns of one conversation through the bridge, on an armed clock whose time is the next of argv's, with the real
# SDK, then the first turn of another that runs past its time; their results, trajectories and what the SDK holds, as
# JSON.
DRIVER = '''
import asyncio, json, sys
from pathlib import Path
import google.antigravity as ag
from google.antigravity import types as agt
import antigravity_bridge, antigravity_date

times = iter(sys.argv[2:-1])
antigravity_date.clock_armed = lambda: True
reads, moved = 0, None
async def env_now():
    global reads
    reads += 1
    return moved if moved and reads > 1 else current
antigravity_date._env_now = env_now

async def turn(prompt, session_id, timeout_s=300):
    events = []
    result = await antigravity_bridge.run_antigravity(
        prompt, model="m", session_id=session_id, mcp_servers=None, system_prompt=None, timeout_s=timeout_s, role=None,
        builtin_tools=None, web_tools=True, max_turns=None, workspace=sys.argv[1], event_callback=events.append)
    return {"response": result.response, "error": result.error, "error_code": result.error_code,
            "session_id": result.session_id, "tool_calls": result.tool_call_count, "usage": result.usage.to_dict(),
            "events": events, "stored": antigravity_bridge.conversation_store_exists(result.session_id)}

current, moved = next(times), sys.argv[-1]
first = asyncio.run(turn("Run the shell command `echo 42` and reply with what it prints.", None))
current, moved, reads = next(times), None, 0
second = asyncio.run(turn("Thanks. Reply with the word done.", first["session_id"]))
current = next(times)
stalled = asyncio.run(turn("STALL: run the shell command `echo 42`.", None, timeout_s=20))
print(json.dumps({"turns": [first, second, stalled], "tools": sorted(tool.value for tool in agt.BuiltinTools),
                  "harness": str(Path(ag.__file__).parent / "bin" / "localharness")}))
'''


@pytest.fixture(scope="module")
def sdk_python(tmp_path_factory) -> Path:
    """A Python with the agent's requirements, the pinned SDK among them, as the image installs them."""
    uv = shutil.which("uv")
    assert uv, "uv installs the pinned Antigravity SDK, as install.sh does"
    venv = tmp_path_factory.mktemp("antigravity") / "venv"
    subprocess.run([uv, "venv", "--quiet", "--python", sys.executable, str(venv)], check=True, timeout=120)
    python = venv / "bin" / "python"
    subprocess.run([uv, "pip", "install", "--quiet", "--python", str(python), "-r", str(AGENT_DIR / "requirements.txt")],
                   check=True, timeout=900)
    return python


def _arguments(schema: dict, workspace: Path, tool: str) -> dict:
    """Arguments for ``tool``, whatever the SDK names them: a command that runs ``echo 42`` in the workspace, the
    workspace listed, or its file viewed."""
    arguments = {}
    for name in schema.get("required") or []:
        kind = (schema.get("properties") or {}).get(name, {}).get("type")
        lowered = name.lower()
        if "command" in lowered:
            arguments[name] = "echo 42"
        elif "cwd" in lowered or "dir" in lowered or "path" in lowered or "file" in lowered:
            arguments[name] = str(workspace / "notes.txt" if tool == "view_file" else workspace)
        elif (schema.get("properties") or {}).get(name, {}).get("enum"):
            arguments[name] = schema["properties"][name]["enum"][0]
        elif kind == "boolean":
            arguments[name] = False
        elif kind in ("integer", "number"):
            arguments[name] = 10000
        else:
            arguments[name] = ""
    return arguments


def _user_texts(body: dict) -> list[str]:
    texts = []
    for message in body.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"text": content}]
        texts += [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    return texts


def _machine_dates() -> set[str]:
    return {time.strftime("%Y-%m-%d", time.localtime(time.time() + offset)) for offset in (-_DAY_S, 0, _DAY_S)} | {
        time.strftime("%Y-%m-%d", time.gmtime(time.time() + offset)) for offset in (-_DAY_S, 0, _DAY_S)}


def _is_static(binary: Path) -> bool:
    """Whether an ELF binary has no program interpreter, the dynamic loader that would preload libfaketime."""
    data = binary.read_bytes()
    assert data[:4] == b"\x7fELF" and data[4] == 2, f"{binary} isn't a 64-bit ELF binary"
    endian = "<" if data[5] == 1 else ">"
    phoff, = struct.unpack_from(f"{endian}Q", data, 0x20)
    phentsize, phnum = struct.unpack_from(f"{endian}HH", data, 0x36)
    types = [struct.unpack_from(f"{endian}I", data, phoff + index * phentsize)[0] for index in range(phnum)]
    return _PT_INTERP not in types


@pytest.fixture(scope="module")
def run(sdk_python, tmp_path_factory) -> tuple[dict, fake_openai.FakeOpenAI]:
    folder = tmp_path_factory.mktemp("turns")
    workspace = folder / "work"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("forty-two\n")

    def respond(body: dict) -> fake_openai.Text | fake_openai.ToolCall:
        if any("done" in text for text in _user_texts(body)[-1:]):
            return fake_openai.Text("done")
        ran = sum(message.get("role") == "tool" for message in body.get("messages") or [])
        if ran and any("STALL" in text for text in _user_texts(body)[-1:]):
            time.sleep(_STALL_S)
            return fake_openai.Text("late")
        if ran == len(TOOLS):
            return fake_openai.Text("It printed 42.")
        offered = [tool["function"] for tool in body.get("tools") or [] if TOOLS[ran] in tool["function"]["name"]]
        if len(offered) != 1:
            names = [tool["function"]["name"] for tool in body.get("tools") or []]
            return fake_openai.Text(f"no single tool named for {TOOLS[ran]!r} among {names}")
        return fake_openai.ToolCall(offered[0]["name"], _arguments(offered[0].get("parameters") or {}, workspace,
                                                                   TOOLS[ran]))

    driver = folder / "driver.py"
    driver.write_text(DRIVER)
    with fake_openai.FakeOpenAI(respond=respond) as endpoint:
        env = {**{key: value for key, value in os.environ.items() if not key.startswith(("LITELLM_", "AGENTENV_"))},
               "LITELLM_BASE_URL": f"{endpoint.url}/v1", "LITELLM_API_KEY": "sk-fake", "HOME": str(folder),
               "ANTIGRAVITY_HOME": str(folder / ".antigravity"), "PYTHONPATH": PYTHONPATH}
        out = subprocess.run([str(sdk_python), "-u", str(driver), str(workspace), *ENV_TIMES, MOVED_TO],
                             cwd=folder, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1]), endpoint


def test_the_harness_is_a_static_binary_so_it_keeps_real_time(run):
    harness = Path(run[0]["harness"])

    assert harness.is_file() and _is_static(harness), (
        f"the SDK's harness {harness} loads through the dynamic loader, so the image's libfaketime would put it on "
        "the env's clock, and its conversation store, which orders turns by time, with it")


def test_the_sdks_tools_are_the_ones_a_task_selects_and_the_three_it_cant(run):
    assert set(run[0]["tools"]) == {*antigravity_bridge.AntigravityTool.__args__, "ask_question", "generate_image",
                                    "finish"}, "the SDK's tools changed: record them in AntigravityTool"


def test_on_an_armed_clock_the_model_is_told_the_env_time_and_no_other(run):
    result, endpoint = run
    bodies = [seen.body for seen in endpoint.completions()]
    stalled = [body for body in bodies if any("STALL" in text for text in _user_texts(body))]
    first = [body for body in bodies if len(_user_texts(body)) == 1 and body not in stalled]
    second = [body for body in bodies if len(_user_texts(body)) >= 2]

    assert first and second, [len(_user_texts(body)) for body in bodies]
    assert all(f"The current local time is: {ENV_TIMES[0]}." in _user_texts(body)[0] for body in first), (
        "the harness no longer opens a turn's user message with the line antigravity_date.LOCAL_TIME finds")
    for body in second:
        earlier, latest = _user_texts(body)[0], _user_texts(body)[-1]
        assert f"The current local time is: {ENV_TIMES[0]}." in earlier
        assert f"The current local time is: {ENV_TIMES[1]}." in latest
    assert stalled and all(f"The current local time is: {ENV_TIMES[2]}." in _user_texts(body)[0] for body in stalled)
    machine = _machine_dates()
    for body in bodies:
        written = json.dumps(body)
        found = [written[max(0, at - 160):at + 60] for date in machine if (at := written.find(date)) >= 0]
        assert not found, f"the harness wrote this machine's date where the relay doesn't rewrite it: {found}"


def test_each_tool_result_is_tied_to_the_call_the_sdk_streamed(run):
    first = run[0]["turns"][0]

    assert not first["error"], first["error"]
    calls = [event for event in first["events"] if event.get("type") == "ToolCall"]
    results = [event for event in first["events"] if event.get("kind") == "tool_result"]
    assert len(calls) == len(results) == first["tool_calls"] == len(TOOLS), first["events"]
    assert [result["call_ref"] for result in results] == [call["id"] for call in calls], json.dumps(first["events"])
    assert [result["name"] for result in results] == ["run_command", "list_skills", "view_file"], results
    assert "42" in json.dumps(results[0])
    assert first["response"] == "It printed 42."


def test_the_turns_usage_is_every_model_call_the_harness_made_with_the_endpoints_key(run):
    result, endpoint = run
    completions = endpoint.completions()
    usage = [turn["usage"] for turn in result["turns"][:2]]
    completions = [seen for seen in completions if not any("STALL" in text for text in _user_texts(seen.body))]

    assert sum(turn["provider_details"]["model_calls"] for turn in usage) == len(completions)
    assert sum(turn["input_tokens"] for turn in usage) == fake_openai.USAGE["prompt_tokens"] * len(completions)
    assert {seen.headers.get("authorization") for seen in completions} == {"Bearer sk-fake"}
    assert result["turns"][1]["session_id"] == result["turns"][0]["session_id"]
    assert result["turns"][1]["response"] == "done"


def test_a_first_turn_past_its_time_leaves_the_conversation_the_sdk_started_for_the_next(run):
    stalled = run[0]["turns"][2]

    assert stalled["error_code"] == "timeout", stalled
    assert stalled["session_id"] and stalled["stored"], stalled
    assert stalled["session_id"] != run[0]["turns"][0]["session_id"]


def test_a_date_change_during_a_turn_is_noted_on_the_command_result_after_it_in_that_turn_and_the_next(run):
    _, endpoint = run
    note = antigravity_date.DATE_CHANGED.format(MOVED_TO)
    resent = [body for body in (seen.body for seen in endpoint.completions())
              if f"The current local time is: {ENV_TIMES[0]}." in (_user_texts(body) or [""])[0]
              and any(message.get("role") == "tool" for message in body["messages"])]

    assert len({len(_user_texts(body)) for body in resent}) == 2, "the first turn's result was never sent again"
    for body in resent:
        results = [message["content"] for message in body["messages"] if message.get("role") == "tool"]
        assert results[0].endswith(note), results[0][-300:]
        assert not any("The date has changed" in result for result in results[1:])
