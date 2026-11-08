"""The macos-harness driver against a fake harness: its one tool runs a list of command lines in order, the first that
fails stopping the rest, a line that isn't one of its commands running nothing, a read's screenshot returned as an
image, and the harness given none of the driver's environment and no telemetry; its instructions map its shell to the
tool and send the agent to its skill, within the length Claude Code shows; readiness checks the pinned version and
skill."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

import agent_env
from agentenv_protocol.computer import _macos

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "macos-harness" / "driver.py"
FAKE = textwrap.dedent('''\
    import json, os, sys
    from pathlib import Path
    state = Path.home() / "fake"
    with open(state / "calls.jsonl", "a") as out:
        out.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\\n")
    command = sys.argv[1]
    if command == "--version":
        print((state / "version").read_text())
    elif command == "skill":
        sys.stdout.write((state / "skill").read_text())
    elif command == "get_app_state":
        print("Window: \\"BenchLab\\", App: BenchLab.\\n\\t5 button Next\\n")
        print(f"Screenshot (2x1 px): {state / 'shot.jpeg'}")
    elif "slow" in sys.argv:
        (state / "pid.tmp").write_text(str(os.getpid()))
        os.replace(state / "pid.tmp", state / "pid")  # whole, so a reader that sees the file sees the pid
        import time
        time.sleep(60)
    elif "fail" in sys.argv:
        print("no element 99", file=sys.stderr)
        sys.exit(3)
    else:
        print(f"Done {command}.")
''')


class Granted:
    def missing(self) -> list[str]:
        return []


@pytest.fixture
def module(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTENV_CONTROL_TOKEN", "a-deploys-secret")
    monkeypatch.setattr(_macos, "Quartz", Granted)
    spec = importlib.util.spec_from_file_location("macos_harness_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    state = tmp_path / "fake"
    state.mkdir()
    (state / "version").write_text("0.2.1")
    (state / "skill").write_text(loaded.SKILL.read_text())
    (state / "shot.jpeg").write_bytes(b"\xff\xd8jpeg")
    harness = tmp_path / "macos-harness"
    harness.write_text(f"#!{sys.executable}\n{FAKE}")
    harness.chmod(0o755)
    monkeypatch.setattr(loaded, "EXECUTABLE", harness)
    return loaded


def _calls(home: Path) -> list[dict]:
    path = home / "fake" / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.asyncio
async def test_command_lines_run_in_order_until_one_fails_and_a_reads_screenshot_comes_back(module, tmp_path):
    result = await module.MacosHarness().macos_harness(commands=[
        ["get_app_state", "--app", "BenchLab"], ["click", "--app", "BenchLab", "--element_index", "99", "fail"],
        ["list_apps"]])

    text, image, failed = result.content
    assert result.isError and text.text.startswith("$ macos-harness get_app_state --app BenchLab\nWindow:")
    assert image.type == "image" and image.mimeType == "image/jpeg" and image.data == "/9hqcGVn"
    assert failed.text == ("$ macos-harness click --app BenchLab --element_index 99 fail\nno element 99\n(exit 3)")
    assert [call["argv"][0] for call in _calls(tmp_path)] == ["get_app_state", "click"]


@pytest.mark.asyncio
async def test_a_line_that_isnt_one_of_its_commands_runs_nothing(module, tmp_path):
    result = await module.MacosHarness().macos_harness(commands=[["list_apps"], ["install"]])
    helped = await module.MacosHarness().macos_harness(commands=[["help"]])

    assert result.isError and "not ['install']. Nothing ran." in result.content[0].text
    assert not helped.isError and [call["argv"] for call in _calls(tmp_path)] == [["help"]]


@pytest.mark.asyncio
async def test_the_harness_gets_none_of_the_drivers_environment_and_no_telemetry(module, tmp_path):
    result = await module.MacosHarness().macos_harness(commands=[["list_apps"]])

    (call,) = _calls(tmp_path)
    assert not result.isError and result.content[0].text == "$ macos-harness list_apps\nDone list_apps."
    assert call["env"]["DO_NOT_TRACK"] == "1" and "AGENTENV_CONTROL_TOKEN" not in call["env"]
    assert call["env"]["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


@pytest.mark.asyncio
async def test_its_instructions_send_the_agent_to_its_skill_and_readiness_checks_the_version_and_skill(module,
                                                                                                       tmp_path):
    driver = module.MacosHarness()
    instructions = driver.harness_instructions()
    ready = await driver.ready_checks()
    (tmp_path / "fake" / "version").write_text("0.3.0")
    driver._pinned = None
    (old, _) = await driver.ready_checks()
    description, body, files = await driver.bundled_skill()

    assert instructions == module.PREFACE and len(instructions) < 4096 and 'run [["skill"]]' in instructions
    assert "the files the harness names stay on this computer" in instructions
    assert "chain joined with `&&` is a list" in instructions
    assert [(check.name, check.ok) for check in ready] == [("macos-harness", True), ("macOS permissions", True)]
    assert not old.ok and old.detail == "version '0.3.0', not the pinned 0.2.1"
    assert description.startswith("Read and operate the apps on this Mac") and body.startswith("# macOS Harness")
    assert files == {}


@pytest.mark.asyncio
async def test_a_cancelled_call_stops_the_command_it_was_running(module, tmp_path):
    call = asyncio.create_task(module.MacosHarness().macos_harness(commands=[["scroll", "--app", "BenchLab", "slow"]]))
    pid_file = tmp_path / "fake" / "pid"
    for _ in range(500):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
