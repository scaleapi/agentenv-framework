"""computer_exec against a computer at its driver's boundary (``tst.unit.computer_home``): a command run in its home as
given or through ``sh -c``, in a folder and with variables of its own, its run recorded; a failure or a timeout failing
the step unless ``check`` is off; a ``background`` command started and left running, and failing when its program
isn't there or it exits non-zero right after starting, a string command's included; a multi env's computer found by
name; and the configs it refuses."""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

import pytest

from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.multi_env import MultiEnv, WebsiteBrowser
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.computer import computer_exec
from agent_env.task_step.task_steps.computer.computer_exec import RECORDS_KEY, ComputerExecTaskStep
from tst.unit.computer_home import TOKEN, Computer


@pytest.fixture
def computer(tmp_path, monkeypatch):
    return Computer(tmp_path).install(monkeypatch)


def _context(*deployed) -> TaskStepContext:
    return TaskStepContext(deployed_agents=[], deployed_envs=list(deployed), metadata={})


def _step(command, **fields) -> ComputerExecTaskStep:
    return ComputerExecTaskStep(id="exec", version=None, env_id=fields.pop("env_id", "desk"), command=command, **fields)


@pytest.mark.asyncio
async def test_a_command_runs_in_the_home_as_given_with_the_computers_token_and_is_recorded(computer):
    ctx = await _step(["sh", "-c", "pwd; echo out; echo err >&2"]).execute(_context(computer.deployed()))

    record = ctx.metadata[RECORDS_KEY]["exec"]
    assert record["exit_code"] == 0 and not record["timed_out"]
    assert record["stdout"] == f"{computer.home}\nout\n" and record["stderr"] == "err\n"
    assert computer.headers == [{"Authorization": f"Bearer {TOKEN}"}]
    _, params, _ = computer.calls[0]
    assert params["argv"] == ["sh", "-c", "pwd; echo out; echo err >&2"] and params["timeout_s"] == 120.0


@pytest.mark.asyncio
async def test_a_string_runs_through_sh_in_its_folder_with_its_variables(computer):
    (computer.home / "work").mkdir()

    ctx = await _step('echo "$GREETING" > made.txt', cwd="work", env={"GREETING": "hi"},
                      timeout_s=5).execute(_context(computer.deployed()))

    assert (computer.home / "work" / "made.txt").read_text() == "hi\n"
    assert ctx.metadata[RECORDS_KEY]["exec"]["argv"] == ["sh", "-c", 'echo "$GREETING" > made.txt']
    _, params, _ = computer.calls[0]
    assert params["cwd"] == "work" and params["env"] == {"GREETING": "hi"} and params["timeout_s"] == 5


@pytest.mark.asyncio
async def test_a_command_that_fails_fails_the_step_with_the_end_of_its_stderr_and_is_still_recorded(computer):
    ctx = _context(computer.deployed())

    with pytest.raises(RuntimeError, match=r"exited 3 in env 'desk': no such thing"):
        await _step("echo 'no such thing' >&2; exit 3").execute(ctx)

    assert ctx.metadata[RECORDS_KEY]["exec"]["exit_code"] == 3


@pytest.mark.asyncio
async def test_with_check_off_a_failure_is_recorded_and_the_step_goes_on(computer):
    ctx = await _step("exit 4", check=False).execute(_context(computer.deployed()))

    assert ctx.metadata[RECORDS_KEY]["exec"]["exit_code"] == 4


@pytest.mark.asyncio
async def test_a_command_that_times_out_fails_the_step(computer):
    with pytest.raises(RuntimeError, match=r"timed out after 0.2 s"):
        await _step("sleep 5", timeout_s=0.2).execute(_context(computer.deployed()))


@pytest.fixture
def short_start(monkeypatch):
    monkeypatch.setattr(computer_exec, "START_WINDOW_S", 0.3)


@pytest.mark.asyncio
async def test_a_background_command_is_started_and_left_running(computer, short_start):
    started = time.monotonic()

    ctx = await _step("sleep 3; echo done > late.txt", background=True).execute(_context(computer.deployed()))

    assert time.monotonic() - started < 2.5
    assert not (computer.home / "late.txt").exists()
    assert ctx.metadata[RECORDS_KEY]["exec"]["argv"][3:] == ["computer_exec", "sh", "-c", "sleep 3; echo done > late.txt"]
    for _ in range(60):
        if (computer.home / "late.txt").exists():
            break
        time.sleep(0.1)
    assert (computer.home / "late.txt").read_text() == "done\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("command, exit_code, stderr", [
    ("thunarr-xyz Desktop", 127, "not found"),
    (["sh", "-c", "echo 'unknown option --bad' >&2; exit 2"], 2, "unknown option --bad"),
])
async def test_a_background_app_that_exits_non_zero_right_after_starting_fails_with_its_stderr(
        computer, short_start, command, exit_code, stderr):
    ctx = _context(computer.deployed())

    with pytest.raises(RuntimeError, match=rf"(?s)^computer_exec 'exec': sh -c .{{0,80}} exited {exit_code} in env "
                                           rf"'desk': .*right after starting.*{stderr}"):
        await _step(command, background=True).execute(ctx)

    assert ctx.metadata[RECORDS_KEY]["exec"]["exit_code"] == exit_code


@pytest.mark.asyncio
async def test_a_background_app_that_hands_off_and_exits_0_has_started(computer, short_start):
    ctx = await _step("echo handed off >&2; exit 0", background=True).execute(_context(computer.deployed()))
    assert ctx.metadata[RECORDS_KEY]["exec"]["exit_code"] == 0


@pytest.mark.parametrize("command", [["no-such-app-xyz", "a.txt"], ["./missing.sh"]])
@pytest.mark.asyncio
async def test_a_background_command_whose_program_isnt_there_fails_the_step(computer, command):
    ctx = _context(computer.deployed())

    with pytest.raises(RuntimeError, match=r"exited 127 in env 'desk': .*not found or not executable"):
        await _step(command, background=True).execute(ctx)

    ctx = await _step(command, background=True, check=False).execute(ctx)
    assert ctx.metadata[RECORDS_KEY]["exec"]["exit_code"] == 127


@pytest.mark.asyncio
async def test_a_timeout_above_what_the_computer_allows_is_refused_before_anything_runs(computer):
    with pytest.raises(ValueError, match="above the 3600 s"):
        await _step("true", timeout_s=4000).execute(_context(computer.deployed()))
    assert computer.calls == []


def _multi(computer: Computer, *names: str) -> DeployedGatewayEnv:
    return DeployedGatewayEnv(env_id="multi", env_version=1, gateway_url="http://gw", sandbox_id="sb-gw",
                              environment_card={"name": "multi"},
                              child_envs={name: dataclasses.asdict(computer.deployed(env_id=name)) for name in names})


@pytest.mark.asyncio
async def test_a_multi_envs_only_computer_is_found_by_itself_and_one_of_several_by_name(computer):
    ctx = await _step("echo one", env_id="multi").execute(_context(_multi(computer, "desk")))
    assert ctx.metadata[RECORDS_KEY]["exec"]["stdout"] == "one\n"

    ctx = await _step("echo two", env_id="multi", computer="b").execute(_context(_multi(computer, "a", "b")))
    assert ctx.metadata[RECORDS_KEY]["exec"]["stdout"] == "two\n"

    with pytest.raises(RuntimeError, match=r"has 2 computers .*: name one as computer"):
        await _step("true", env_id="multi").execute(_context(_multi(computer, "a", "b")))
    with pytest.raises(RuntimeError, match=r"has no computer named 'c'"):
        await _step("true", env_id="multi", computer="c").execute(_context(_multi(computer, "a", "b")))


@pytest.mark.asyncio
async def test_an_env_with_no_computer_exec_is_refused(computer):
    with pytest.raises(RuntimeError, match="has no computers serving"):
        await _step("true").execute(_context(computer.deployed(card={"name": "desk"})))


def _stored_multi(*computers: str, browser: str | None = None) -> MultiEnv:
    def computer_env(name):
        env = ComputerEnv.__new__(ComputerEnv)
        env.environment_name = name
        return env

    office = MultiEnv.__new__(MultiEnv)
    office.id, office._computers = "office", SimpleNamespace(envs=[computer_env(name) for name in computers])
    office.website_browser = lambda: WebsiteBrowser(computer=computer_env(browser) if browser else None)
    return office


@pytest.mark.parametrize("computers, browser, named, problem", [
    (("desk",), None, None, None),
    ((), "chromium", None, None),
    (("desk",), "chromium", "chromium", None),
    (("desk",), "chromium", None, "has 2 computers: name one as computer, of ['desk', 'chromium']"),
    (("desk",), None, "chromium", "has no computer named 'chromium' (it has ['desk'])"),
    ((), None, None, "has no computers"),
], ids=["its-only-computer", "its-browser-computer-alone", "its-browser-computer-by-name", "two-unnamed",
        "an-unknown-name", "none"])
def test_preflight_finds_a_multi_envs_computers_and_its_browser_computer(monkeypatch, computers, browser, named,
                                                                          problem):
    monkeypatch.setattr("agent_env.task_step.task_steps.computer.computer_exec.stored_env",
                        lambda id: _stored_multi(*computers, browser=browser))
    monkeypatch.setattr(TaskStep, "preflight", lambda self: [])

    problems = _step("true", env_id="office", computer=named).preflight()

    assert problems == ([] if problem is None else [f"step 'exec': env 'office' {problem}"])


@pytest.mark.parametrize("fields, problem", [
    ({"command": []}, "command must be a non-empty list"),
    ({"command": "  "}, "command must not be blank"),
    ({"command": ["ls", 3]}, "command must be a non-empty list"),
    ({"command": "true", "cwd": "/tmp"}, "relative to the session user's home"),
    ({"command": "true", "env": {"A": 1}}, "env maps variable names to string values"),
    ({"command": "true", "timeout_s": 0}, "timeout_s must be a number of seconds above 0"),
    ({"command": "true", "background": "yes"}, "background must be true or false"),
    ({"command": "true", "background": True, "timeout_s": 4}, "timeout_s must be at least 5"),
])
def test_bad_configs_are_named(fields, problem):
    assert any(problem in p for p in _step(**fields).config_problems())


def test_it_round_trips_through_the_registry():
    step = _step(["open", "x.xlsx"], computer="desk", cwd="Desktop", env={"A": "1"}, timeout_s=30, background=True,
                 check=False)
    again = get_task_step_registry()["computer_exec"].from_dict(step.to_dict())
    assert again.to_dict() == step.to_dict()
