"""computer_exec against a computer at its driver's boundary (``tst.unit.computer_home``): a command run in its home as
given or through ``sh -c``, in a folder and with variables of its own, its run recorded; a failure or a timeout failing
the step unless ``check`` is off; a ``background`` command started and left running, and failing when its program
isn't there or it exits non-zero right after starting, a string command's included; a ``system`` command run as root
through the computer's sandbox, in the session user's folder; a multi env's computer found by name; and the configs it
refuses."""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import time
from types import SimpleNamespace

import pytest
from agentenv_protocol.computer import COMPUTER

from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.multi_env import MultiEnv, WebsiteBrowser
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.computer import computer_exec
from agent_env.task_step.task_steps.computer.computer_exec import RECORDS_KEY, ComputerExecTaskStep
from tst.unit.computer_home import CARD, TOKEN, Computer


@pytest.fixture
def computer(tmp_path, monkeypatch):
    return Computer(tmp_path).install(monkeypatch)


def _context(*deployed) -> TaskStepContext:
    return TaskStepContext(deployed_agents=[], deployed_envs=list(deployed), metadata={})


def _step(command, **fields) -> ComputerExecTaskStep:
    return ComputerExecTaskStep(id="exec", version=None, env_id=fields.pop("env_id", "desk"), command=command, **fields)


class _Sandbox:
    """A computer's sandbox whose exec is ``uid``'s, running what it's given on this machine."""

    type = "fake"

    def __init__(self):
        self.uid = "0"
        self.calls: list[list[str]] = []

    async def exec_with_output(self, *argv):
        self.calls.append(list(argv))
        if list(argv) == ["id", "-u"]:
            return 0, f"{self.uid}\n", ""
        done = await asyncio.to_thread(subprocess.run, list(argv), capture_output=True, text=True)
        return done.returncode, done.stdout, done.stderr


@pytest.fixture
def sandbox(monkeypatch):
    held = _Sandbox()

    async def get_sandbox(sandbox_id):
        return held

    monkeypatch.setattr(computer_exec, "build_sandbox_provider",
                        lambda sandbox_type: SimpleNamespace(get_sandbox=get_sandbox))
    return held


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
    assert ctx.metadata[RECORDS_KEY]["exec"]["argv"][3:] == ["computer_exec", "sh", "-c",
                                                             "sleep 3; echo done > late.txt"]
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
async def test_a_system_command_runs_as_root_through_the_sandbox_in_the_session_users_folder_and_home(
        computer, sandbox):
    (computer.home / "work").mkdir()

    ctx = await _step('pwd; echo "$HOME $GREETING"', scope="system", cwd="work", env={"GREETING": "hi"}).execute(
        _context(computer.deployed()))

    record = ctx.metadata[RECORDS_KEY]["exec"]
    assert record["exit_code"] == 0 and record["scope"] == "system"
    assert record["stdout"] == f"{computer.home}/work\n{computer.home} hi\n"
    assert sandbox.calls[-1][4:9] == ["65536", "timeout", "-k", "5", "120"]
    assert [params["argv"] for _, params, _ in computer.calls] == [["pwd"]]


@pytest.mark.asyncio
async def test_a_system_command_that_times_out_or_cant_start_fails_unless_check_is_off_and_is_recorded(
        computer, sandbox):
    with pytest.raises(RuntimeError, match=r"sh -c 'sleep 5' as root timed out after 0.5 s"):
        await _step("sleep 5", scope="system", timeout_s=0.5).execute(_context(computer.deployed()))

    ctx = await _step(["no-such-app-xyz"], scope="system", check=False).execute(_context(computer.deployed()))
    assert ctx.metadata[RECORDS_KEY]["exec"]["exit_code"] == 127
    ctx = await _step("true", scope="system", cwd="missing", check=False).execute(_context(computer.deployed()))
    assert ctx.metadata[RECORDS_KEY]["exec"]["exit_code"] == 125


@pytest.mark.asyncio
async def test_a_system_command_timed_out_when_the_deadline_stopped_or_killed_it_not_when_it_exits_so_itself(
        computer, sandbox, monkeypatch):
    monkeypatch.setattr(computer_exec, "SYSTEM_KILL_AFTER_S", 1)

    ctx = await _step("exit 124", scope="system", check=False).execute(_context(computer.deployed()))
    record = ctx.metadata[RECORDS_KEY]["exec"]
    assert (record["exit_code"], record["timed_out"]) == (124, False)

    # It ignores the stop signal, so timeout kills it, and exits 137 rather than 124.
    ctx = await _step("trap '' TERM; sleep 30", scope="system", timeout_s=0.5, check=False).execute(
        _context(computer.deployed()))
    record = ctx.metadata[RECORDS_KEY]["exec"]
    assert (record["exit_code"], record["timed_out"]) == (None, True)


@pytest.mark.asyncio
async def test_a_system_commands_output_is_capped_in_the_computer_and_any_bytes_decode(computer, sandbox, monkeypatch):
    monkeypatch.setattr(computer_exec, "SYSTEM_OUTPUT_BYTES", 1000)

    ctx = await _step("yes 0123456789 | head -c 55000; printf 'bad \\377\\376 bytes' >&2", scope="system").execute(
        _context(computer.deployed()))

    record = ctx.metadata[RECORDS_KEY]["exec"]
    assert record["truncated"] and len(record["stdout"]) == 1000 and record["stdout"].endswith("0123456789\n")
    assert record["stderr"] == "bad \ufffd\ufffd bytes"


@pytest.mark.asyncio
async def test_a_system_command_is_refused_on_a_disk_computer_and_on_a_sandbox_whose_exec_isnt_roots(
        computer, sandbox):
    with pytest.raises(RuntimeError, match="boots from a disk"):
        await _step("true", scope="system").execute(_context(computer.deployed(form="disk")))

    sandbox.uid = "1000"
    with pytest.raises(RuntimeError, match="as uid 1000; deploy it on a VM sandbox"):
        await _step("true", scope="system").execute(_context(computer.deployed()))


@pytest.mark.asyncio
async def test_on_a_windows_computer_a_string_or_background_command_is_refused_and_a_program_runs(computer):
    windows = {**CARD, "capabilities": {**CARD["capabilities"], "extensions": [
        *CARD["capabilities"]["extensions"], {"uri": COMPUTER, "params": {"platform": {"os": "windows"}}}]}}

    for step in (_step("echo hi"), _step(["notepad.exe"], background=True)):
        with pytest.raises(RuntimeError, match="runs Windows, which has no sh"):
            await step.execute(_context(computer.deployed(card=windows)))
    assert computer.calls == []

    ctx = await _step(["echo", "hi"]).execute(_context(computer.deployed(card=windows)))
    assert ctx.metadata[RECORDS_KEY]["exec"]["stdout"] == "hi\n"


@pytest.mark.asyncio
async def test_a_background_start_still_running_when_its_window_ends_leaves_no_files(computer, tmp_path, monkeypatch):
    monkeypatch.setattr(computer_exec, "START_WINDOW_S", 0.3)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    await _step(["sleep", "3"], background=True, env={"TMPDIR": str(scratch)}).execute(_context(computer.deployed()))

    assert list(scratch.iterdir()) == []


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
    ({"command": "true", "scope": "root"}, "scope is 'session', the session user's, or 'system', root's"),
    ({"command": "true", "scope": "system", "background": True}, "a 'system' command runs to its end"),
])
def test_bad_configs_are_named(fields, problem):
    assert any(problem in p for p in _step(**fields).config_problems())


def test_it_round_trips_through_the_registry():
    step = _step(["open", "x.xlsx"], computer="desk", cwd="Desktop", env={"A": "1"}, timeout_s=30, background=True,
                 check=False)
    again = get_task_step_registry()["computer_exec"].from_dict(step.to_dict())
    assert again.to_dict() == step.to_dict()
    root = _step("apt-get install -y jq", scope="system")
    assert get_task_step_registry()["computer_exec"].from_dict(root.to_dict()).scope == "system"
