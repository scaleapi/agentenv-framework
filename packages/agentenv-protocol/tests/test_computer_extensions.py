"""The framework extensions: computer-display, computer-keyboard and computer-exec with the Linux default, which runs
commands in the session, caps their output, times them out and answers a repeated request_id with the first result."""
from __future__ import annotations

import asyncio
import sys
import time

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeDriver, MinimalDriver, http_client  # noqa: E402

from agentenv_protocol.computer import ComputerSettings, ExecResult, linux, run_in_session, set_display  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the Linux defaults need a POSIX host")


def _run(**body) -> dict:
    return {"argv": ["sh", "-c", "true"], "request_id": "r", **body}


@pytest.mark.asyncio
async def test_display_get_and_set_within_the_declared_modes():
    driver = FakeDriver()
    async with http_client(driver) as http:
        got = (await http.get("/agentenv/ext/computer-display/get")).json()
        changed = await http.post("/agentenv/ext/computer-display/set", json={"width": 640, "height": 400, "scale": 2})
        refused = await http.post("/agentenv/ext/computer-display/set", json={"width": 800, "height": 600})
    assert got == {"width": 1280, "height": 800, "scale": 1.0, "modes": [[1280, 800], [1920, 1080], [640, 400]]}
    assert changed.json()["width"] == 640 and changed.json()["scale"] == 2.0
    assert driver.screen.image.size == (1280, 800)
    assert refused.status_code == 400 and "800×600 is not one of this computer's display modes" in refused.json()["error"]["message"]


@pytest.mark.asyncio
async def test_a_set_display_that_returns_nothing_answers_with_the_display_it_reads():
    class Quiet(FakeDriver):
        @set_display
        async def resize(self, width, height, scale):
            self.screen.resize(width, height, scale)

    async with http_client(Quiet()) as http:
        changed = (await http.post("/agentenv/ext/computer-display/set", json={"width": 1920, "height": 1080})).json()
    assert (changed["width"], changed["height"]) == (1920, 1080)


@pytest.mark.asyncio
async def test_keyboard_settings_reach_the_driver_within_their_ranges():
    driver = FakeDriver()
    async with http_client(driver) as http:
        before = (await http.get("/agentenv/ext/computer-keyboard/get")).json()
        after = (await http.post("/agentenv/ext/computer-keyboard/set", json={"tap_hold_ms": 80, "layout": "us"})).json()
        too_long = await http.post("/agentenv/ext/computer-keyboard/set", json={"max_hold_ms": 600_001})
        layout = await http.post("/agentenv/ext/computer-keyboard/set", json={"layout": "de"})
    assert before == {"tap_hold_ms": 50, "type_delay_ms": 10, "layout": "us", "max_hold_ms": 10_000}
    assert after["tap_hold_ms"] == 80 and driver.settings.tap_hold_ms == 80
    assert too_long.status_code == 400 and too_long.json()["error"]["code"] == "invalid_argument"
    assert layout.status_code == 501 and layout.json()["error"]["code"] == "unsupported"


@pytest.mark.asyncio
async def test_a_command_runs_in_the_session_with_its_env_and_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "work").mkdir()
    async with http_client(FakeDriver()) as http:
        r = await http.post("/agentenv/ext/computer-exec/run", json=_run(
            argv=["sh", "-c", 'echo "$GREETING from $(pwd)"; echo oops >&2; exit 3'], cwd="work",
            env={"GREETING": "hi"}))
    assert r.json() == {"exit_code": 3, "stdout": f"hi from {(tmp_path / 'work').resolve()}\n", "stderr": "oops\n",
                        "truncated": False, "timed_out": False}


@pytest.mark.asyncio
async def test_a_command_past_its_timeout_is_killed_with_its_children():
    async with http_client(FakeDriver()) as http:
        started = time.monotonic()
        r = await http.post("/agentenv/ext/computer-exec/run", json=_run(
            argv=["sh", "-c", "sleep 30 & sleep 30; echo never"], timeout_s=0.3))
    assert r.json()["timed_out"] is True and r.json()["exit_code"] is None and r.json()["stdout"] == ""
    assert time.monotonic() - started < 5


@pytest.mark.asyncio
async def test_output_is_capped_per_stream():
    driver = FakeDriver(settings=ComputerSettings(exec_output_bytes=10))
    async with http_client(driver) as http:
        r = await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=["sh", "-c", "yes | head -c 100000"]))
    assert r.json()["stdout"] == "y\ny\ny\ny\ny\n" and r.json()["truncated"] is True


@pytest.mark.asyncio
async def test_a_background_app_holding_the_output_open_doesnt_hold_the_call():
    async with http_client(FakeDriver()) as http:
        started = time.monotonic()
        r = await http.post("/agentenv/ext/computer-exec/run", json=_run(
            argv=["sh", "-c", "echo started; (sleep 3; echo late) &"]))
        elapsed = time.monotonic() - started
    assert r.json()["exit_code"] == 0 and r.json()["stdout"] == "started\n" and elapsed < 2.5


@pytest.mark.asyncio
async def test_a_repeated_request_id_returns_the_first_result_without_running_again(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    count = ["sh", "-c", "echo x >> runs; wc -l < runs"]
    driver = FakeDriver()
    async with http_client(driver) as http:
        first = (await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=count, request_id="a"))).json()
        again = (await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=count, request_id="a"))).json()
        both = await asyncio.gather(*(http.post("/agentenv/ext/computer-exec/run",
                                                json=_run(argv=["sh", "-c", "sleep 0.2; " + count[2]], request_id="b"))
                                      for _ in range(3)))
        other = (await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=count, request_id="c"))).json()
    assert first == again and first["stdout"].strip() == "1"
    assert len({r.json()["stdout"] for r in both}) == 1 and both[0].json()["stdout"].strip() == "2"
    assert other["stdout"].strip() == "3"


@pytest.mark.asyncio
async def test_a_failed_command_is_not_remembered():
    async with http_client(FakeDriver()) as http:
        missing = await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=["no-such-binary-xyz"]))
        again = await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=["no-such-binary-xyz"]))
    assert missing.status_code == again.status_code == 400
    assert missing.json()["error"]["code"] == "invalid_argument" and "no-such-binary-xyz" in missing.text


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"argv": []}, {"argv": "ls -l"}, {"argv": ["ls", 1]}, {"request_id": ""}, {"env": {"A": 1}},
    {"timeout_s": 0}, {"timeout_s": 3601},
])
async def test_a_bad_command_request_is_refused(body):
    async with http_client(FakeDriver()) as http:
        r = await http.post("/agentenv/ext/computer-exec/run", json=_run(**body))
    assert r.status_code == 400 and r.json()["error"]["code"] in ("invalid_argument", "invalid_params")


@pytest.mark.asyncio
async def test_a_drivers_own_run_in_session_is_capped_and_checked():
    class Own(MinimalDriver):
        output = ExecResult(0, b"a" * 50, b"", False)

        @run_in_session
        async def run(self, argv, cwd, env, timeout_s):
            return self.output

    driver = Own()
    driver.settings = ComputerSettings(exec_output_bytes=8)
    async with http_client(driver) as http:
        capped = (await http.post("/agentenv/ext/computer-exec/run", json=_run())).json()
        driver.output = "not a result"
        broken = await http.post("/agentenv/ext/computer-exec/run", json=_run(request_id="other"))
    assert capped == {"exit_code": 0, "stdout": "aaaaaaaa", "stderr": "", "truncated": True, "timed_out": False}
    assert broken.status_code == 500 and broken.json()["error"]["code"] == "internal"


@pytest.mark.asyncio
async def test_a_background_apps_output_is_drained_after_the_call_returns(monkeypatch):
    monkeypatch.setattr(linux, "DRAIN_S", 0.1)
    async with http_client(FakeDriver()) as http:
        r = await http.post("/agentenv/ext/computer-exec/run", json=_run(
            argv=["sh", "-c", "echo now; (sleep 0.4; echo later) &"]))
        draining = len(linux._draining)

        async def drained():
            while linux._draining:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(drained(), timeout=10)
    assert r.json()["stdout"] == "now\n" and draining >= 1


@pytest.mark.asyncio
async def test_only_the_latest_command_results_are_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agentenv_protocol.computer.runtime.KEEP_EXEC_RESULTS", 1)
    count = ["sh", "-c", "echo x >> runs; wc -l < runs"]
    async with http_client(FakeDriver()) as http:
        for request_id in ("a", "b", "a"):
            last = (await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=count, request_id=request_id))).json()
    assert last["stdout"].strip() == "3", "a's first result was dropped, so a ran again"


@pytest.mark.asyncio
async def test_a_running_command_is_joined_however_many_others_finish_meanwhile(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agentenv_protocol.computer.runtime.KEEP_EXEC_RESULTS", 1)
    slow = _run(argv=["sh", "-c", "sleep 0.6; echo a >> runs"], request_id="a")
    async with http_client(FakeDriver()) as http:
        async def others_then_retry():
            await asyncio.sleep(0.1)
            for request_id in ("b", "c"):
                await http.post("/agentenv/ext/computer-exec/run", json=_run(argv=["true"], request_id=request_id))
            return await http.post("/agentenv/ext/computer-exec/run", json=slow)

        first, retried = await asyncio.gather(http.post("/agentenv/ext/computer-exec/run", json=slow),
                                              others_then_retry())
    assert first.json() == retried.json() and (tmp_path / "runs").read_text() == "a\n"
