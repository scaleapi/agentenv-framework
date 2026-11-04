"""Offline unit tests for the Agent37 sandbox handle: exec and its resends."""

from __future__ import annotations

import ssl

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.agent37 import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.agent37.provider import Agent37SandboxProvider
from agent_env.providers.sandbox_providers.agent37.sandbox import Agent37Error
from tst.unit.providers.sandbox_providers.agent37_fakes import API_KEY, FakeAgent37


@pytest.fixture(autouse=True)
def _no_resend_delays(monkeypatch):
    monkeypatch.setattr(sandbox_module, "RESEND_DELAYS", (0, 0, 0, 0, 0))


@pytest.fixture
def api():
    return FakeAgent37()


@pytest_asyncio.fixture
async def sandbox(api):
    provider = Agent37SandboxProvider(api_key=API_KEY, transport=api.transport())
    return await provider.create_vm(setup_for_gateway=False)


def _result(stdout: str = "", exit_code: int = 0) -> dict:
    return {"exit_code": exit_code, "stdout": stdout, "stderr": "", "truncated": False}


def _answer(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": code}})


@pytest.mark.asyncio
async def test_exec_runs_argv_as_root_with_the_sudo_prefix_dropped(api, sandbox):
    api.exec_answers = [{"exit_code": 3, "stdout": "out\n", "stderr": "err\n", "truncated": False}]
    assert await sandbox.exec_with_output("sudo", "bash", "-c", "echo 'a b'; exit 3") == (3, "out\n", "err\n")
    [body] = api.execs
    assert body["command"] == "bash -c 'echo '\"'\"'a b'\"'\"'; exit 3'"
    assert body["user"] == "root"
    assert len(body["name"]) == 32


@pytest.mark.asyncio
async def test_a_command_whose_answer_is_lost_is_resent_under_the_same_name(api, sandbox):
    api.exec_answers = [httpx.ReadError("reset"), _answer(503, "try_again"), ssl.SSLError(1, "bad record mac"),
                        _answer(504, "try_again"), _result("done\n")]
    assert await sandbox.exec_with_output("sleep", "20") == (0, "done\n", "")
    assert len(api.execs) == 5
    assert len({body["name"] for body in api.execs}) == 1


@pytest.mark.asyncio
async def test_a_command_no_resend_reaches_is_exit_minus_one(api, sandbox):
    api.exec_answers = [httpx.ConnectError("unreachable")] * 6
    exit_code, stdout, stderr = await sandbox.exec_with_output("true")
    assert (exit_code, stdout) == (sandbox_module.UNKNOWN_EXIT_CODE, "")
    assert "ConnectError" in stderr
    assert len(api.execs) == 6


@pytest.mark.asyncio
async def test_exec_script_retries_an_idempotent_script_after_exit_minus_one(api, sandbox):
    api.exec_answers = [httpx.ConnectError("unreachable")] * 6 + [_result("ok\n")]
    assert await sandbox.exec_script("echo ok", max_retries=1) == "ok\n"
    assert len({body["name"] for body in api.execs}) == 2


@pytest.mark.asyncio
async def test_an_exec_refusal_that_is_not_transient_raises(api, sandbox):
    api.exec_answers = [_answer(401, "unauthorized")]
    with pytest.raises(Agent37Error, match="401 unauthorized"):
        await sandbox.exec("true")
    assert len(api.execs) == 1


@pytest.mark.asyncio
async def test_exec_in_a_deleted_instance_raises_the_edge_error(api, sandbox):
    await sandbox.terminate()
    with pytest.raises(Agent37Error, match="404 not_found"):
        await sandbox.exec("true")


@pytest.mark.asyncio
async def test_truncated_output_is_returned_with_a_warning(api, sandbox, caplog):
    api.exec_answers = [{"exit_code": 0, "stdout": "head…tail", "stderr": "", "truncated": True}]
    assert (await sandbox.exec_with_output("cat", "big"))[1] == "head…tail"
    assert "512 KB" in caplog.text
