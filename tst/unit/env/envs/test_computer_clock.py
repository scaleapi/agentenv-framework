"""A computer's clock follower from core's side: its config written through the root channel, root's and the follower's
alone and renamed into place, its URL as the computer's sandbox reaches the gateway, kept in the record without its
token, and written again on the fresh computer a reset boots; each time core waits on the computer's own sync_time, and
a follower that can't follow is logged as a warning naming why, the deploy or reset going on."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from agentenv_protocol.client import ExtensionCallError
from agentenv_protocol.computer.clock import CONFIG

from agent_env.env.env import DeployedSandboxEnv
from agent_env.env.envs.computer.utils import clock


def _sandbox(calls: list, written: dict) -> SimpleNamespace:
    async def exec_with_output(*argv):
        calls.append(argv)
        return 0, "0\n" if argv == ("id", "-u") else "", ""

    async def write_file_from_text(content, path):
        calls.append(("write", path))
        written[path] = json.loads(content)

    return SimpleNamespace(url_from_sandbox=lambda url: url.replace("http://localhost:8123", "http://host.docker.internal:8123"),
                           exec_with_output=exec_with_output, write_file_from_text=write_file_from_text)


@pytest.fixture(autouse=True)
def synced(monkeypatch) -> AsyncMock:
    """The computer's sync_time, which answers once its follower follows."""
    invoke = AsyncMock(return_value={"synced": True})
    monkeypatch.setattr(DeployedSandboxEnv, "invoke", invoke)
    return invoke


def _record(**metadata) -> DeployedSandboxEnv:
    return DeployedSandboxEnv(env_id="desktop", env_version=1, sandbox_id="sb", sandbox_type="local",
                              control_token="token-desktop", metadata={"computer": {"form": "container", **metadata}})


@pytest.mark.asyncio
async def test_the_config_is_written_through_the_root_channel_readable_by_the_follower_alone_and_renamed_into_place(
        synced):
    calls, written, record = [], {}, _record()

    await clock.configure(_sandbox(calls, written), record, "http://localhost:8123/", "desktop")

    (staged, config), = written.items()
    assert staged.startswith(f"{CONFIG}.") and config == {
        "watch_url": "http://host.docker.internal:8123/clock/watch", "name": "desktop", "token": "token-desktop"}
    assert calls == [("id", "-u"), ("mkdir", "-p", "/etc/agentenv/clock"), ("write", staged),
                     ("sh", "-c", f"chown root:agentenv-clock {staged} && chmod 0640 {staged} && mv -f {staged} {CONFIG}")]
    assert record.metadata["computer"]["clock"] == {"watch_url": "http://host.docker.internal:8123/clock/watch",
                                                    "name": "desktop"}
    synced.assert_awaited_once_with("urn:agentenv:clock/v1", "sync_time",
                                    {"env_get_time_url": "http://host.docker.internal:8123/clock/time"}, timeout=30)


@pytest.mark.asyncio
async def test_a_reset_writes_the_kept_config_again_with_the_records_token_and_one_without_writes_nothing(synced):
    calls, written = [], {}
    kept = {"watch_url": "https://gw.example/clock/watch", "name": "desktop"}

    await clock.reconfigure(_sandbox(calls, written), _record(clock=kept))
    await clock.reconfigure(_sandbox(calls, written), _record())

    assert list(written.values()) == [{**kept, "token": "token-desktop"}]
    assert len(calls) == 4 and synced.await_count == 1


@pytest.mark.asyncio
async def test_a_follower_that_cant_follow_is_logged_as_a_warning_naming_why_and_the_reset_goes_on(synced, caplog):
    answer = httpx.Response(503, json={"ok": False, "error": {"code": "target_unavailable", "message": (
        "this computer's clock follower isn't following the gateway: it can't reach the gateway: URLError: refused"),
        "retryable": False}}, request=httpx.Request("POST", "http://desktop/agentenv/ext/sync_time"))
    synced.side_effect = ExtensionCallError(answer)
    written = {}

    with caplog.at_level("WARNING"):
        await clock.reconfigure(_sandbox([], written), _record(clock={"watch_url": "https://gw.example/clock/watch",
                                                                       "name": "desktop"}))

    assert len(written) == 1
    assert ("desktop: its clock follower doesn't follow the gateway at https://gw.example/clock/watch, so its session "
            "reads real time until it does: 503 target_unavailable: this computer's clock follower isn't following "
            "the gateway: it can't reach the gateway: URLError: refused") in caplog.text
    assert "retryable" not in caplog.text


def test_a_computer_follows_the_clock_when_its_card_serves_sync_time():
    card = {"name": "desktop", "capabilities": {"extensions": [
        {"uri": "urn:agentenv:clock/v1", "params": {"methods": {"sync_time": {"method": "POST", "endpoint": "/x"}}}}]}}

    assert clock.follows(DeployedSandboxEnv(env_id="desktop", env_version=1, sandbox_id="sb", environment_card=card))
    assert not clock.follows(DeployedSandboxEnv(env_id="desktop", env_version=1, sandbox_id="sb",
                                                environment_card={"name": "desktop"}))
