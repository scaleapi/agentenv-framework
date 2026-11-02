"""A computer app's data plane, served by its own server. An app that declares only paths, none inside another, gets
the defaults: a load writes files under them, through no link out of them, an export returns them, and a reset puts back
the copy its server kept before it served; its card declares a file tree. Hooks override paths, and read the parts they
get with part_bytes. Around a load and a reset the app is stopped and relaunched, a failed load is followed by a reset,
and data calls take turns. A call past data_timeout_s fails then and is cancelled, but a sync hook runs in a thread,
which its reset and relaunch, and the next call, wait for. The app serves its skill file."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import time

import pytest

pytest.importorskip("mcp.server.fastmcp")

import httpx  # noqa: E402

from agentenv_protocol import add_data, client, reset_data  # noqa: E402
from agentenv_protocol.computer import ComputerAppEnvironment, Launch, linux, part_bytes  # noqa: E402
from agentenv_protocol.computer import app, app_runtime, session  # noqa: E402
from agentenv_protocol.computer.app import COMPUTER_APP  # noqa: E402
from agentenv_protocol.computer.platform import Platform  # noqa: E402
from agentenv_protocol.types import (DATA_OBJECTS_EXTENSION_URI, INTAKE_EXTENSION_URI, RPC_PATH,  # noqa: E402
                                     SKILL_EXTENSION_URI, WELL_KNOWN_PATH)

MAIL = ["Mail", ".config/mail/accounts.json"]


@linux(paths=MAIL, launch=Launch(command=["mail"]))
class PathsMail(ComputerAppEnvironment):
    skill = "SKILL.md"


@linux(paths=MAIL, launch=Launch(command=["mail"]))
class HookedMail(ComputerAppEnvironment):
    loaded: dict
    resets = 0

    @add_data
    async def load(self, parts: list) -> None:
        if any(part.metadata[DATA_OBJECTS_EXTENSION_URI]["path"] == "bad" for part in parts):
            raise RuntimeError("the seed is bad")
        self.loaded = {part.metadata[DATA_OBJECTS_EXTENSION_URI]["path"]: await part_bytes(part) for part in parts}

    @reset_data
    def wipe(self) -> None:
        type(self).resets += 1


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The session user's home, the app's folder as the current directory, on Linux, the app named mail."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "folder").mkdir()
    monkeypatch.chdir(tmp_path / "folder")
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux"))
    monkeypatch.setenv("ENVIRONMENT_NAME", "mail")
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    return tmp_path / "home"


@pytest.fixture
def desktop(monkeypatch):
    """Each launch a new pid, alive until stopped; what's launched and stopped, in order."""
    events, running = [], set()

    async def launch(argv, cwd, env):
        pid = 4242 + len(events)
        running.add(pid)
        events.append(("launch", pid))
        return pid

    async def stop(pid, timeout_s):
        running.discard(pid)
        events.append(("stop", pid))

    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "stop", stop)
    monkeypatch.setattr(session, "alive", lambda pid: pid in running)
    return events


@pytest.mark.asyncio
async def test_an_app_with_only_paths_loads_exports_and_resets_them_as_installed(home):
    (home / "Mail").mkdir()
    (home / "Mail/inbox.txt").write_text("installed")
    mail = PathsMail()
    mail._computer_app_data_plane().keep_initial()

    async with _http(mail) as http:
        card = (await http.get(WELL_KNOWN_PATH)).json()
        await _rpc(http, "data/add", {"parts": [_inline("Mail/new.txt", b"new"), _inline("Mail/inbox.txt", b"loaded"),
                                                _inline(".config/mail/accounts.json", b"{}")]})
        loaded = _files(await _rpc(http, "data/get", {}))
        await _rpc(http, "data/reset", {})
        reset = _files(await _rpc(http, "data/get", {}))

    assert sorted(card["capabilities"]["operations"]) == ["data/add", "data/get", "data/reset"]
    assert client.extension_params(card, COMPUTER_APP)["paths"] == MAIL
    assert client.find_extension(card, INTAKE_EXTENSION_URI) is not None
    assert loaded == {"Mail/inbox.txt": b"loaded", "Mail/new.txt": b"new", ".config/mail/accounts.json": b"{}"}
    assert reset == {"Mail/inbox.txt": b"installed"}


@pytest.mark.asyncio
async def test_a_file_under_none_of_its_paths_or_through_a_link_out_of_them_is_refused_and_nothing_is_written(home):
    (home / "Documents").mkdir()
    (home / "Mail").mkdir()
    os.symlink(home / "Documents", home / "Mail" / "shared")
    mail = PathsMail()
    mail._computer_app_data_plane().keep_initial()

    async with _http(mail) as http:
        outside = await http.post(RPC_PATH, json=_call("data/add", {"parts": [_inline("Mail/a.txt", b"a"),
                                                                             _inline("Documents/b.txt", b"b")]}))
        linked = await http.post(RPC_PATH, json=_call("data/add", {"parts": [_inline("Mail/a.txt", b"a"),
                                                                            _inline("Mail/shared/b.txt", b"b")]}))

    assert "'Documents/b.txt' is under none of this app's paths" in outside.json()["error"]["message"]
    assert "'Mail/shared/b.txt' lead out of this app's paths through a link" in linked.json()["error"]["message"]
    assert sorted(p.name for p in (home / "Mail").iterdir()) == ["shared"] and not any((home / "Documents").iterdir())


@pytest.mark.parametrize("paths, problem", [
    (["/etc/mail"], "paths are normalized paths relative to the session user's home"),
    (["Mail", "Mail/inbox"], "an app's paths each hold files of their own, but 'Mail' and 'Mail/inbox' overlap"),
    (["Mail", "Mail"], "but 'Mail' and 'Mail' overlap"),
], ids=["absolute", "nested", "twice"])
def test_paths_that_arent_relative_or_overlap_are_refused_where_theyre_declared(paths, problem):
    with pytest.raises(ValueError, match=problem):
        linux(paths=paths)


@pytest.mark.asyncio
async def test_hooks_override_its_paths_and_read_each_part_they_get_with_part_bytes(home):
    mail = HookedMail()

    async with _http(mail) as http:
        card = (await http.get(WELL_KNOWN_PATH)).json()
        await _rpc(http, "data/add", {"parts": [_inline("Mail/a.txt", b"a")]})

    assert mail.loaded == {"Mail/a.txt": b"a"} and not (home / "Mail").exists()
    assert sorted(card["capabilities"]["operations"]) == ["data/add", "data/reset"]
    assert client.find_extension(card, INTAKE_EXTENSION_URI) is None
    assert HookedMail.declaration().data == ["data/add", "data/reset"]


@pytest.mark.asyncio
async def test_a_load_and_a_reset_stop_the_running_app_and_relaunch_it_and_an_export_leaves_it_be(home, desktop):
    mail = PathsMail()
    mail._computer_app_data_plane().keep_initial()
    await mail._computer_app.launch_app()

    async with _http(mail) as http:
        await _rpc(http, "data/add", {"parts": [_inline("Mail/a.txt", b"a")]})
        await _rpc(http, "data/get", {})
        await _rpc(http, "data/reset", {})

    assert desktop == [("launch", 4242), ("stop", 4242), ("launch", 4244), ("stop", 4244), ("launch", 4246)]


@pytest.mark.asyncio
async def test_an_app_that_isnt_running_is_left_stopped(home, desktop):
    mail = PathsMail()
    mail._computer_app_data_plane().keep_initial()

    async with _http(mail) as http:
        await _rpc(http, "data/add", {"parts": [_inline("Mail/a.txt", b"a")]})

    assert desktop == []


@pytest.mark.asyncio
async def test_a_failed_load_is_followed_by_a_reset_and_the_app_is_relaunched(home, desktop):
    mail, HookedMail.resets = HookedMail(), 0
    await mail._computer_app.launch_app()

    async with _http(mail) as http:
        answer = await http.post(RPC_PATH, json=_call("data/add", {"parts": [_inline("bad", b"x")]}))

    assert answer.json()["error"]["message"] == "the seed is bad"
    assert HookedMail.resets == 1 and desktop == [("launch", 4242), ("stop", 4242), ("launch", 4244)]


@pytest.mark.asyncio
async def test_an_async_call_past_its_timeout_fails_saying_so_and_is_cancelled_so_the_next_takes_its_turn(
        home, desktop):
    @linux(launch=Launch(command=["slow"]))
    class Slow(ComputerAppEnvironment):
        data_timeout_s = 0.2

        @add_data
        async def load(self, parts: list) -> None:
            if len(parts) == 2:
                await asyncio.Event().wait()

    async with _http(Slow()) as http:
        answer = await http.post(RPC_PATH, json=_call("data/add", {"parts": [_inline("a.txt", b"a"),
                                                                            _inline("b.txt", b"b")]}))
        await _rpc(http, "data/add", {"parts": [_inline("c.txt", b"c")]})

    assert answer.json()["error"]["message"] == "add_data ran past the app's data_timeout_s, 0.2 s"


@pytest.mark.asyncio
async def test_calls_take_turns_and_one_past_its_timeout_fails_then_while_what_follows_waits_for_its_thread(
        home, desktop):
    @linux(launch=Launch(command=["slow"]))
    class Slow(ComputerAppEnvironment):
        data_timeout_s = 1.0

        @add_data
        def load(self, parts: list) -> None:
            desktop.append(("load", len(parts)))
            time.sleep(1.4 if len(parts) == 2 else 0)
            desktop.append(("loaded", len(parts)))

        @reset_data
        def wipe(self) -> None:
            desktop.append(("reset",))

    slow = Slow()
    await slow._computer_app.launch_app()

    async with _http(slow) as http:
        late = await http.post(RPC_PATH, json=_call("data/add", {"parts": [_inline("a", b"a"), _inline("b", b"b")]}))
        desktop.append(("answered",))
        card = await http.get(WELL_KNOWN_PATH)
        await _rpc(http, "data/add", {"parts": [_inline("c", b"c")]})

    assert late.json()["error"]["message"] == "add_data ran past the app's data_timeout_s, 1 s"
    assert card.status_code == 200
    assert desktop == [("launch", 4242), ("stop", 4242), ("load", 2), ("answered",), ("loaded", 2), ("reset",),
                       ("launch", 4248), ("stop", 4248), ("load", 1), ("loaded", 1), ("launch", 4252)]


@pytest.mark.asyncio
async def test_it_serves_its_skill_file_and_no_skill_without_one(home, tmp_path):
    (tmp_path / "folder" / "SKILL.md").write_text("---\ndescription: Reading and sending mail.\n---\n\n# Mail\nUse it.\n")

    async with _http(PathsMail()) as http:
        card = (await http.get(WELL_KNOWN_PATH)).json()
        skill = (await http.get(client.find_extension_method(card, SKILL_EXTENSION_URI, "get")["endpoint"],
                                params={"name": "mail"})).json()
    async with _http(HookedMail()) as http:
        unskilled = (await http.get(WELL_KNOWN_PATH)).json()

    assert skill["description"] == "Reading and sending mail." and skill["skill_md"].endswith("# Mail\nUse it.\n")
    assert client.find_extension(unskilled, SKILL_EXTENSION_URI) is None


@contextlib.asynccontextmanager
async def _http(served: ComputerAppEnvironment):
    asgi = served.create_app().streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://t") as http:
        yield http


def _call(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


async def _rpc(http: httpx.AsyncClient, method: str, params: dict) -> dict:
    answer = (await http.post(RPC_PATH, json=_call(method, params))).json()
    assert "error" not in answer, answer
    return answer["result"]


def _inline(path: str, data: bytes) -> dict:
    return {"kind": "file", "file": {"bytes": base64.b64encode(data).decode(), "name": path.rsplit("/", 1)[-1]},
            "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path}}}


def _files(result: dict) -> dict[str, bytes]:
    return {part["metadata"][DATA_OBJECTS_EXTENSION_URI]["path"]: base64.b64decode(part["file"]["bytes"])
            for part in result["parts"]}
