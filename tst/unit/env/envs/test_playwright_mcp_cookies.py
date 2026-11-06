"""browser/v1's cookies in the Playwright MCP browser, in a website browser's shapes: cookies/set keeps the cookies in
the storage state file every context Playwright MCP opens starts from and sets them in each context of a running
browser, cookies/delete removes those with all of name, domain and path given, or all of them, and neither answer, nor
an error, repeats a cookie's value."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from pathlib import Path

import httpx
import pytest

pytest.importorskip("av")

from agentenv_protocol import client as protocol_client  # noqa: E402

from agent_env.env.envs import playwright_mcp  # noqa: E402
from agent_env.env.envs.playwright_mcp import recorder  # noqa: E402
from agent_env.env.gateway.constants import GATEWAY_EXTENSIONS  # noqa: E402
from agent_env.env.gateway.rest_proxy import rewrite_child_card  # noqa: E402

BROWSER_EXTENSION_URI = recorder.BROWSER_EXTENSION_URI
SET, DELETE = "/agentenv/ext/browser/cookies/set", "/agentenv/ext/browser/cookies/delete"
SID = {"name": "sid", "value": "s3cret", "domain": "github.com", "secure": True, "httpOnly": True,
       "expires": 1_900_000_000}
OTHER = {"name": "pref", "value": "dark", "domain": ".example.com"}


class _DevTools:
    """A browser's DevTools with a default context and two of Playwright's, each holding its cookies."""

    def __init__(self, refuse: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.jars: dict = {None: [], "ctx-1": [], "ctx-2": []}
        self.refuse = refuse

    async def call(self, method, params=None, session=None):
        await asyncio.sleep(0)
        params = dict(params or {})
        self.calls.append((method, params))
        jar = self.jars[params.pop("browserContextId", None)]
        if method == "Target.getBrowserContexts":
            return {"browserContextIds": ["ctx-1", "ctx-2"]}
        if self.refuse:
            raise recorder.CdpError("Invalid cookie fields: s3cret")
        if method == "Storage.setCookies":
            jar += [{**c, "session": "expires" not in c, "size": 1} for c in params["cookies"]]
        elif method == "Storage.clearCookies":
            jar.clear()
        elif method == "Storage.getCookies":
            return {"cookies": list(jar)}
        return {}


def _recorder(tmp_path: Path, devtools: _DevTools | None) -> recorder.BrowserRecorder:
    class _Connection:
        async def __aenter__(self):
            return devtools

        async def __aexit__(self, *exc):
            return False

    return recorder.BrowserRecorder(find=lambda: None if devtools is None else 9222,
                                    connect=lambda port: _Connection(),
                                    storage_state_path=tmp_path / "storage-state.json")


async def _post(app, path: str, body: dict) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://recorder") as http:
        return await http.post(path, json=body)


@pytest.mark.asyncio
async def test_cookies_go_into_the_storage_state_file_before_the_browser_runs(tmp_path):
    app = recorder.create_app(_recorder(tmp_path, None))
    answer = await _post(app, SET, {"cookies": [SID, OTHER]})
    assert answer.json() == {"set": 2, "open_browser": False} and "s3cret" not in answer.text
    path = tmp_path / "storage-state.json"
    state = json.loads(path.read_text())
    assert state["origins"] == []
    assert state["cookies"] == [
        {"name": "sid", "value": "s3cret", "domain": "github.com", "path": "/", "expires": 1_900_000_000,
         "httpOnly": True, "secure": True, "sameSite": "Lax"},
        {"name": "pref", "value": "dark", "domain": ".example.com", "path": "/", "expires": -1, "httpOnly": False,
         "secure": False, "sameSite": "Lax"}]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    await _post(app, SET, {"cookies": [{**SID, "value": "newer"}]})
    assert [c["value"] for c in json.loads(path.read_text())["cookies"]] == ["newer", "dark"]


@pytest.mark.asyncio
async def test_a_cookie_at_a_url_or_with_a_max_age_is_placed_and_timed_as_a_browser_would(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder.time, "time", lambda: 1_000.0)
    app = recorder.create_app(_recorder(tmp_path, None))
    await _post(app, SET, {"cookies": [{"name": "a", "value": "1", "url": "https://shop.example/cart"},
                                       {"name": "b", "value": "2", "domain": "shop.example", "max_age": 60}]})
    assert [(c["domain"], c["path"], c["secure"], c["expires"])
            for c in json.loads((tmp_path / "storage-state.json").read_text())["cookies"]] == [
        ("shop.example", "/", True, -1), ("shop.example", "/", False, 1_060.0)]


@pytest.mark.asyncio
async def test_cookies_are_set_in_every_context_of_a_running_browser(tmp_path):
    devtools = _DevTools()
    app = recorder.create_app(_recorder(tmp_path, devtools))
    assert (await _post(app, SET, {"cookies": [SID]})).json() == {"set": 1, "open_browser": True}
    for jar in devtools.jars.values():
        assert jar == [{"name": "sid", "value": "s3cret", "domain": "github.com", "path": "/", "secure": True,
                        "httpOnly": True, "sameSite": "Lax", "expires": 1_900_000_000, "session": False, "size": 1}]


@pytest.mark.asyncio
async def test_deleting_a_domains_cookies_keeps_the_others_and_deleting_all_keeps_none(tmp_path):
    devtools = _DevTools()
    app = recorder.create_app(_recorder(tmp_path, devtools))
    await _post(app, SET, {"cookies": [SID, OTHER]})
    assert (await _post(app, DELETE, {"domain": "github.com"})).json() == {"deleted": 1, "open_browser": True}
    assert [c["name"] for c in json.loads((tmp_path / "storage-state.json").read_text())["cookies"]] == ["pref"]
    for jar in devtools.jars.values():
        assert [(c["name"], c.get("expires")) for c in jar] == [("pref", None)]
    assert (await _post(app, DELETE, {})).json() == {"deleted": 1, "open_browser": True}
    assert json.loads((tmp_path / "storage-state.json").read_text())["cookies"] == []
    assert all(jar == [] for jar in devtools.jars.values())


@pytest.mark.asyncio
async def test_concurrent_changes_never_restore_what_another_removed(tmp_path):
    devtools = _DevTools()
    app = recorder.create_app(_recorder(tmp_path, devtools))
    await _post(app, SET, {"cookies": [SID, OTHER]})
    await asyncio.gather(_post(app, DELETE, {"domain": "github.com"}), _post(app, DELETE, {"domain": ".example.com"}))
    assert json.loads((tmp_path / "storage-state.json").read_text())["cookies"] == []
    assert all(jar == [] for jar in devtools.jars.values())


@pytest.mark.asyncio
async def test_a_bad_cookie_or_a_browser_that_refuses_answers_without_the_value(tmp_path):
    app = recorder.create_app(_recorder(tmp_path, _DevTools(refuse=True)))
    bad = await _post(app, SET, {"cookies": [{"name": "sid", "value": "s3cret"}]})
    assert bad.status_code == 400 and "names neither a url nor a domain" in bad.text and "s3cret" not in bad.text
    refused = await _post(app, SET, {"cookies": [SID]})
    assert refused.status_code == 502 and refused.json()["error"]["code"] == "browser_unreachable"
    assert "s3cret" not in refused.text
    assert json.loads((tmp_path / "storage-state.json").read_text())["cookies"][0]["name"] == "sid"


@pytest.mark.asyncio
async def test_the_gateway_nests_browser_v1_under_the_browsers_child_env(tmp_path):
    app = recorder.create_app(_recorder(tmp_path, None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://recorder") as http:
        served = (await http.get("/.well-known/agent-env.json")).json()
    composed = {"name": "env", "capabilities": {"extensions": GATEWAY_EXTENSIONS},
                "children_environments": [rewrite_child_card("mcp-playwright-mcp", served, GATEWAY_EXTENSIONS)]}
    child = protocol_client.find_child(composed, playwright_mcp.PLAYWRIGHT_MCP_ENVIRONMENT_NAME)
    methods = protocol_client.find_extension(child, BROWSER_EXTENSION_URI)["params"]["methods"]
    assert {op: m["endpoint"] for op, m in methods.items()} == {
        "cookies/set": "/svc/mcp-playwright-mcp/agentenv/ext/browser/cookies/set",
        "cookies/delete": "/svc/mcp-playwright-mcp/agentenv/ext/browser/cookies/delete"}


def test_playwright_mcp_starts_each_context_from_the_file_the_recorder_writes():
    entrypoint = (Path(playwright_mcp.__file__).parent / "entrypoint.sh").read_text()
    default = re.search(r'PLAYWRIGHT_STORAGE_STATE="\$\{PLAYWRIGHT_STORAGE_STATE:-([^}]+)\}"', entrypoint).group(1)
    assert default == recorder.DEFAULT_STORAGE_STATE
    assert '--storage-state "${PLAYWRIGHT_STORAGE_STATE}"' in entrypoint
