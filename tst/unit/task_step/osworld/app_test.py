"""The osworld-server app: its card advertises the osworld-guest/v1 extension at the prefix its server forwards to the
guest server under, that forward keeps the method, path, query and body and drops the computer's control token, an
unreachable guest server is a 502, the computer fetches upstream OSWorld's guest server at the package's commit, checked
by sha256, and the start script runs it on loopback without Werkzeug's debugger; the osworld bundle ships the app under
its alias."""
from __future__ import annotations

import importlib.util
import json
import sys
import tomllib

import httpx
import pytest
from agentenv_protocol.agent_env_environment import _EXT_ATTR, _EXT_ROUTE_ATTR
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from agent_env.task_step.task_steps.computer import osworld


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("osworld_server_app", osworld.APP_DIR / "osworld_server_app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["osworld_server_app"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("osworld_server_app", None)


def _forwarding(module, upstream) -> TestClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))

    async def route(request):
        return await module.forward(request, client)

    return TestClient(Starlette(routes=[Route(f"{module.PREFIX}/{{path:path}}", route, methods=module.METHODS)]))


def test_the_env_toml_names_the_app_and_its_class(app_module):
    toml = tomllib.loads((osworld.APP_DIR / "env.toml").read_text())
    assert toml["type"] == "computer_app" and toml["environment_name"] == osworld.APP_NAME
    file, _, cls = toml["module"].partition(":")
    assert (osworld.APP_DIR / file).is_file() and hasattr(app_module, cls)


def test_the_card_advertises_the_guest_extension_at_the_forward_prefix(app_module):
    advertised = getattr(app_module.OSWorldServer.guest, _EXT_ATTR)
    assert advertised.uri == osworld.GUEST_EXTENSION == app_module.GUEST_EXTENSION
    assert advertised.params["endpoint"] == osworld.PROXY_PREFIX == app_module.PREFIX
    assert getattr(app_module.OSWorldServer.guest, _EXT_ROUTE_ATTR) == (osworld.PROXY_PREFIX, "GET")


@pytest.mark.asyncio
async def test_the_guest_extension_says_whether_the_guest_server_is_up(app_module, monkeypatch):
    def refused(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(app_module.socket, "create_connection", refused)
    answer = await app_module.OSWorldServer.guest(None)
    assert answer["ready"] is False and "refused" in answer["detail"]
    assert app_module.OSWorldServer.guest_server(None).ok is False


def test_the_forward_keeps_the_request_and_drops_the_control_token(app_module):
    seen = []

    def upstream(request):
        seen.append(request)
        return httpx.Response(201, json={"ok": True}, headers={"x-guest": "1"})

    response = _forwarding(app_module, upstream).post(
        "/osworld/setup/launch?a=1&a=2", content=json.dumps({"command": ["gedit"]}),
        headers={"Authorization": "Bearer control", "Content-Type": "application/json"})
    assert response.status_code == 201 and response.json() == {"ok": True} and response.headers["x-guest"] == "1"
    [request] = seen
    assert str(request.url) == "http://127.0.0.1:5000/setup/launch?a=1&a=2"
    assert request.method == "POST" and json.loads(request.content) == {"command": ["gedit"]}
    assert "authorization" not in request.headers


def test_an_unreachable_guest_server_is_a_502(app_module):
    def upstream(request):
        raise httpx.ConnectError("refused")

    response = _forwarding(app_module, upstream).get("/osworld/platform")
    assert response.status_code == 502 and "unreachable" in response.text


def test_the_computer_fetches_upstreams_guest_server_at_the_packages_commit_checked_by_sha256(app_module):
    downloads = app_module.OSWorldServer._installs["linux"].install.downloads
    assert app_module.OSWORLD_COMMIT == osworld.OSWORLD_COMMIT
    assert {(d.url, d.sha256, d.to) for d in downloads} == {
        (f"https://raw.githubusercontent.com/xlang-ai/OSWorld/{osworld.OSWORLD_COMMIT}/desktop_env/server/{name}", sha256,
         f"payload/{name}") for name, sha256 in app_module.GUEST_FILES.items()}
    assert set(app_module.GUEST_FILES) == {"main.py", "pyxcursor.py", "screen_recording.py"}


def test_the_start_script_runs_the_guest_server_on_loopback_without_the_debugger():
    script = (osworld.APP_DIR / "start-osworld-server.sh").read_text()
    assert 'run_name="osworld_guest"' in script and 'run_name="__main__"' not in script
    assert 'run(host="127.0.0.1", port=5000, debug=False' in script


def test_the_bundle_ships_the_app_under_its_alias():
    aliases = tomllib.loads((osworld.BUNDLE_DIR / "aliases.toml").read_text())
    assert aliases == {"envs": {osworld.APP_NAME: [osworld.APP_NAME]}}
