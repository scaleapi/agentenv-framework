"""The osworld-server app: its card advertises the osworld-guest/v1 extension at the prefix its server forwards to the
guest server under, that forward keeps the method, path, query and body and drops the computer's control token, an
unreachable guest server is a 502, and staging copies the app's folder with the guest server's source."""
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

from agent_env import osworld
from agent_env.osworld.__main__ import main


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


def test_staging_copies_the_app_with_the_guest_servers_source(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    for name in osworld.PAYLOAD_FILES:
        (source / name).write_text(f"# {name}")
    out = tmp_path / "out"
    assert main(["stage", "--source", str(source), "--out", str(out)]) == 0
    assert (out / "env.toml").is_file() and (out / "start-osworld-server.sh").is_file()
    assert (out / "payload" / "main.py").read_text() == "# main.py"
    assert main(["stage", "--source", str(source), "--out", str(out)]) == 1
    with pytest.raises(FileNotFoundError, match="pyxcursor.py"):
        (source / "pyxcursor.py").unlink() or osworld.stage_server(source, tmp_path / "again")
