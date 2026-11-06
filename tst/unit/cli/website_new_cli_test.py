"""agent-env env website new: a site like the Notes example, which imports and passes the website/v1 conformance
check as it is written; refused over a non-empty folder without --force; and, from an MCP server's source or a tools
JSON, one stub @api route per tool with a seed collection for each."""

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from agent_env.cli.env.website import website
from agent_env.cli.env.website_scaffold import CLIENT_DIR, tools_from_json, tools_from_mcp
from agentenv_protocol.website import SESSION_PATH, WebsiteEnvironment, conformance
from agentenv_protocol.website import website as declare

MAIL_SERVER = Path(__file__).resolve().parents[3] / "src" / "agent_env" / "examples" / "mail-mcp" / "envs" / "mail"
FILES = {".gitignore", "Dockerfile", "Dockerfile.dockerignore", "README.md", "env.toml", "site.py", "data/seed.json",
         "web/index.html", "web/package.json", "web/vite.config.js", "web/src/main.js", "web/src/style.css",
         "web/src/site-client.js", "web/src/site-client.d.ts"}
FASTMCP_SERVER = '''
from typing import Annotated, Literal, Optional
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

mcp = FastMCP("calendar")


@mcp.tool(name="list_events")
async def list_events(day: str, limit: int = 10, ctx: Context = None) -> list:
    """The events of a day."""


@mcp.tool()
def create_event(
    title: Annotated[str, Field(description="What the event is.")],
    attendees: list[str],
    kind: Literal["meeting", "focus"] = "meeting",
    notes: Optional[str] = None,
    room: str = Field(None, description="Where."),
) -> dict:
    """Add an event."""


def helper(x: int) -> int:
    return x
'''


def _new(*args):
    return CliRunner().invoke(website, ["new", *args])


def _load(site_dir: Path) -> type[WebsiteEnvironment]:
    """The generated ``site.py``'s site, serving a stand-in page instead of ``web/dist``, which only a Vite build makes."""
    spec = importlib.util.spec_from_file_location(f"generated_site_{uuid.uuid4().hex}", site_dir / "site.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cls = next(v for v in vars(module).values() if isinstance(v, type) and issubclass(v, WebsiteEnvironment)
               and v is not WebsiteEnvironment)
    page = site_dir / "stand-in"
    page.mkdir()
    (page / "index.html").write_text("<!doctype html><html><head><title>site</title></head><body></body></html>")
    declared = cls.declaration()._replace(static=str(page))
    return declare(**declared._asdict())(type(cls.__name__, (cls,), {}))


def _passes_conformance(client: TestClient) -> None:
    """The conformance check, whose clock check serves a clock on 127.0.0.1 for the site to read."""
    results = conformance.check("http://testserver", client=client)
    assert {r.name: r.status for r in results} == dict.fromkeys(
        ["card", "any_host", "health", "universe_now", "reserved_paths", "data_plane", "clock", "forwarded", "cookies"],
        "passed") | {"mcp": "skipped"}, results


def test_a_new_site_has_notes_layout_and_the_sdks_browser_client(tmp_path):
    result = _new("shop", "--dir", str(tmp_path / "shop"), "--domain", "shop.example", "--domain", "*.shop.example")
    assert result.exit_code == 0, result.output
    assert "agent-env env website dev" in result.output and "--health-path" not in result.output
    site = tmp_path / "shop"
    assert {str(p.relative_to(site)) for p in site.rglob("*") if p.is_file()} == FILES
    assert (site / "env.toml").read_text() == ('type = "website"\nenvironment_name = "shop"\n'
                                               'backend_image = { dockerfile = "Dockerfile" }\n'
                                               'domains = ["shop.example", "*.shop.example"]\n')
    assert "--health-path" not in (site / "README.md").read_text()
    package = json.loads((site / "web" / "package.json").read_text())
    assert package["devDependencies"] == {"vite": "5.4.10"} and "dependencies" not in package
    vite = (site / "web" / "vite.config.js").read_text()
    assert 'base: "./"' in vite and '"/api":' in vite and "healthz" not in vite
    assert 'from "./site-client.js"' in (site / "web" / "src" / "main.js").read_text()
    page = (site / "web" / "index.html").read_text()
    assert page.index('location.pathname.endsWith("/")') < page.index('<script type="module"')
    for client in ("site-client.js", "site-client.d.ts"):
        assert (site / "web" / "src" / client).read_text() == (CLIENT_DIR / client).read_text()
    assert "{{" not in "".join(p.read_text() for p in site.rglob("*") if p.is_file())
    assert 'domains=["shop.test"]' in (site / "site.py").read_text()
    assert "COPY site.py /app/site.py" in (site / "Dockerfile").read_text()


def test_the_new_site_imports_and_passes_the_conformance_check(tmp_path, socket_enabled):
    assert _new("shop", "--dir", str(tmp_path / "shop")).exit_code == 0
    app = _load(tmp_path / "shop")().create_app()
    client = TestClient(app)
    _passes_conformance(client)
    card = client.get("/.well-known/agent-env.json").json()
    site = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:website/v1")["params"]
    assert (card["name"], site["domains"]) == ("shop", ["shop.test"])
    assert client.get(site["methods"]["ready"]["endpoint"]).json()["ready"] is True
    assert client.get("/api/session").json()["user"] == "ada"
    created = client.post("/api/items", json={"title": " Pack "})
    assert created.status_code == 201 and created.json()["title"] == "Pack"
    assert [i["title"] for i in client.get("/api/items").json()] == ["Pack", "Try the site"]
    assert client.post("/api/sign-in", json={"user": "grace"}).json() == {"user": "grace"}
    assert [i["title"] for i in client.get("/api/items").json()] == ["Write the first task"]


def test_a_new_site_without_a_domain_has_none_in_its_env_toml_and_a_bad_one_is_refused(tmp_path):
    assert _new("shop", "--dir", str(tmp_path / "shop")).exit_code == 0
    assert "domains" not in (tmp_path / "shop" / "env.toml").read_text()
    refused = _new("mail", "--dir", str(tmp_path / "mail"), "--domain", "not a host")
    assert refused.exit_code == 1 and "Error:" in refused.output
    assert not (tmp_path / "mail").exists()


def test_a_non_empty_folder_is_refused_unless_forced_and_a_bad_name_always(tmp_path):
    (tmp_path / "keep.txt").write_text("mine")
    refused = _new("shop", "--dir", str(tmp_path))
    assert refused.exit_code == 1 and "not empty; pass --force" in refused.output
    assert not (tmp_path / "site.py").exists()
    forced = _new("shop", "--dir", str(tmp_path), "--force")
    assert forced.exit_code == 0, forced.output
    assert (tmp_path / "site.py").is_file() and (tmp_path / "keep.txt").read_text() == "mine"
    bad = _new("My Shop", "--dir", str(tmp_path / "other"))
    assert bad.exit_code == 1 and "lowercase letters, digits and hyphens" in bad.output
    both = _new("shop", "--dir", str(tmp_path / "x"), "--from-mcp", str(MAIL_SERVER),
                "--from-tools-json", str(tmp_path / "keep.txt"))
    assert both.exit_code == 1 and "mutually exclusive" in both.output


def test_an_mcp_servers_tools_are_read_from_its_source_without_running_it(tmp_path):
    assert tools_from_mcp(MAIL_SERVER) == [{
        "name": "search_emails",
        "description": "Search the inbox: the emails holding every word of the query, ignoring case, as a JSON list.",
        "inputSchema": {"type": "object", "required": ["query"], "properties": {"query": {
            "type": "string", "description": "Words each email returned holds, in its sender, subject or body."}}}}]
    (tmp_path / "server.py").write_text(FASTMCP_SERVER)
    listed, created = tools_from_mcp(tmp_path)
    assert listed["name"] == "list_events" and listed["description"] == "The events of a day."
    assert listed["inputSchema"] == {"type": "object", "required": ["day"], "properties": {
        "day": {"type": "string"}, "limit": {"type": "integer", "default": 10}}}
    assert created["name"] == "create_event"
    assert created["inputSchema"]["required"] == ["title", "attendees"]
    assert created["inputSchema"]["properties"] == {
        "title": {"type": "string", "description": "What the event is."},
        "attendees": {"type": "array", "items": {"type": "string"}},
        "kind": {"enum": ["meeting", "focus"], "default": "meeting"},
        "notes": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "room": {"type": "string", "description": "Where.", "default": None}}
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "x.py").write_text("def f():\n    pass\n")
    refused = _new("cal", "--dir", str(tmp_path / "cal"), "--from-mcp", str(empty))
    assert refused.exit_code == 1 and "--from-tools-json" in refused.output


def test_a_site_from_an_mcp_server_has_a_stub_route_per_tool_and_passes_the_check(tmp_path, socket_enabled):
    (tmp_path / "mcp").mkdir()
    (tmp_path / "mcp" / "server.py").write_text(FASTMCP_SERVER)
    result = _new("calendar", "--dir", str(tmp_path / "site"), "--from-mcp", str(tmp_path / "mcp"))
    assert result.exit_code == 0, result.output
    assert "2 stub route(s)" in result.output
    seed = json.loads((tmp_path / "site" / "data" / "seed.json").read_text())
    assert seed == {"current_user": "user", "users": [{"id": "user", "name": "Sample User"}], "events": []}
    cls = _load(tmp_path / "site")
    assert [(r.method, r.path, r.name) for r in cls.routes()] == [
        ("GET", "/api/users", "list_users"), ("POST", "/api/sign-in", "sign_in"),
        ("GET", "/api/list-events", "list_events"), ("POST", "/api/create-event", "create_event")]
    client = TestClient(cls().create_app())
    _passes_conformance(client)
    listed = client.get("/api/list-events", params={"day": "2026-01-05"})
    assert (listed.status_code, listed.json()["code"]) == (501, "not_implemented")
    assert client.get("/api/list-events", params={"limit": "x"}).status_code == 400
    created = client.post("/api/create-event", json={"title": "Standup", "attendees": ["ann"], "kind": "focus"})
    assert created.status_code == 501, created.text
    assert client.post("/api/create-event", json={"title": "Standup", "kind": "party"}).status_code == 400
    main = (tmp_path / "site" / "web" / "src" / "main.js").read_text()
    assert '"path": "api/create-event"' in main and '"name": "list_events"' in main


@pytest.mark.parametrize("shape", ["tools/list", "rpc", "list", "card"])
def test_tools_json_in_any_of_its_shapes_makes_stub_routes(tmp_path, shape, socket_enabled):
    tools = [{"name": "send_email", "description": "Send an email.", "inputSchema": {"type": "object", "properties": {
        "to": {"type": "array", "items": {"type": "string"}}, "from": {"type": "string"},
        "urgent": {"type": "boolean", "default": False}}, "required": ["to"]}},
        {"name": "list_users", "description": "A tool named like a built-in route.", "inputSchema": {}}]
    data = {"tools/list": {"tools": tools}, "rpc": {"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}},
            "list": tools, "card": {"name": "mail", "capabilities": {"tools": tools}}}[shape]
    (tmp_path / "tools.json").write_text(json.dumps(data))
    assert [t["name"] for t in tools_from_json(tmp_path / "tools.json")] == ["send_email", "list_users"]
    result = _new("mail", "--dir", str(tmp_path / "site"), "--from-tools-json", str(tmp_path / "tools.json"))
    assert result.exit_code == 0, result.output
    cls = _load(tmp_path / "site")
    assert [r.name for r in cls.routes()] == ["list_users", "sign_in", "send_email", "list_users_tool"]
    client = TestClient(cls().create_app())
    _passes_conformance(client)
    assert client.post("/api/send-email", json={"to": ["a@b.c"], "from": "me@b.c"}).status_code == 501
    assert client.post("/api/send-email", json={}).status_code == 400
    assert client.get("/api/list-users").status_code == 501
    assert client.get("/api/users").json() == [{"id": "user", "name": "Sample User"}]


def test_a_tool_named_like_a_path_the_sdk_serves_gets_a_path_of_its_own(tmp_path):
    (tmp_path / "tools.json").write_text(json.dumps([{"name": "session", "inputSchema": {}}]))
    result = _new("mail", "--dir", str(tmp_path / "site"), "--from-tools-json", str(tmp_path / "tools.json"))
    assert result.exit_code == 0, result.output
    stub = next(r for r in _load(tmp_path / "site").routes() if r.name == "session")
    assert SESSION_PATH.startswith("/api/") and stub.path == f"{SESSION_PATH}-tool"


def test_a_tool_named_with_a_leading_digit_makes_a_site_that_loads(tmp_path):
    (tmp_path / "tools.json").write_text(json.dumps([{"name": "3d_render", "inputSchema": {}}]))
    result = _new("render", "--dir", str(tmp_path / "site"), "--from-tools-json", str(tmp_path / "tools.json"))
    assert result.exit_code == 0, result.output
    assert "class N3dRenderRequest(BaseModel)" in (tmp_path / "site" / "site.py").read_text()
    assert [r.name for r in _load(tmp_path / "site").routes()][-1] == "3d_render"


def test_a_tool_description_ending_in_a_quote_or_a_backslash_makes_a_site_that_loads(tmp_path):
    tools = [{"name": "set_status", "description": 'Set status to "done"', "inputSchema": {}},
             {"name": "read_path", "description": "Read C:\\temp\\", "inputSchema": {}},
             {"name": "find_note", "description": 'Find a "note"\n\nBy its "title"', "inputSchema":
                 {"type": "object", "properties": {"q": {"type": "string", "description": 'like "x"'}}}}]
    (tmp_path / "tools.json").write_text(json.dumps(tools))
    result = _new("status", "--dir", str(tmp_path / "site"), "--from-tools-json", str(tmp_path / "tools.json"))
    assert result.exit_code == 0, result.output
    cls = _load(tmp_path / "site")
    assert cls.set_status.__doc__ == 'Set status to "done"' and cls.read_path.__doc__ == "Read C:\\temp\\"


def test_a_tools_json_that_lists_no_tools_is_refused(tmp_path):
    (tmp_path / "tools.json").write_text(json.dumps({"tools": "nope"}))
    refused = _new("mail", "--dir", str(tmp_path / "site"), "--from-tools-json", str(tmp_path / "tools.json"))
    assert refused.exit_code == 1 and "lists no tools" in refused.output
    assert not (tmp_path / "site").exists()


def test_tools_named_after_what_the_site_module_uses_make_a_site_that_loads(tmp_path):
    names = ["api", "ready_check", "website", "environment_card", "ApiContext", "ApiError", "NotFound", "BaseModel",
             "Field", "Optional", "Any", "Literal", "SignIn", "USERS", "list", "dict", "str", "int", "float", "bool",
             "after_all", "data", "reset", "add", "export", "copy", "json", "Path", "SEED"]
    (tmp_path / "tools.json").write_text(json.dumps([{"name": n, "inputSchema": {}} for n in names]))
    result = _new("shadow", "--dir", str(tmp_path / "site"), "--from-tools-json", str(tmp_path / "tools.json"))
    assert result.exit_code == 0, result.output
    cls = _load(tmp_path / "site")
    routes = cls.routes()
    assert [r.name for r in routes] == ["list_users", "sign_in", *names]
    client = TestClient(cls().create_app())
    for route in routes[2:]:
        answer = client.request(route.method, route.path, **({} if route.method == "GET" else {"json": {}}))
        assert (answer.status_code, answer.json()["code"]) == (501, "not_implemented"), route
