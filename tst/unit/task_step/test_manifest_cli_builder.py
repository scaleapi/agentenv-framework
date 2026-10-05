from __future__ import annotations

import asyncio
import json
import os
import py_compile
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_env.task_step.task_steps.mcp_cli_builder import generate_cli_script
from agentenv_protocol import (
    GET_INTERFACES_EXTENSION_URI,
    MANIFEST_VERSION,
    WELL_KNOWN_PATH,
    CliManifest,
)

from agent_env.task_step.task_steps.mcp_cli_builder.build_mcp_cli import (
    fetch_interface_manifest,
)


CRM_MANIFEST = {
    "service": "crm",
    "manifest_version": "0.1.0",
    "interface": "cli",
    "entities": [
        {
            "entity": "CRMContact",
            "commands": {
                "list": {
                    "tool": "crm_search_contacts",
                    "params": [
                        {"name": "full_name", "type": "string", "required": True},
                    ],
                },
            },
        },
        {
            "entity": "Lead",
            "commands": {
                "list": {
                    "tool": "crm_search_leads",
                    "params": [
                        {
                            "name": "status",
                            "type": "string",
                            "required": False,
                            "enum": ["new", "qualified"],
                        },
                    ],
                },
            },
        },
    ],
    "actions": [
        {
            "name": "show_data",
            "tool": "crm_show_data",
            "params": [
                {"name": "mode", "type": "string", "enum": ["raw", "summary"]},
            ],
            "description": "Show raw data",
        },
    ],
}

CRM_TOOLS = [
    {
        "name": "crm_search_contacts",
        "description": "Search contacts",
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
                "cursor": {"type": ["string", "null"], "default": None},
            },
            "required": ["full_name"],
        },
    },
    {
        "name": "crm_search_leads",
        "description": "Search leads",
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": ["string", "null"]}},
            "required": [],
        },
    },
    {
        "name": "crm_show_data",
        "description": "Show data",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 100},
                "cursor": {"type": ["string", "null"], "default": None},
                "mode": {"type": "string"},  # live schema: free string; manifest adds the enum
            },
            "required": [],
        },
    },
]


_MANIFEST_ENDPOINT = "/svc/mcp-crm/agentenv/interface-manifest"

# A composed gateway card: the backing server nests under children_environments
# with its endpoint already rewritten to /svc/{key}/..., which is what the
# gateway serves in production.
CRM_CARD = {
    "name": "AgentEnvGateway",
    "protocolVersion": "1.0",
    "url": "/agentenv",
    "capabilities": {"extensions": []},
    "children_environments": [
        {
            "name": "crm",
            "capabilities": {
                "extensions": [
                    {
                        "uri": GET_INTERFACES_EXTENSION_URI,
                        "params": {
                            "endpoint": _MANIFEST_ENDPOINT,
                            "methods": {"get_interface_manifests": {"method": "GET"}},
                        },
                    }
                ]
            },
        }
    ],
}


class _GatewayHandler(BaseHTTPRequestHandler):
    calls: list[dict] = []
    serve_manifest = True
    allowed_names: set[str] | None = None
    # Card + manifest are driven by attributes rather than by replacing do_GET, so a
    # test that reshapes one response cannot silently remove the other route.
    card: dict | None = None
    manifest_reply: tuple[int, bytes, str] | None = None
    # Matched exactly, so a fetch landing anywhere else 404s: this is what pins the
    # constructed URL, and lets a test move the manifest to prove the card is read.
    manifest_path: str = f"{_MANIFEST_ENDPOINT}/cli"

    def log_message(self, format, *args):
        pass

    def _json(self, status, body):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _raw(self, status, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            self._json(200, {"mcp_servers": [{"name": "crm", "tools": CRM_TOOLS}]})
        elif self.path == WELL_KNOWN_PATH:
            if self.card is None:
                self._json(404, {"detail": "not found"})
            else:
                self._json(200, self.card)
        elif self.path == self.manifest_path:
            if self.manifest_reply is not None:
                self._raw(*self.manifest_reply)
            elif self.serve_manifest:
                self._json(200, CRM_MANIFEST)
            else:
                self._json(404, {"detail": "not found"})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/step":
            self._json(404, {"detail": "not found"})
        elif payload.get("action") == "list_tools":
            tools = CRM_TOOLS
            if self.allowed_names is not None:
                tools = [
                    tool
                    for tool in tools
                    if tool["name"] in self.allowed_names
                ]
            self._json(200, {"tools": tools})
        elif payload.get("action") == "call_tool":
            self.calls.append(payload)
            self._json(200, {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "tool": payload["tool_name"],
                        "arguments": payload["arguments"],
                    }),
                }],
            })
        else:
            self._json(400, {"error": "unknown action"})


@pytest.fixture
def fake_gateway(socket_enabled):
    _GatewayHandler.calls = []
    _GatewayHandler.serve_manifest = True
    _GatewayHandler.allowed_names = None
    _GatewayHandler.card = CRM_CARD
    _GatewayHandler.manifest_reply = None
    _GatewayHandler.manifest_path = f"{_MANIFEST_ENDPOINT}/cli"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _write_cli(tmp, manifest=CRM_MANIFEST):
    path = os.path.join(tmp, "crm")
    with open(path, "w") as f:
        f.write(generate_cli_script("crm", interface_manifest=manifest))
    py_compile.compile(path, doraise=True)
    # The test runs this generated CLI via its shebang, so it needs an execute bit; 0o700 is the
    # least-permissive executable mode (owner-only, no group/world). 0o644 would drop +x.
    os.chmod(path, 0o700)  # nosemgrep: insecure-file-permissions
    return path


def _manifest(entities, actions=()):
    return {
        "service": "crm",
        "manifest_version": MANIFEST_VERSION,
        "interface": "cli",
        "entities": list(entities),
        "actions": list(actions),
    }


# `Crm` in service crm strips to nothing, so its verbs render at the root.
_NAMESAKE_ENTITY = {
    "entity": "Crm",
    "commands": {
        "list": {
            "tool": "crm_search_contacts",
            "params": [{"name": "full_name", "type": "string", "required": True}],
        },
    },
}


def _cli_env(gateway_url):
    return {
        "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"],
        "AGENT_ENV_GATEWAY_URL": gateway_url,
    }


def _run_cli(path, *args, env, timeout=15):
    # `path` is a temp CLI this test just wrote and `args` are static test literals, so the
    # non-literal argv is not externally controllable.
    return subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
        [str(path), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _json_reply(status, body):
    return (status, json.dumps(body).encode(), "application/json")


def test_manifest_fetch_and_404_fallback(fake_gateway):
    manifest = asyncio.run(fetch_interface_manifest(fake_gateway, "crm"))
    assert manifest == CRM_MANIFEST

    _GatewayHandler.serve_manifest = False
    assert asyncio.run(fetch_interface_manifest(fake_gateway, "crm")) is None


def test_manifest_endpoint_comes_from_the_card(fake_gateway):
    # Proves the card is actually read: the manifest moves to an endpoint the
    # constructed path can never reach, so a fetch that ignores the card 404s.
    _GatewayHandler.manifest_path = "/svc/mcp-elsewhere/agentenv/interface-manifest/cli"
    _GatewayHandler.card = {
        **CRM_CARD,
        "children_environments": [
            {
                "name": "crm",
                "capabilities": {
                    "extensions": [
                        {
                            "uri": GET_INTERFACES_EXTENSION_URI,
                            "params": {"endpoint": "/svc/mcp-elsewhere/agentenv/interface-manifest"},
                        }
                    ]
                },
            }
        ],
    }
    assert asyncio.run(fetch_interface_manifest(fake_gateway, "crm")) == CRM_MANIFEST


@pytest.mark.parametrize(
    "card",
    [
        None,                                                          # card unreachable
        {"name": "gw", "children_environments": []},                   # legacy server: no card to compose
        {"name": "gw", "children_environments": [                      # child predating the extension
            {"name": "crm", "capabilities": {"extensions": []}}]},
    ],
    ids=["unreachable", "no-children", "no-extension"],
)
def test_manifest_endpoint_falls_back_to_the_constructed_path(fake_gateway, card):
    # One case per branch of _resolve_manifest_endpoint. Images built before the
    # extension existed advertise nothing, so discovery must degrade to the path
    # the renderer used to build by hand.
    _GatewayHandler.card = card
    assert asyncio.run(fetch_interface_manifest(fake_gateway, "crm")) == CRM_MANIFEST


def test_manifest_fetch_validates_service(fake_gateway):
    _GatewayHandler.manifest_reply = _json_reply(200, {**CRM_MANIFEST, "service": "other"})
    with pytest.raises(ValueError, match="does not match deployed service"):
        asyncio.run(fetch_interface_manifest(fake_gateway, "crm"))


def test_manifest_fetch_degrades_on_server_error(fake_gateway):
    # A non-404 error status (e.g. a transient 502 while the server is still
    # booting) must degrade to the flat CLI, not fail the build.
    _GatewayHandler.manifest_reply = _json_reply(502, {"detail": "boom"})
    assert asyncio.run(fetch_interface_manifest(fake_gateway, "crm")) is None


def test_manifest_fetch_degrades_on_connection_error(socket_enabled):
    # An unreachable gateway degrades to the flat CLI rather than raising — the
    # card fetch fails first, then the constructed path fails too.
    assert asyncio.run(fetch_interface_manifest("http://127.0.0.1:1", "crm")) is None


def test_manifest_fetch_degrades_on_non_json(fake_gateway):
    _GatewayHandler.manifest_reply = (200, b"<html>not json</html>", "text/html")
    assert asyncio.run(fetch_interface_manifest(fake_gateway, "crm")) is None


def test_manifest_fetch_degrades_on_non_manifest_json(fake_gateway):
    # A 200 JSON body without a `service` field is not a manifest; degrade.
    _GatewayHandler.manifest_reply = _json_reply(200, {"unrelated": "payload"})
    assert asyncio.run(fetch_interface_manifest(fake_gateway, "crm")) is None


def test_reserved_entity_name_rejected():
    # An entity normalizing to the reserved "actions" would be shadowed; reject it.
    bad = {
        "service": "crm",
        "manifest_version": MANIFEST_VERSION,
        "interface": "cli",
        "entities": [
            {"entity": "CRMActions", "commands": {"list": {"tool": "crm_search_x", "params": []}}}
        ],
        "actions": [],
    }
    with pytest.raises(ValueError, match="reserved"):
        generate_cli_script("crm", interface_manifest=bad)


@pytest.mark.parametrize("name", ["", None])
def test_unusable_entity_name_rejected(name):
    # Missing, empty, or null: all normalize to nothing and would bake a blank,
    # unreachable subcommand. Guard the misconfigured manifest instead.
    bad = _manifest([{"entity": name, "commands": {"list": {"tool": "crm_search_x", "params": []}}}])
    with pytest.raises(ValueError, match="no usable CLI command name"):
        generate_cli_script("crm", interface_manifest=bad)


@pytest.mark.parametrize("tool", ["", None])
def test_unusable_action_name_rejected(tool):
    bad = _manifest([], [{"name": "broken", "tool": tool, "params": []}])
    with pytest.raises(ValueError, match="no usable CLI command name"):
        generate_cli_script("crm", interface_manifest=bad)


@pytest.mark.parametrize("where", ["entity", "action"])
@pytest.mark.parametrize(
    "params, match",
    [
        ({"limit": {"name": "limit"}}, "params must be a list"),
        (["limit"], "params entries must be objects"),
        ([{"required": True}], "no usable name"),
    ],
)
def test_malformed_params_rejected(where, params, match):
    # A malformed params projection must fail here, not inside the generated CLI.
    if where == "entity":
        bad = _manifest(
            [{"entity": "CRMContact", "commands": {"list": {"tool": "crm_search_x", "params": params}}}]
        )
    else:
        bad = _manifest([], [{"name": "show-data", "tool": "crm_show_data", "params": params}])
    with pytest.raises(ValueError, match=match):
        generate_cli_script("crm", interface_manifest=bad)


def test_namesake_entity_renders_verbs_at_root(fake_gateway, tmp_path):
    # `Crm` in service crm strips to nothing: `crm list`, not `crm crm list`.
    path = _write_cli(str(tmp_path), manifest=_manifest([_NAMESAKE_ENTITY]))
    result = _run_cli(path, "--help", env=_cli_env(fake_gateway))
    assert result.returncode == 0, result.stderr
    listing = result.stdout.partition("Commands:")[2].split()
    assert "list" in listing
    assert "crm" not in listing

    called = _run_cli(path, "list", "--full-name", "Ada", env=_cli_env(fake_gateway))
    assert called.returncode == 0, called.stderr
    assert _GatewayHandler.calls[-1]["tool_name"] == "crm_search_contacts"


@pytest.mark.parametrize(
    "entities, match",
    [
        ([_NAMESAKE_ENTITY, {"entity": "CRM", "commands": {}}], "only one can supply root commands"),
        ([_NAMESAKE_ENTITY, {"entity": "CRMList", "commands": {}}], "collides with another root"),
        (
            [{**_NAMESAKE_ENTITY, "commands": {"actions": {"tool": "crm_search_contacts", "params": []}}}],
            "collides with another root",
        ),
    ],
)
def test_root_command_collisions_rejected(entities, match):
    with pytest.raises(ValueError, match=match):
        generate_cli_script("crm", interface_manifest=_manifest(entities))


def test_manifest_live_schema_drift_warns(fake_gateway, tmp_path):
    # The renderer parses whatever is served: surface drift instead of dropping it silently.
    drifted = _manifest(
        [
            {
                "entity": "CRMContact",
                "commands": {
                    "list": {
                        "tool": "crm_search_contacts",
                        "params": [
                            {"name": "ghost", "type": "string"},
                            {"name": "full_name", "type": "string", "required": False},
                        ],
                    },
                },
            },
        ]
    )
    path = _write_cli(str(tmp_path), manifest=drifted)
    result = _run_cli(path, "contact", "--help", env=_cli_env(fake_gateway))
    assert result.returncode == 0
    assert "manifest param 'ghost' is absent from the live tool schema" in result.stderr
    assert "live tool requires 'full_name'" in result.stderr


def test_service_name_is_repr_quoted():
    script = generate_cli_script("crm", environment_name='we"ird')
    assert 'SERVICE_NAME = \'we"ird\'' in script
    compile(script, "<generated>", "exec")


def test_manifest_cli_groups_entities_and_actions(fake_gateway, tmp_path):
    path = _write_cli(str(tmp_path))
    result = _run_cli(path, "--help", env=_cli_env(fake_gateway))
    assert result.returncode == 0
    assert "contact" in result.stdout
    assert "lead" in result.stdout
    assert "actions" in result.stdout


def test_custom_command_name_still_scopes_real_service(fake_gateway, tmp_path):
    path = tmp_path / "sales"
    path.write_text(
        generate_cli_script(
            "sales",
            interface_manifest=CRM_MANIFEST,
            environment_name="crm",
        )
    )
    path.chmod(0o700)  # owner-only rwx: executable for the test runner, no group/world access
    result = _run_cli(path, "--help", env=_cli_env(fake_gateway))
    assert result.returncode == 0
    assert "contact" in result.stdout


def test_entity_command_uses_manifest_and_live_optional_params(fake_gateway, tmp_path):
    path = _write_cli(str(tmp_path))
    result = _run_cli(
        path, "contact", "list", "--full-name", "Alice", "--limit", "7",
        env=_cli_env(fake_gateway),
    )
    assert result.returncode == 0, result.stderr
    assert _GatewayHandler.calls[-1] == {
        "action": "call_tool",
        "tool_name": "crm_search_contacts",
        "arguments": {"full_name": "Alice", "limit": 7},
    }


def test_manifest_enum_overrides_live_schema(fake_gateway, tmp_path):
    path = _write_cli(str(tmp_path))
    result = _run_cli(path, "lead", "list", "--status", "invalid", env=_cli_env(fake_gateway))
    assert result.returncode == 2
    assert "Invalid value for '--status'" in result.stderr


def test_action_command_uses_live_parameters(fake_gateway, tmp_path):
    path = _write_cli(str(tmp_path))
    result = _run_cli(path, "actions", "show-data", "--limit", "5", env=_cli_env(fake_gateway))
    assert result.returncode == 0, result.stderr
    assert _GatewayHandler.calls[-1]["tool_name"] == "crm_show_data"
    assert _GatewayHandler.calls[-1]["arguments"] == {"limit": 5}


def test_action_command_honors_manifest_enum(fake_gateway, tmp_path):
    # `mode` is a free string in the live schema; the manifest's enum must still
    # apply to the action, so a bad value is rejected.
    path = _write_cli(str(tmp_path))
    result = _run_cli(path, "actions", "show-data", "--mode", "bogus", env=_cli_env(fake_gateway))
    assert result.returncode == 2
    assert "Invalid value for '--mode'" in result.stderr


def test_role_filtered_tools_disappear_without_rebuild(fake_gateway, tmp_path):
    path = _write_cli(str(tmp_path))
    env = _cli_env(fake_gateway)

    _GatewayHandler.allowed_names = {"crm_search_contacts"}
    result = _run_cli(path, "--help", env=env)
    assert result.returncode == 0
    assert "contact" in result.stdout
    assert "lead" not in result.stdout
    assert "actions" not in result.stdout


def test_server_without_manifest_keeps_flat_tool_commands(fake_gateway, tmp_path):
    path = _write_cli(str(tmp_path), manifest=None)
    result = _run_cli(path, "--help", env=_cli_env(fake_gateway))
    assert result.returncode == 0
    assert "crm-search-contacts" in result.stdout
    assert "\n  contact " not in result.stdout


def test_manifest_rejects_unsupported_version_and_collisions():
    with pytest.raises(ValueError, match="Unsupported Interface Manifest version"):
        generate_cli_script(
            "crm",
            interface_manifest={**CRM_MANIFEST, "manifest_version": "9.0.0"},
        )

    # Pre-1.0, a minor bump is a breaking change (semver): reject it.
    with pytest.raises(ValueError, match="Unsupported Interface Manifest version"):
        generate_cli_script(
            "crm",
            interface_manifest={**CRM_MANIFEST, "manifest_version": "0.2.0"},
        )

    # Patch-level skew is tolerated: a 0.1.x server must not force a lockstep
    # renderer deploy.
    generate_cli_script(
        "crm",
        interface_manifest={**CRM_MANIFEST, "manifest_version": "0.1.7"},
    )

    # Two entities projecting to the same CLI command name is a build error.
    duplicate = json.loads(json.dumps(CRM_MANIFEST))
    duplicate["entities"].append(json.loads(json.dumps(duplicate["entities"][0])))
    with pytest.raises(ValueError, match="Duplicate entity CLI command"):
        generate_cli_script("crm", interface_manifest=duplicate)


def test_protocol_manifest_round_trips_through_codegen():
    # Contract test: what a server emits through the protocol's CliManifest
    # models is exactly what the renderer accepts.
    model = CliManifest.model_validate(CRM_MANIFEST)
    round_tripped = json.loads(model.to_json())
    generate_cli_script("crm", interface_manifest=round_tripped)

    # A newer producer may add fields this renderer does not know; they are
    # dropped at validation rather than breaking the round trip.
    extended = {**CRM_MANIFEST, "color_scheme": "dark"}
    assert not hasattr(CliManifest.model_validate(extended), "color_scheme")
