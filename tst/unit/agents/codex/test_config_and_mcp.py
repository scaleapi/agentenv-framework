"""Config side effects, the model endpoint and MCP registration in config.toml, the SDK's peer-agents tools and the
attribution probe."""

import asyncio
import json
import os
import tomllib
from pathlib import Path

import httpx

from tst.unit.agents.codex.agent import a2a_server, codex_bridge


def _call(codex, method: str, path: str, body: dict | None = None) -> httpx.Response:
    async def exercise():
        async with codex.client() as client:
            return await client.request(method, path, json=body)

    return asyncio.run(exercise())


def _servers(codex_home: Path) -> dict:
    return tomllib.loads((codex_home / "config.toml").read_text())["mcp_servers"]


# ── agent-config ─────────────────────────────────────────────────────────────────


def test_a_set_without_effort_clears_the_previous_one(codex, monkeypatch):
    seen = []

    async def fake_run_codex(**kwargs):
        seen.append(kwargs["effort"])
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            await client.post("/ext/agent-config", json={"effort": "xhigh", "model": "m"})
            first = (await client.get("/ext/agent-config")).json()
            await codex.send(client, "one")
            await client.post("/ext/agent-config", json={"model": "m2"})
            second = (await client.get("/ext/agent-config")).json()
            await codex.send(client, "two")
            await client.post("/ext/agent-config", json={"effort": None})
            third = (await client.get("/ext/agent-config")).json()
            return first, second, third

    first, second, third = asyncio.run(exercise())
    assert first == {"config": {"effort": "xhigh", "model": "m"}}
    assert second == {"config": {"model": "m2"}}
    assert third == {"config": {"model": "m2", "effort": None}}
    assert seen == ["xhigh", None]


def test_name_and_description_update_the_card(codex):
    assert _call(codex, "POST", "/ext/agent-config", {"name": "Solver", "description": "d2"}).json() == {
        "status": "updated"}
    card = _call(codex, "GET", "/.well-known/agent.json").json()
    assert (card["name"], card["description"]) == ("Solver", "d2")


def test_an_unsupported_or_mistyped_field_is_refused(codex):
    assert _call(codex, "POST", "/ext/agent-config", {"allowed_tools": ["Bash"]}).status_code == 400
    assert _call(codex, "POST", "/ext/agent-config", {"timeout_seconds": "soon"}).status_code == 400
    assert _call(codex, "GET", "/ext/agent-config").json() == {"config": {}}


# ── MCP ──────────────────────────────────────────────────────────────────────────


def test_the_server_start_writes_the_model_endpoint_into_config_toml(codex, monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://llm.example.com/")
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps({"X-Gateway": "g"}))

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app):
            pass

    asyncio.run(exercise())

    config = tomllib.loads((Path(os.environ["CODEX_HOME"]) / "config.toml").read_text())
    assert config == {
        "features": {"multi_agent_v2": True},
        "model_provider": "litellm",
        "model_providers": {"litellm": {
            "name": "LiteLLM proxy",
            "base_url": "https://llm.example.com/openai/v1",
            "env_key": "LITELLM_API_KEY",
            "wire_api": "responses",
            "request_max_retries": 2,
            "stream_max_retries": 2,
            "stream_idle_timeout_ms": 60000,
            "env_http_headers": {"x-litellm-customer-id": "LITELLM_CUSTOMER_ID", "x-litellm-tags": "LITELLM_TAGS"},
            "http_headers": {"X-Gateway": "g"},
        }},
    }


def test_without_an_endpoint_only_the_features_are_written(codex, monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app):
            pass

    asyncio.run(exercise())

    config = tomllib.loads((Path(os.environ["CODEX_HOME"]) / "config.toml").read_text())
    assert config == {"features": {"multi_agent_v2": True}}


def test_each_turn_writes_every_server_with_the_role_header(codex, monkeypatch):
    codex_home = Path(os.environ["CODEX_HOME"])
    for name in ("NODE_EXTRA_CA_CERTS", "NODE_TLS_REJECT_UNAUTHORIZED", "SSL_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("A2A_PORT", "8000")
    snapshots = []

    async def fake_run_codex(**kwargs):
        snapshots.append(_servers(codex_home))
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            await client.post("/ext/agent-config", json={"role": "solver"})
            added = await client.post("/ext/mcp-config", json={
                "url": "https://gw/mcp", "name": "env4821", "headers": {"X-Tenant": "t"}})
            await codex.send(client, "one")
            await client.post("/ext/agent-config", json={"role": "judge"})
            await codex.send(client, "two")
            return added.json()

    assert asyncio.run(exercise()) == {"status": "added", "name": "env4821", "url": "https://gw/mcp"}

    first, second = snapshots
    assert first["env4821"] == {
        "command": "mcp-remote",
        "args": ["https://gw/mcp", "--allow-http", "--header", "X-Tenant: t", "--header", "AgentEnv-Role: solver"],
        "env": {},
        "startup_timeout_sec": 60,
        "required": True,
    }
    assert first["peer_agents"]["args"] == [
        "http://127.0.0.1:8000/ext/peer-agents/mcp", "--allow-http", "--header", "AgentEnv-Role: solver"]
    assert second["env4821"]["args"][-1] == "AgentEnv-Role: judge"
    assert {name: entry["required"] for name, entry in second.items()} == {"peer_agents": True, "env4821": True}


def test_a_turn_rewrites_config_toml_only_when_it_changes_and_never_in_place(codex, monkeypatch):
    config = Path(os.environ["CODEX_HOME"]) / "config.toml"
    inodes = []

    async def fake_run_codex(**kwargs):
        inodes.append(config.stat().st_ino)
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            await client.post("/ext/agent-config", json={"role": "solver"})
            await client.post("/ext/mcp-config", json={"url": "https://gw/mcp", "name": "env"})
            await codex.send(client, "one")
            await codex.send(client, "two")
            await client.post("/ext/agent-config", json={"role": "judge"})
            await codex.send(client, "three")

    asyncio.run(exercise())

    first, second, third = inodes
    assert first == second and third != second
    assert [p.name for p in config.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_config_toml_is_owner_only_when_new_and_keeps_its_mode_when_rewritten(codex, monkeypatch):
    config = Path(os.environ["CODEX_HOME"]) / "config.toml"
    modes = []

    async def fake_run_codex(**kwargs):
        modes.append(config.stat().st_mode & 0o777)
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            await client.post("/ext/agent-config", json={"role": "solver"})
            await codex.send(client, "one")
            config.chmod(0o640)
            await client.post("/ext/agent-config", json={"role": "judge"})
            await codex.send(client, "two")

    asyncio.run(exercise())

    assert modes == [0o600, 0o640]


def test_a_duplicate_mcp_url_or_name_is_a_conflict(codex):
    assert _call(codex, "POST", "/ext/mcp-config", {"url": "https://gw/mcp", "name": "a"}).status_code == 200
    assert _call(codex, "POST", "/ext/mcp-config", {"url": "https://gw/mcp", "name": "b"}).status_code == 409
    assert _call(codex, "POST", "/ext/mcp-config", {"url": "https://gw2/mcp", "name": "a"}).status_code == 409
    minted = _call(codex, "POST", "/ext/mcp-config", {"url": "https://gw3/mcp"}).json()["name"]
    assert minted.startswith("mcp_") and len(minted) == 12


# ── peer agents ──────────────────────────────────────────────────────────────────


def test_peers_are_served_by_the_sdk(codex):
    peers = [{"name": "probe", "url": "http://example.invalid"}, {"name": "judge", "url": "http://j"}]

    assert _call(codex, "POST", "/ext/peer-agents", {"peers": peers}).status_code == 200
    assert _call(codex, "GET", "/ext/peer-agents").json() == {"peers": peers}


# ── attribution probe ────────────────────────────────────────────────────────────


def test_the_probe_reports_what_the_bridge_last_sent(codex, monkeypatch):
    monkeypatch.setattr(codex_bridge, "_last_attribution",
                        {"project_id": None, "task_id": None, "last_seen_at_utc": None})

    assert _call(codex, "POST", "/ext/attribution-probe", {}).json() == {"last_seen_attribution": {}}

    codex_bridge._record_attribution(project_id="p-1", task_id="t-1")
    probed = _call(codex, "POST", "/ext/attribution-probe", {}).json()
    assert probed["last_seen_attribution"] == {"project_id": "p-1", "task_id": "t-1"}
    assert probed["last_seen_at_utc"] == codex_bridge._last_attribution["last_seen_at_utc"]


# ── triggers (SDK-owned) ─────────────────────────────────────────────────────────


def test_triggers_register_decide_and_state(codex):
    registered = _call(codex, "POST", "/ext/triggers", {"triggers": [
        {"id": "hello", "when": {"type": "step", "turn": 1}, "actions": [{"type": "say", "text": "hi"}]},
    ]})
    assert registered.status_code == 200
    decided = _call(codex, "POST", "/ext/triggers/decide", {"turn": 1, "solver_message": "x"}).json()
    assert (decided["parts"], decided["done"], decided["fired"]) == ([{"kind": "text", "text": "hi"}], False, ["hello"])
    assert _call(codex, "GET", "/ext/triggers").json()["firing_log"]
