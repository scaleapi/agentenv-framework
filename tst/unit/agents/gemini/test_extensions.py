"""agent-config, mcp-config (and each run's settings it makes), the startup bootstrap, skill-config,
attribution-probe, peer-agents and triggers over HTTP, as agent-env calls them."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from agentenv_protocol.a2a_agent import AGENT_CONFIG_V1
from starlette.testclient import TestClient

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge, gemini_config
from tst.unit.agents.gemini.support import card, namespace_grant, send

AGENT_CONFIG_PATH = AGENT_CONFIG_V1.endpoint
MCP_CONFIG_PATH = a2a_server.MCP_CONFIG_V1.endpoint
SKILL_CONFIG_PATH = a2a_server.SKILL_CONFIG_V1.endpoint
PROBE_PATH = a2a_server.ATTRIBUTION_PROBE_V1.endpoint
PEERS_PATH = a2a_server.PEER_AGENTS_V1.endpoint
CHANGELOG_PATH = "/ext/snapshot/changelog"


# ── agent-config ─────────────────────────────────────────────────────────────────


def test_config_set_is_read_back_and_renames_the_card(app, client):
    assert client.post(AGENT_CONFIG_PATH, json={"role": "solver", "name": "Renamed", "model": "m"}).json() == {
        "status": "updated"
    }
    assert client.post(AGENT_CONFIG_PATH, json={"description": "New description", "allowed_tools": ["x"]}).status_code == 200

    assert client.get(AGENT_CONFIG_PATH).json() == {
        "config": {"role": "solver", "name": "Renamed", "model": "m", "description": "New description", "allowed_tools": ["x"]}
    }
    assert (card(app)["name"], card(app)["description"]) == ("Renamed", "New description")


@pytest.mark.parametrize(
    "body",
    [
        {"effort": "high"},
        {"output_format": {"type": "json_schema"}},
        {"max_turns": 3},
        {"timeout_seconds": "soon"},
        {"allowed_tools": "run_shell_command"},
        {"name": ""},
    ],
)
def test_config_set_refuses_what_gemini_does_not_support(client, body):
    response = client.post(AGENT_CONFIG_PATH, json=body)

    assert response.status_code == 400
    assert client.get(AGENT_CONFIG_PATH).json() == {"config": {}}


# ── mcp-config and settings.json ─────────────────────────────────────────────────


def test_a_server_is_registered_without_underscores(home, client):
    added = client.post(MCP_CONFIG_PATH, json={"url": " https://gw/env_1/mcp ", "name": "env_0042"})
    generated = client.post(MCP_CONFIG_PATH, json={"url": "https://gw/other/mcp", "headers": {"CF-Access-Client-Id": "id"}})

    assert added.json() == {"status": "added", "name": "env0042", "url": "https://gw/env_1/mcp"}
    name = generated.json()["name"]
    assert re.fullmatch(r"mcp[0-9a-f]{8}", name)
    listed = client.get(MCP_CONFIG_PATH).json()["mcp_servers"]
    peer = listed.pop("peer_agents")
    assert listed == {
        "env0042": {"url": "https://gw/env_1/mcp", "has_headers": False},
        name: {"url": "https://gw/other/mcp", "has_headers": True},
    }
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/ext/peer-agents/mcp", peer["url"])
    assert not (home / ".gemini" / "settings.json").exists()


def test_each_run_gets_its_servers_role_and_system_prompt_in_settings_of_its_own(monkeypatch, home, client):
    """Turns of different roles running at once share no settings: each CLI's system settings name its turn's role,
    the SDK's peer tools among the servers, and its system prompt as a context file named by its content, so the same
    prompt keeps one path from turn to turn; all are removed once the turn ends."""
    runs = []

    async def fake_run_gemini(**kwargs):
        path = Path(kwargs["env"]["GEMINI_CLI_SYSTEM_SETTINGS_PATH"])
        run = json.loads(path.read_text())
        context = [home / ".gemini" / name for name in run.get("context", {}).get("fileName", [])[1:]]
        runs.append((path, run, {c.name: c.read_text() for c in context}))
        return gemini_bridge.GeminiResult(response="ok", tool_call_count=0, session_id=None)

    monkeypatch.setattr(a2a_server, "run_gemini", fake_run_gemini)
    client.post(MCP_CONFIG_PATH, json={"url": "https://gw/a/mcp", "name": "before", "headers": {"X-Key": "k"}})
    client.post(MCP_CONFIG_PATH, json={"url": "https://gw/b/mcp", "name": "audited", "headers": {"agentenv-role": "auditor"}})
    client.post(AGENT_CONFIG_PATH, json={"role": "solver", "system_prompt": "be brief"})
    send(client, "one", context_id="ctx-1")
    client.post(AGENT_CONFIG_PATH, json={"role": None, "system_prompt": None})
    send(client, "two", context_id="ctx-2")
    client.post(AGENT_CONFIG_PATH, json={"role": "solver", "system_prompt": "be brief"})
    send(client, "three", context_id="ctx-1")

    (first_path, first, first_context), (second_path, second, second_context), (_, third, _) = runs
    peer = first["mcpServers"]["peeragents"]["httpUrl"]
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/ext/peer-agents/mcp", peer)
    assert first["mcpServers"] == {
        "before": {"httpUrl": "https://gw/a/mcp", "headers": {"X-Key": "k", "AgentEnv-Role": "solver"}},
        "audited": {"httpUrl": "https://gw/b/mcp", "headers": {"AgentEnv-Role": "solver"}},
        "peeragents": {"httpUrl": peer, "headers": {"AgentEnv-Role": "solver"}},
    }
    (prompt_file,) = first_context
    assert first["context"] == {"fileName": ["GEMINI.md", prompt_file]} and first_context[prompt_file] == "be brief"
    assert third["context"] == first["context"]
    assert second["mcpServers"] == {
        "before": {"httpUrl": "https://gw/a/mcp", "headers": {"X-Key": "k"}},
        "audited": {"httpUrl": "https://gw/b/mcp", "headers": {"agentenv-role": "auditor"}},
        "peeragents": {"httpUrl": peer},
    }
    assert "context" not in second and second_context == {}
    assert first_path != second_path and not first_path.exists() and not second_path.exists()
    assert not (home / ".gemini" / prompt_file).exists()



def test_turns_sharing_a_system_prompt_share_its_file_until_the_last_ends(home):
    with gemini_config.run_settings({}, "solver", "be brief") as first:
        with gemini_config.run_settings({}, "auditor", "be brief") as second:
            names = [json.loads(p.read_text())["context"]["fileName"][1] for p in (first, second)]
            assert names[0] == names[1]
        prompt = home / ".gemini" / names[0]
        assert prompt.read_text() == "be brief" and not second.exists()
    assert not prompt.exists() and not first.exists()

def test_a_duplicate_url_or_name_is_a_409(home, client):
    client.post(MCP_CONFIG_PATH, json={"url": "https://gw/a/mcp", "name": "env1"})

    same_url = client.post(MCP_CONFIG_PATH, json={"url": "https://gw/a/mcp", "name": "other"})
    same_name = client.post(MCP_CONFIG_PATH, json={"url": "https://gw/b/mcp", "name": "env_1"})
    empty = client.post(MCP_CONFIG_PATH, json={"url": "  "})

    assert (same_url.status_code, same_url.json()) == (409, {"detail": "MCP URL 'https://gw/a/mcp' already registered as 'env1'"})
    assert (same_name.status_code, same_name.json()) == (409, {"detail": "MCP name 'env1' is already registered"})
    assert (empty.status_code, empty.json()) == (400, {"detail": "url must not be empty"})


def test_the_changelog_hook_follows_gemini_cli_home(monkeypatch, tmp_path, home, client):
    cli_home = tmp_path / "cli-home"
    monkeypatch.setenv("GEMINI_CLI_HOME", str(cli_home))

    client.post(CHANGELOG_PATH, json={"write_namespace": namespace_grant(), "roots": [str(tmp_path)]})

    assert list(json.loads((cli_home / ".gemini" / "settings.json").read_text())) == ["hooks"]
    assert not (home / ".gemini" / "settings.json").exists()


def test_startup_bootstraps_the_project_slug_a_load_needs(monkeypatch, tmp_path, home, app):
    fresh = tmp_path / "fresh" / ".gemini"
    for module in (gemini_bridge, a2a_server):
        monkeypatch.setattr(module, "GEMINI_HOME", fresh)
        monkeypatch.setattr(module, "GEMINI_TMP", fresh / "tmp")
        monkeypatch.setattr(module, "GEMINI_PROJECTS_JSON", fresh / "projects.json")

    with TestClient(app):
        pass

    assert json.loads((fresh / "projects.json").read_text()) == {"projects": {a2a_server.GEMINI_CWD: "workspace"}}
    assert (fresh / "tmp" / "workspace" / "chats").is_dir()
    assert (fresh / "tmp" / "workspace" / ".project_root").is_file()
    assert gemini_bridge.chats_dir_for_cwd(a2a_server.GEMINI_CWD) == fresh / "tmp" / "workspace" / "chats"


def test_startup_leaves_an_existing_project_registry_alone(home, app):
    registry = home / ".gemini" / "projects.json"
    registry.write_text(json.dumps({"projects": {a2a_server.GEMINI_CWD: "workspace-2"}}))

    with TestClient(app):
        pass

    assert json.loads(registry.read_text()) == {"projects": {a2a_server.GEMINI_CWD: "workspace-2"}}


# ── skill-config ─────────────────────────────────────────────────────────────────


def test_an_inline_skill_is_installed_listed_and_projected(home, app, client):
    response = client.post(SKILL_CONFIG_PATH, json={"name": "demo", "description": "d", "skill_md": "# Demo\n"})

    assert response.json() == {"name": "demo"}
    assert (a2a_server.SKILLS_DIR / "demo" / "SKILL.md").read_text() == "# Demo\n"
    assert client.get(SKILL_CONFIG_PATH).json() == {
        "skills": {
            "gemini": {"description": "Runs the Google Gemini CLI with full tool access (shell, file edits, web)."},
            "demo": {"description": "d"},
        }
    }
    assert card(app)["skills"][-1] == {"id": "skill-demo", "name": "demo", "description": "d", "tags": ["skill"]}


def test_a_duplicate_or_identity_skill_name_is_a_409(home, client):
    body = {"name": "demo", "description": "d", "skill_md": "# Demo\n"}
    assert client.post(SKILL_CONFIG_PATH, json=body).status_code == 200

    duplicate = client.post(SKILL_CONFIG_PATH, json=body)
    identity = client.post(SKILL_CONFIG_PATH, json={**body, "name": "gemini"})

    assert (duplicate.status_code, duplicate.json()) == (409, {"detail": "Skill 'demo' is already registered"})
    assert (identity.status_code, identity.json()) == (409, {"detail": "Skill 'gemini' is already registered"})
    assert not (a2a_server.SKILLS_DIR / "gemini").exists()


def test_a_bad_skill_name_or_empty_content_is_a_400_and_leaves_nothing(home, client):
    dotted = client.post(SKILL_CONFIG_PATH, json={"name": "a..b", "description": "d", "skill_md": "x"})
    traversal = client.post(SKILL_CONFIG_PATH, json={"name": "../x", "description": "d", "skill_md": "x"})
    empty = client.post(SKILL_CONFIG_PATH, json={"name": "blank", "description": "d", "skill_md": "  \n"})

    assert (dotted.status_code, dotted.json()) == (
        400, {"detail": "Invalid skill name 'a..b': must not contain path separators or '..'"})
    assert traversal.status_code == 400
    assert (empty.status_code, empty.json()) == (400, {"detail": "Skill content must not be empty"})
    assert not (a2a_server.SKILLS_DIR / "blank").exists()
    assert set(client.get(SKILL_CONFIG_PATH).json()["skills"]) == {"gemini"}


# ── attribution-probe ────────────────────────────────────────────────────────────


def test_the_probe_reports_the_last_attribution_the_proxy_sent(monkeypatch, client):
    monkeypatch.setattr(
        gemini_bridge, "_last_attribution", {"project_id": None, "task_id": None, "last_seen_at_utc": None}
    )
    assert client.post(PROBE_PATH, json={}).json() == {"last_seen_attribution": {}}

    gemini_bridge.record_attribution("p-1", None)

    answer = client.post(PROBE_PATH).json()
    assert answer["last_seen_attribution"] == {"project_id": "p-1"}
    assert answer["last_seen_at_utc"] == gemini_bridge.get_last_attribution_snapshot()["last_seen_at_utc"]


# ── peer-agents ──────────────────────────────────────────────────────────────────


def test_peers_are_served_by_the_sdk(client):
    peers = [{"name": "helper", "url": "https://peer/a2a"}]

    assert client.post(PEERS_PATH, json={"peers": peers}).status_code == 200
    assert client.get(PEERS_PATH).json() == {"peers": peers}


def test_a_peer_list_that_is_not_a_list_is_a_400(client):
    assert client.post(PEERS_PATH, json={"peers": {"name": "helper"}}).status_code == 400


# ── triggers ─────────────────────────────────────────────────────────────────────


def test_triggers_register_decide_and_report_their_state(client):
    script = {"triggers": [{"id": "hello", "when": {"type": "step", "turn": 1}, "actions": [{"type": "say", "text": "hi"}]}]}

    registered = client.post("/ext/triggers", json=script).json()
    decided = client.post("/ext/triggers/decide", json={"turn": 1, "solver_message": "x"}).json()
    state = client.get("/ext/triggers").json()

    assert registered == {"ok": True, "added": ["hello"], "all": ["hello"]}
    assert decided == {"parts": [{"kind": "text", "text": "hi"}], "done": False, "fired": ["hello"]}
    assert [entry.get("trigger_id") for entry in state["firing_log"] if entry["kind"] == "fired"] == ["hello"]
