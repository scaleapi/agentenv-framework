"""agent-config, skill-config and triggers over HTTP, as agent-env calls them."""

from __future__ import annotations

import pytest
from agentenv_protocol.a2a_agent import AGENT_CONFIG_V1

from tst.unit.agents.claude_code.support import card
from tst.unit.agents.claude_code.agent import a2a_server

AGENT_CONFIG_PATH = AGENT_CONFIG_V1.endpoint
SKILL_CONFIG_PATH = a2a_server.SKILL_CONFIG_V1.endpoint


# ── agent-config ─────────────────────────────────────────────────────────────────


def test_config_set_is_read_back_and_renames_the_card(app, client):
    assert client.post(AGENT_CONFIG_PATH, json={"role": "solver", "name": "Renamed", "model": "m"}).json() == {
        "status": "updated"
    }
    assert client.post(AGENT_CONFIG_PATH, json={"description": "New description"}).status_code == 200

    assert client.get(AGENT_CONFIG_PATH).json() == {
        "config": {"role": "solver", "name": "Renamed", "model": "m", "description": "New description"}
    }
    assert (card(app)["name"], card(app)["description"]) == ("Renamed", "New description")


@pytest.mark.parametrize(
    "body",
    [
        {"temperature": 0.5},
        {"timeout_seconds": "soon"},
        {"allowed_tools": "Bash"},
        {"name": ""},
    ],
)
def test_config_set_refuses_unknown_fields_wrong_types_and_empty_identity(client, body):
    response = client.post(AGENT_CONFIG_PATH, json=body)

    assert response.status_code == 400
    assert client.get(AGENT_CONFIG_PATH).json() == {"config": {}}


# ── skill-config ─────────────────────────────────────────────────────────────────


def test_an_inline_skill_is_installed_listed_and_projected(monkeypatch, tmp_path, app, client):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")

    response = client.post(SKILL_CONFIG_PATH, json={"name": "demo", "description": "d", "skill_md": "# Demo\n"})

    assert response.json() == {"name": "demo"}
    assert (tmp_path / "skills" / "demo" / "SKILL.md").read_text() == "# Demo\n"
    assert client.get(SKILL_CONFIG_PATH).json() == {
        "skills": {
            "claude-code": {"description": "Runs the Claude Code CLI with full tool access (bash, read, edit, glob, grep)."},
            "demo": {"description": "d"},
        }
    }
    assert card(app)["skills"][-1] == {"id": "skill-demo", "name": "demo", "description": "d", "tags": ["skill"]}


def test_a_duplicate_or_identity_skill_name_is_a_409(monkeypatch, tmp_path, client):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")
    body = {"name": "demo", "description": "d", "skill_md": "# Demo\n"}
    assert client.post(SKILL_CONFIG_PATH, json=body).status_code == 200

    duplicate = client.post(SKILL_CONFIG_PATH, json=body)
    identity = client.post(SKILL_CONFIG_PATH, json={**body, "name": "claude-code"})

    assert (duplicate.status_code, duplicate.json()) == (409, {"detail": "Skill 'demo' is already registered"})
    assert (identity.status_code, identity.json()) == (409, {"detail": "Skill 'claude-code' is already registered"})
    assert not (tmp_path / "skills" / "claude-code").exists()


def test_a_bad_skill_name_or_empty_content_is_a_400_and_leaves_nothing(monkeypatch, tmp_path, client):
    monkeypatch.setattr(a2a_server, "SKILLS_DIR", tmp_path / "skills")

    dotted = client.post(SKILL_CONFIG_PATH, json={"name": "a..b", "description": "d", "skill_md": "x"})
    traversal = client.post(SKILL_CONFIG_PATH, json={"name": "../x", "description": "d", "skill_md": "x"})
    empty = client.post(SKILL_CONFIG_PATH, json={"name": "blank", "description": "d", "skill_md": "  \n"})

    assert (dotted.status_code, dotted.json()) == (
        400, {"detail": "Invalid skill name 'a..b': must not contain path separators or '..'"})
    assert traversal.status_code == 400
    assert (empty.status_code, empty.json()) == (400, {"detail": "Skill content must not be empty"})
    assert not (tmp_path / "skills" / "blank").exists()
    assert set(client.get(SKILL_CONFIG_PATH).json()["skills"]) == {"claude-code"}


# ── triggers ─────────────────────────────────────────────────────────────────────


def test_triggers_register_decide_and_report_their_state(client):
    script = {"triggers": [{"id": "hello", "when": {"type": "step", "turn": 1}, "actions": [{"type": "say", "text": "hi"}]}]}

    registered = client.post("/ext/triggers", json=script).json()
    decided = client.post("/ext/triggers/decide", json={"turn": 1, "solver_message": "x"}).json()
    state = client.get("/ext/triggers").json()

    assert registered == {"ok": True, "added": ["hello"], "all": ["hello"]}
    assert decided == {"parts": [{"kind": "text", "text": "hi"}], "done": False, "fired": ["hello"]}
    assert [entry.get("trigger_id") for entry in state["firing_log"] if entry["kind"] == "fired"] == ["hello"]
