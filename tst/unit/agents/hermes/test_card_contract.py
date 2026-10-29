"""The Agent Card agent-env negotiates against: the wrapper's identity and extensions, and the card as agent-env reads
it, by importing the server."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tst.unit.agents.hermes.agent import AGENT_DIR, hermes_bridge


def _methods(card: dict) -> dict[str, list[str]]:
    return {ext["uri"]: sorted((ext.get("params") or {}).get("methods") or {}) for ext in card["capabilities"]["extensions"]}


def test_identity(hermes):
    card = hermes.card
    assert {key: card[key] for key in ("name", "version", "defaultInputModes", "defaultOutputModes")} == {
        "name": "Hermes Agent",
        "version": "1.0.0",
        "defaultInputModes": ["text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"],
        "defaultOutputModes": ["text"],
    }
    assert [skill["id"] for skill in card["skills"]] == ["hermes"]


def test_the_extensions_are_the_wrappers(hermes):
    assert _methods(hermes.card) == {
        "urn:agentenv:agent-config/v1": ["get", "set"],
        "urn:agentenv:mcp-config/v1": ["add", "list"],
        "urn:agentenv:trajectory/v1": ["get"],
        "urn:agentenv:attribution-probe/v1": ["probe"],
        "urn:agentenv:peer-agents/v1": ["list", "set"],
        "urn:agentenv:skill-config/v1": ["add", "list"],
        "urn:agentenv:snapshot/v1": ["load", "save"],
        "urn:agentenv:staging/v1": [],
    }


@pytest.mark.asyncio
async def test_the_attribution_probe_reads_back_what_the_last_turn_sent(monkeypatch, hermes):
    monkeypatch.setattr(hermes_bridge, "_last_attribution", {"project_id": None, "task_id": None,
                                                             "last_seen_at_utc": None})
    hermes_bridge._record_attribution("p-1", "t-1")

    async with hermes.client() as client:
        probed = (await client.post("/ext/attribution-probe")).json()

    assert probed["last_seen_attribution"] == {"project_id": "p-1", "task_id": "t-1"}
    assert probed["last_seen_at_utc"]


def test_the_card_is_read_by_importing_agent_card():
    """agent-env imports AGENT_CARD from the image with no agent environment, and parses its stdout as the card."""
    script = "import json; from a2a_server import AGENT_CARD; print(json.dumps(AGENT_CARD.model_dump(mode='json')))"
    env = {key: value for key, value in os.environ.items() if not key.startswith("LITELLM_")}
    out = subprocess.run([sys.executable, "-c", script], cwd=AGENT_DIR, env=env, capture_output=True, text=True,
                         check=True)

    assert json.loads(out.stdout)["name"] == "Hermes Agent"


@pytest.mark.asyncio
async def test_errors_are_answered_with_a_detail_body(hermes):
    async with hermes.client() as client:
        response = await client.post("/ext/snapshot", json={"context_id": "never-ran", "objects": {
            "trajectory": {"media_type": "application/octet-stream", "max_bytes": 1024,
                           "write": {"kind": "http-put", "url": "https://objects.example/o",
                                     "expires_at": "2099-01-01T00:00:00Z"}}}})

    assert (response.status_code, response.json()) == (404, {"detail": "No active session for context_id"})
