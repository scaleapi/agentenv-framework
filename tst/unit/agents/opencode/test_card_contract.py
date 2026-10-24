"""The Agent Card agent-env negotiates against: the wrapper's identity and extensions, plus the attribution probe, with
the changelog offered only where tar can make its archives; install/v1's commands, which reach install.sh with the
endpoint as environment; and the card as an install reads it, by importing the server."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from tst.unit.agents.opencode.agent import AGENT_DIR, a2a_server, opencode_bridge
from tst.unit.agents.opencode.conftest import Harness

_ENDPOINT = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]


def _methods(card: dict) -> dict[str, list[str]]:
    return {ext["uri"]: sorted((ext.get("params") or {}).get("methods") or {}) for ext in card["capabilities"]["extensions"]}


def test_identity(oc):
    card = oc.card
    assert {key: card[key] for key in ("name", "description", "version", "defaultInputModes", "defaultOutputModes",
                                       "skills")} == {
        "name": "OpenCode",
        "description": "A2A-compliant wrapper around the OpenCode CLI.",
        "version": "1.0.0",
        "defaultInputModes": ["text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"],
        "defaultOutputModes": ["text"],
        "skills": [{"id": "opencode", "name": "opencode", "tags": ["coding", "agent"],
                    "description": "Runs the OpenCode CLI with full tool access (bash, read, edit, glob, grep)."}],
    }


def test_the_extensions_are_the_wrappers_and_the_attribution_probe(oc):
    assert _methods(oc.card) == {
        "urn:agentenv:agent-config/v1": ["get", "set"],
        "urn:agentenv:mcp-config/v1": ["add", "list"],
        "urn:agentenv:trajectory/v1": ["get"],
        "urn:agentenv:attribution-probe/v1": ["probe"],
        "urn:agentenv:install/v1": [],
        "urn:agentenv:peer-agents/v1": ["list", "set"],
        "urn:agentenv:skill-config/v1": ["add", "list"],
        "urn:agentenv:snapshot/v1": ["apply-changelog", "enable-changelog", "load", "save"],
        "urn:agentenv:staging/v1": [],
    }


@pytest.mark.parametrize("gnu, changelog", [(True, True), (False, False)])
def test_the_changelog_is_offered_where_tar_is_gnu_tar(monkeypatch, gnu, changelog):
    monkeypatch.setattr(a2a_server, "gnu_tar", lambda: gnu)

    methods = _methods(Harness(a2a_server.agent_for_this_host()).card)["urn:agentenv:snapshot/v1"]

    assert ("enable-changelog" in methods and "apply-changelog" in methods) is changelog
    assert {"save", "load"} <= set(methods)


@pytest.mark.parametrize(("commands", "params"), [
    (a2a_server._INSTALL_COMMANDS, ["container", "work_dir", "agent_ctx_tar", "a2a_port", *_ENDPOINT]),
    (a2a_server._INSTALL_COMMANDS_HOST, ["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *_ENDPOINT]),
])
def test_every_install_placeholder_is_a_declared_parameter(oc, commands, params):
    (install,) = [ext for ext in oc.card["capabilities"]["extensions"] if ext["uri"] == "urn:agentenv:install/v1"]
    used = {name for command in commands for name in re.findall(r"\{(\w+)\}", command)}

    assert used == set(params)
    assert set(install["params"]["required_params"]) | set(install["params"]["required_params_host"]) >= used
    for name in ("litellm_api_key", "litellm_base_url", "litellm_extra_headers"):
        assert f"{name.upper()}={{{name}}}" in commands[-1]
    assert "AGENT_TRUST_ENV={trust_env}" in commands[-1]


@pytest.mark.asyncio
async def test_the_attribution_probe_reads_back_what_the_last_run_sent(monkeypatch, oc):
    monkeypatch.setattr(opencode_bridge, "_last_attribution", {"project_id": None, "task_id": None,
                                                               "last_seen_at_utc": None})
    opencode_bridge._record_attribution("p-1", "t-1")

    async with oc.client() as client:
        probed = (await client.post("/ext/attribution-probe")).json()

    assert probed["last_seen_attribution"] == {"project_id": "p-1", "task_id": "t-1"}
    assert probed["last_seen_at_utc"]


def test_install_mode_reads_the_card_by_importing_agent_card():
    """agent-env's install step imports AGENT_CARD from the image with no agent environment, and parses its stdout as
    the card."""
    script = "import json; from a2a_server import AGENT_CARD; print(json.dumps(AGENT_CARD.model_dump(mode='json')))"
    env = {key: value for key, value in os.environ.items() if not key.startswith("LITELLM_")}
    out = subprocess.run([sys.executable, "-c", script], cwd=AGENT_DIR, env=env, capture_output=True, text=True,
                         check=True)
    extensions = {ext["uri"]: ext for ext in json.loads(out.stdout)["capabilities"]["extensions"]}

    install = extensions["urn:agentenv:install/v1"]["params"]
    assert install["install_commands"] == a2a_server._INSTALL_COMMANDS
    assert install["install_commands_host"] == a2a_server._INSTALL_COMMANDS_HOST
    assert (install["workdir_host"], install["user_host"]) == ("/app", "root")
    assert extensions["urn:agentenv:staging/v1"]["params"]["endpoint"] == "/ext/staging"


@pytest.mark.asyncio
async def test_errors_are_answered_with_a_detail_body(oc):
    async with oc.client() as client:
        response = await client.post("/ext/snapshot", json={"context_id": "never-ran", "objects": {
            "trajectory": {"media_type": "application/octet-stream", "max_bytes": 1024,
                           "write": {"kind": "http-put", "url": "https://objects.example/o",
                                     "expires_at": "2099-01-01T00:00:00Z"}}}})

    assert (response.status_code, response.json()) == (404, {"detail": "No active session for context_id"})
