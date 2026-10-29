"""The regression checklist for gemini-cli's generated Agent Card: what agent-env
negotiates against has to stay where it was, minus the S3 forms."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tst.unit.agents.gemini.agent import AGENT_DIR, a2a_server
from tst.unit.agents.gemini.support import card

_GEMINI_SKILL = {
    "id": "gemini",
    "name": "gemini",
    "description": "Runs the Google Gemini CLI with full tool access (shell, file edits, web).",
    "tags": ["coding", "agent"],
}


def _extensions(app) -> dict:
    return {ext["uri"]: ext for ext in card(app)["capabilities"]["extensions"]}


def test_identity(app):
    served = card(app)
    assert {key: served[key] for key in (
        "name", "description", "version", "url", "defaultInputModes", "defaultOutputModes", "skills",
    )} == {
        "name": "Gemini",
        "description": "A2A-compliant wrapper around the Google Gemini CLI.",
        "version": "1.0.0",
        "url": "/a2a",
        "defaultInputModes": [
            "text",
            "image/png", "image/jpeg", "image/gif", "image/webp",
            "audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg",
            "application/pdf",
            "video/mp4",
        ],
        "defaultOutputModes": ["text"],
        "skills": [_GEMINI_SKILL],
    }
    capabilities = served["capabilities"]
    assert (capabilities["streaming"], capabilities["pushNotifications"], capabilities["stateTransitionHistory"]) == (
        False, False, False)


def test_extensions_are_the_same_set_in_the_framework_order(app):
    """The framework lists agent-config first, then the declared extensions in order.
    Consumers look an extension up by URI, so the order is cosmetic."""
    assert list(_extensions(app)) == [
        "urn:agentenv:agent-config/v1",
        "urn:agentenv:install/v1",
        "urn:agentenv:attribution-probe/v1",
        "urn:agentenv:trajectory/v1",
        "urn:agentenv:mcp-config/v1",
        "urn:agentenv:skill-config/v1",
        "urn:agentenv:peer-agents/v1",
        "urn:agentenv:snapshot/v1",
        "urn:agentenv:triggers/v1",
        "urn:agentenv:staging/v1",
    ]


def test_descriptions(app):
    assert {uri: ext["description"] for uri, ext in _extensions(app).items()} == {
        "urn:agentenv:install/v1": (
            "Commands to install this agent into a running task container, or directly onto a task VM (host mode)."
        ),
        "urn:agentenv:agent-config/v1": "Set or read agent configuration such as model, role, and allowed tools.",
        "urn:agentenv:attribution-probe/v1": (
            "Read back the LiteLLM attribution headers the agent most recently emitted on an outbound "
            "LLM call. Used by the agent-env runtime attribution validator."
        ),
        "urn:agentenv:trajectory/v1": "Retrieve a task's Gemini stream-json trajectory, also while it runs.",
        "urn:agentenv:mcp-config/v1": "Register MCP servers the Gemini CLI should connect to.",
        "urn:agentenv:skill-config/v1": "Register a skill loaded by the Gemini CLI from ~/.gemini/skills/.",
        "urn:agentenv:peer-agents/v1": (
            "Delegate to the task's other agents: the CLI reaches them through the SDK's peer_agents MCP tools."
        ),
        "urn:agentenv:snapshot/v1": (
            "Capture and restore Gemini CLI session transcripts, and, where the host has GNU tar, each tool call's "
            "writes as a changelog to replay onto a fresh agent, via agent-env's object-transfer grants."
        ),
        "urn:agentenv:triggers/v1": (
            "Register a typed trigger script and decide the stakeholder reply per turn "
            "(step / env_trigger / conversational / all / any conditions -> say / end actions)."
        ),
        "urn:agentenv:staging/v1": "Stage objects on the agent for agent-env to push and pull.",
    }


def test_endpoints_and_verbs(app):
    routes = {
        uri: (
            ext["params"].get("endpoint"),
            {name: (method["method"], method.get("endpoint")) for name, method in ext["params"].get("methods", {}).items()},
        )
        for uri, ext in _extensions(app).items()
    }
    assert routes == {
        "urn:agentenv:install/v1": (None, {}),
        "urn:agentenv:agent-config/v1": ("/ext/agent-config", {"set": ("POST", None), "get": ("GET", None)}),
        "urn:agentenv:attribution-probe/v1": ("/ext/attribution-probe", {"probe": ("POST", None)}),
        "urn:agentenv:trajectory/v1": ("/ext/trajectory", {"get": ("POST", None)}),
        "urn:agentenv:mcp-config/v1": ("/ext/mcp-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:skill-config/v1": ("/ext/skill-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:peer-agents/v1": ("/ext/peer-agents", {"set": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:snapshot/v1": ("/ext/snapshot", {
            "save": ("POST", None),
            "load": ("PUT", None),
            "enable-changelog": ("POST", "/ext/snapshot/changelog"),
            "apply-changelog": ("PUT", "/ext/snapshot/changelog"),
        }),
        "urn:agentenv:triggers/v1": ("/ext/triggers", {
            "register": ("POST", None), "decide": ("POST", "/ext/triggers/decide"), "state": ("GET", None),
        }),
        "urn:agentenv:staging/v1": ("/ext/staging", {}),
    }


def test_request_and_response_contracts(app):
    methods = {uri: ext["params"].get("methods", {}) for uri, ext in _extensions(app).items()}
    config = methods["urn:agentenv:agent-config/v1"]
    assert config["set"]["request"]["supported"] == [
        "allowed_tools", "description", "model", "name", "project_id", "role", "system_prompt", "task_id",
        "timeout_seconds",
    ]
    assert set(config["set"]["request"]["schema"]["properties"]) == set(config["set"]["request"]["supported"])
    assert config["get"] == {"method": "GET", "response": {"required": ["config"]}}
    assert methods["urn:agentenv:trajectory/v1"]["get"]["request"] == {
        "oneOf": [
            {"required": ["task_id"]},
            {"required": ["task_id", "objects"]},
            {"required": ["context_id"]},
            {"required": ["context_id", "objects"]},
            {"required": ["task_id", "after"], "optional": ["limit"]},
            {"required": ["context_id", "after"], "optional": ["limit"]},
        ],
    }
    assert methods["urn:agentenv:attribution-probe/v1"]["probe"] == {
        "method": "POST", "response": {"required": ["last_seen_attribution"], "optional": ["last_seen_at_utc"]},
    }
    assert methods["urn:agentenv:mcp-config/v1"] == {
        "add": {"method": "POST", "request": {"required": ["url"], "optional": ["headers", "name"]}},
        "list": {"method": "GET", "response": {"required": ["mcp_servers"]}},
    }
    assert methods["urn:agentenv:skill-config/v1"] == {
        "add": {
            "method": "POST",
            "request": {
                "required": ["name", "description"],
                "oneOf": [{"required": ["skill_md"]}, {"required": ["skill_bundle"]}],
            },
            "response": {"required": ["name"]},
        },
        "list": {"method": "GET", "response": {"required": ["skills"]}},
    }
    assert methods["urn:agentenv:peer-agents/v1"] == {
        "set": {"method": "POST", "request": {"required": ["peers"]}},
        "list": {"method": "GET", "response": {"required": ["peers"]}},
    }
    assert methods["urn:agentenv:snapshot/v1"] == {
        "save": {
            "method": "POST",
            "request": {"required": ["context_id", "objects"]},
            "response": {"required": ["context_id", "objects"]},
        },
        "load": {
            "method": "PUT",
            "request": {"required": ["objects"], "optional": ["target_context_id"]},
            "response": {"required": ["context_id"]},
        },
        "enable-changelog": {
            "method": "POST",
            "endpoint": "/ext/snapshot/changelog",
            "request": {"required": ["write_namespace"], "optional": ["roots"]},
            "response": {"required": ["roots"]},
        },
        "apply-changelog": {
            "method": "PUT",
            "endpoint": "/ext/snapshot/changelog",
            "request": {"required": ["increments"], "optional": ["resume_conversation", "target_context_id"]},
            "response": {"required": ["count"], "optional": ["context_id"]},
        },
    }
    triggers = methods["urn:agentenv:triggers/v1"]
    assert triggers["register"] == {"method": "POST", "request": {"required": ["triggers"]}}
    assert triggers["decide"] == {
        "method": "POST",
        "endpoint": "/ext/triggers/decide",
        "request": {"required": ["turn"],
                    "optional": ["solver_message", "context_id", "env_triggers", "record_unfired"]},
        "response": {"required": ["parts", "done", "fired"]},
    }
    assert triggers["state"] == {"method": "GET", "response": {"required": ["firing_log"]}}


def test_install_params(app):
    params = _extensions(app)["urn:agentenv:install/v1"]["params"]
    assert list(params) == [
        "install_commands", "required_params", "install_commands_host", "a2a_port", "required_params_host",
        "workdir_host", "user_host",
    ]
    assert params["a2a_port"] == 8000
    endpoint = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]
    assert params["required_params"] == ["container", "work_dir", "agent_ctx_tar", "a2a_port", *endpoint]
    assert params["required_params_host"] == ["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *endpoint]
    assert params["install_commands"] == a2a_server._INSTALL_COMMANDS
    assert params["install_commands_host"] == a2a_server._INSTALL_COMMANDS_HOST
    assert (params["workdir_host"], params["user_host"]) == ("/workspace", "root")


def test_the_card_and_health_are_served(client):
    legacy, current, health = (client.get(path) for path in (
        "/.well-known/agent.json", "/.well-known/agent-card.json", "/health",
    ))
    assert legacy.json() == current.json()
    assert legacy.json()["name"] == "Gemini"
    assert health.json() == {"status": "ok"}


def test_install_mode_reads_the_card_by_importing_agent_card():
    """agent-env's InstallAgentTaskStep runs this exact script in the image, with no agent
    environment, and parses its stdout as the card."""
    script = (
        "import sys, json; sys.path.insert(0, '/app'); "
        "from a2a_server import AGENT_CARD; "
        "print(json.dumps(AGENT_CARD.model_dump()))"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("LITELLM_")}
    out = subprocess.run([sys.executable, "-c", script], cwd=AGENT_DIR, env=env, capture_output=True, text=True,
                         check=True)
    exported = json.loads(out.stdout)
    (install,) = [e for e in exported["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:install/v1"]
    assert install["params"]["install_commands"] and install["params"]["install_commands_host"]
