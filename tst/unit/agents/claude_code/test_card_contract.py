"""The Claude Code agent's Agent Card: what agent-env negotiates against, where it expects it."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from agent_env.task_step.task_steps.install_agent import _build_param_resolvers

from tst.unit.agents.claude_code.agent import AGENT_DIR
from tst.unit.agents.claude_code.support import card

_CLAUDE_SKILL = {
    "id": "claude-code",
    "name": "claude-code",
    "description": "Runs the Claude Code CLI with full tool access (bash, read, edit, glob, grep).",
    "tags": ["coding", "agent"],
}


def _extensions(app) -> dict:
    return {ext["uri"]: ext for ext in card(app)["capabilities"]["extensions"]}


def test_identity(app):
    served = card(app)
    assert {key: served[key] for key in (
        "name", "description", "version", "url", "defaultInputModes", "defaultOutputModes", "skills",
    )} == {
        "name": "Claude Code",
        "description": "An A2A agent running the Claude Code CLI.",
        "version": "1.0.0",
        "url": "/a2a",
        "defaultInputModes": ["text", "image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"],
        "defaultOutputModes": ["text"],
        "skills": [_CLAUDE_SKILL],
    }
    capabilities = served["capabilities"]
    assert (capabilities["streaming"], capabilities["pushNotifications"], capabilities["stateTransitionHistory"]) == (
        False, False, False)


def test_extensions_and_their_routes(app):
    routes = {
        uri: (ext["params"].get("endpoint"),
              {name: (method["method"], method.get("endpoint")) for name, method in ext["params"].get("methods", {}).items()})
        for uri, ext in _extensions(app).items()
    }
    assert routes == {
        "urn:agentenv:agent-config/v1": ("/ext/agent-config", {"set": ("POST", None), "get": ("GET", None)}),
        "urn:agentenv:trajectory/v1": ("/ext/trajectory", {"get": ("POST", None)}),
        "urn:agentenv:mcp-config/v1": ("/ext/mcp-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:peer-agents/v1": ("/ext/peer-agents", {"set": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:skill-config/v1": ("/ext/skill-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:snapshot/v1": ("/ext/snapshot", {"save": ("POST", None), "load": ("PUT", None)}),
        "urn:agentenv:install/v1": (None, {}),
        "urn:agentenv:triggers/v1": ("/ext/triggers", {
            "register": ("POST", None), "decide": ("POST", "/ext/triggers/decide"), "state": ("GET", None),
        }),
        "urn:agentenv:staging/v1": ("/ext/staging", {}),
    }


def test_the_config_takes_every_field_prompt_agent_sends_that_the_cli_has(app):
    config = _extensions(app)["urn:agentenv:agent-config/v1"]["params"]["methods"]
    assert config["set"]["request"]["supported"] == [
        "allowed_tools", "description", "effort", "max_thinking_tokens", "max_turns", "model", "name",
        "output_format", "project_id", "role", "system_prompt", "task_id", "timeout_seconds",
    ]
    assert set(config["set"]["request"]["schema"]["properties"]) == set(config["set"]["request"]["supported"])
    assert config["get"] == {"method": "GET", "response": {"required": ["config"]}}


def test_request_and_response_contracts(app):
    methods = {uri: ext["params"].get("methods", {}) for uri, ext in _extensions(app).items()}
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
    assert methods["urn:agentenv:mcp-config/v1"] == {
        "add": {"method": "POST", "request": {"required": ["url"], "optional": ["headers", "name"]}},
        "list": {"method": "GET", "response": {"required": ["mcp_servers"]}},
    }
    assert methods["urn:agentenv:skill-config/v1"] == {
        "add": {
            "method": "POST",
            "request": {"required": ["name", "description"],
                        "oneOf": [{"required": ["skill_md"]}, {"required": ["skill_bundle"]}]},
            "response": {"required": ["name"]},
        },
        "list": {"method": "GET", "response": {"required": ["skills"]}},
    }
    assert methods["urn:agentenv:snapshot/v1"] == {
        "save": {"method": "POST", "request": {"required": ["context_id", "objects"]},
                 "response": {"required": ["context_id", "objects"]}},
        "load": {"method": "PUT", "request": {"required": ["objects"], "optional": ["target_context_id"]},
                 "response": {"required": ["context_id"]}},
    }
    triggers = methods["urn:agentenv:triggers/v1"]
    assert triggers["register"] == {"method": "POST", "request": {"required": ["triggers"]}}
    assert triggers["decide"] == {
        "method": "POST",
        "endpoint": "/ext/triggers/decide",
        "request": {"required": ["turn"], "optional": ["solver_message", "context_id", "env_triggers"]},
        "response": {"required": ["parts", "done", "fired"]},
    }
    assert triggers["state"] == {"method": "GET", "response": {"required": ["firing_log"]}}


def test_the_card_and_health_are_served(client):
    legacy, current, health = (client.get(path) for path in (
        "/.well-known/agent.json", "/.well-known/agent-card.json", "/health",
    ))
    assert legacy.json() == current.json()
    assert legacy.json()["name"] == "Claude Code"
    assert health.json() == {"status": "ok"}


def _install(app) -> dict:
    return _extensions(app)["urn:agentenv:install/v1"]["params"]


def test_install_agent_reads_the_card_and_its_install_commands_from_the_image():
    """install_agent runs this script in the image, with no agent environment, and parses its stdout as the card."""
    script = ("import sys, json; sys.path.insert(0, '/app'); "
              "from a2a_server import AGENT_CARD; "
              "print(json.dumps(AGENT_CARD.model_dump()))")
    env = {**{k: v for k, v in os.environ.items() if not k.startswith(("LITELLM_", "ANTHROPIC_"))},
           "PYTHONPATH": str(AGENT_DIR)}
    out = subprocess.run([sys.executable, "-c", script], cwd=AGENT_DIR, env=env, capture_output=True, text=True,
                         check=True)
    (install,) = [ext for ext in json.loads(out.stdout)["capabilities"]["extensions"]
                  if ext["uri"] == "urn:agentenv:install/v1"]
    assert install["params"]["install_commands"] and install["params"]["install_commands_host"]


def test_each_install_mode_asks_only_for_params_install_agent_supplies(app):
    params = _install(app)
    supplied = set(_build_param_resolvers(container_name="c", work_dir="/w", agent_ctx_tar="/w/a.tgz", a2a_port=8000,
                                          config=None, agent_name="a"))
    assert params["a2a_port"] == 8000
    for commands, required in (("install_commands", "required_params"),
                               ("install_commands_host", "required_params_host")):
        placeholders = {name for command in params[commands] for name in re.findall(r"\{(\w+)\}", command)}
        assert placeholders == set(params[required]) and set(params[required]) <= supplied, commands
        assert "${" not in " ".join(params[commands])


def test_the_install_runs_the_folders_own_script_in_each_mode(app):
    params = _install(app)
    container, host = params["install_commands"][-1], params["install_commands_host"][-1]
    assert "bash {work_dir}/agent/install.sh container {container} {a2a_port}" in container
    assert "bash {work_dir}/agent/install.sh host {agent_name} {a2a_port} /app" in host
    for command in (container, host):
        # The model key reaches the script in its environment, never as an argument.
        assert command.startswith("LITELLM_API_KEY={litellm_api_key} ") and command.count("{litellm_api_key}") == 1


def test_the_install_scripts_parse():
    subprocess.run(["bash", "-n", str(AGENT_DIR / "install.sh")], check=True)
    subprocess.run(["sh", "-n", str(AGENT_DIR / "run.sh")], check=True)


def test_install_sh_installs_the_cli_version_the_image_pins():
    pinned = re.search(r"^ARG CLAUDE_CODE_VERSION=(\S+)$", (AGENT_DIR / "Dockerfile").read_text(), re.MULTILINE)
    assert pinned and "sed -n 's/^ARG CLAUDE_CODE_VERSION=//p'" in (AGENT_DIR / "install.sh").read_text()
