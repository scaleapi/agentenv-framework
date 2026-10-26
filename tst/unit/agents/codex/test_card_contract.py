"""The regression checklist for codex's generated Agent Card: what agent-env negotiates
against has to stay where it was, minus the S3 forms."""

import asyncio

from tst.unit.agents.codex.agent import a2a_server

_CODEX_SKILL = {
    "id": "codex",
    "name": "codex",
    "description": "Runs the OpenAI Codex CLI with full tool access (shell, file edits, web).",
    "tags": ["coding", "agent"],
}


def _extensions(codex) -> dict:
    return {ext["uri"]: ext for ext in codex.card["capabilities"]["extensions"]}


def test_identity(codex):
    card = codex.card
    assert {key: card[key] for key in (
        "name", "description", "version", "url", "defaultInputModes", "defaultOutputModes", "skills",
    )} == {
        "name": "Codex",
        "description": "A2A-compliant wrapper around the OpenAI Codex CLI.",
        "version": "1.0.0",
        "url": "/a2a",
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [_CODEX_SKILL],
    }
    capabilities = card["capabilities"]
    assert (capabilities["streaming"], capabilities["pushNotifications"], capabilities["stateTransitionHistory"]) == (
        False, False, False)


def test_extensions_are_the_same_set_in_the_same_order(codex):
    assert list(_extensions(codex)) == [
        "urn:agentenv:agent-config/v1",
        "urn:agentenv:trajectory/v1",
        "urn:agentenv:attribution-probe/v1",
        "urn:agentenv:mcp-config/v1",
        "urn:agentenv:skill-config/v1",
        "urn:agentenv:install/v1",
        "urn:agentenv:peer-agents/v1",
        "urn:agentenv:snapshot/v1",
        "urn:agentenv:triggers/v1",
        "urn:agentenv:staging/v1",
    ]


def test_descriptions(codex):
    assert {uri: ext["description"] for uri, ext in _extensions(codex).items()} == {
        "urn:agentenv:agent-config/v1": "Set or read agent configuration such as model and role.",
        "urn:agentenv:trajectory/v1": "Retrieve a task's Codex --json trajectory, also while it runs.",
        "urn:agentenv:attribution-probe/v1": (
            "Read back the LiteLLM attribution headers the agent most recently emitted on an outbound "
            "LLM call. Used by the agent-env runtime attribution validator."
        ),
        "urn:agentenv:mcp-config/v1": "Register MCP servers the Codex CLI should connect to.",
        "urn:agentenv:skill-config/v1": "Register a skill loaded by the Codex CLI from ~/.agents/skills/.",
        "urn:agentenv:install/v1": (
            "Commands to install this agent into an already-running task container (in place; no new image)."
        ),
        "urn:agentenv:peer-agents/v1": (
            "Delegate to the task's other agents: the CLI reaches them through the SDK's peer_agents MCP tools."
        ),
        "urn:agentenv:snapshot/v1": (
            "Capture and restore Codex threads and workspace via agent-env's object-transfer grants."
        ),
        "urn:agentenv:triggers/v1": (
            "Register a typed trigger script and decide the stakeholder reply per turn "
            "(step / env_trigger / conversational / all / any conditions -> say / end actions)."
        ),
        "urn:agentenv:staging/v1": "Stage objects on the agent for agent-env to push and pull.",
    }


def test_endpoints_and_verbs(codex):
    routes = {
        uri: (
            ext["params"].get("endpoint"),
            {name: (method["method"], method.get("endpoint")) for name, method in ext["params"].get("methods", {}).items()},
        )
        for uri, ext in _extensions(codex).items()
    }
    assert routes == {
        "urn:agentenv:agent-config/v1": ("/ext/agent-config", {"set": ("POST", None), "get": ("GET", None)}),
        "urn:agentenv:trajectory/v1": ("/ext/trajectory", {"get": ("POST", None)}),
        "urn:agentenv:attribution-probe/v1": ("/ext/attribution-probe", {"probe": ("POST", None)}),
        "urn:agentenv:mcp-config/v1": ("/ext/mcp-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:skill-config/v1": ("/ext/skill-config", {"add": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:install/v1": (None, {}),
        "urn:agentenv:peer-agents/v1": ("/ext/peer-agents", {"set": ("POST", None), "list": ("GET", None)}),
        "urn:agentenv:snapshot/v1": ("/ext/snapshot", {"save": ("POST", None), "load": ("PUT", None)}),
        "urn:agentenv:triggers/v1": ("/ext/triggers", {
            "register": ("POST", None), "decide": ("POST", "/ext/triggers/decide"), "state": ("GET", None),
        }),
        "urn:agentenv:staging/v1": ("/ext/staging", {}),
    }


def test_request_and_response_contracts(codex):
    methods = {uri: ext["params"].get("methods", {}) for uri, ext in _extensions(codex).items()}
    config = methods["urn:agentenv:agent-config/v1"]
    assert config["set"]["request"]["supported"] == [
        "description", "effort", "model", "name", "output_format", "project_id",
        "role", "system_prompt", "task_id", "timeout_seconds",
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


def test_install_params(codex):
    params = _extensions(codex)["urn:agentenv:install/v1"]["params"]
    assert list(params) == [
        "install_commands", "a2a_port", "required_params", "install_commands_host", "required_params_host",
        "workdir_host", "user_host",
    ]
    assert params["a2a_port"] == 8000
    endpoint = ["litellm_api_key", "litellm_base_url", "litellm_extra_headers", "trust_env"]
    assert params["required_params"] == ["container", "work_dir", "agent_ctx_tar", "a2a_port", *endpoint]
    assert params["required_params_host"] == ["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", *endpoint]
    assert params["install_commands"] == a2a_server._INSTALL_COMMANDS
    assert params["install_commands_host"] == a2a_server._INSTALL_COMMANDS_HOST
    assert (params["workdir_host"], params["user_host"]) == ("/app", "root")


def test_the_card_and_health_are_served(codex):
    async def exercise():
        async with codex.client() as client:
            return [await client.get(path) for path in (
                "/.well-known/agent.json", "/.well-known/agent-card.json", "/health",
            )]

    legacy, current, health = asyncio.run(exercise())
    assert legacy.json() == current.json()
    assert legacy.json()["name"] == "Codex"
    assert health.json() == {"status": "ok"}
