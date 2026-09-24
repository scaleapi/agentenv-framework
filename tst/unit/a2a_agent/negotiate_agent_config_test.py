"""Unit tests for A2AAgent.negotiate_agent_config."""

from agent_env.a2a_agent import A2AAgent, NegotiatedAgentConfig


def _card(supported, *, endpoint=None):
    params = {"methods": {"set": {"request": {"supported": supported}}}}
    if endpoint is not None:
        params["endpoint"] = endpoint
    return {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": params}]}}


def test_none_when_extension_absent():
    assert A2AAgent.negotiate_agent_config({}, {"model": "m"}) is None
    assert A2AAgent.negotiate_agent_config({"capabilities": {"extensions": []}}, {"model": "m"}) is None


def test_none_when_intersection_empty():
    assert A2AAgent.negotiate_agent_config(_card(["system_prompt"]), {"model": "m"}) is None
    assert A2AAgent.negotiate_agent_config(_card([]), {"model": "m"}) is None


def test_filters_desired_to_supported_and_returns_endpoint():
    result = A2AAgent.negotiate_agent_config(
        _card(["model", "model_params"]),
        {"model": "m", "system_prompt": "sp", "model_params": {"aws_region_name": "us-west-2"}},
    )
    assert result == NegotiatedAgentConfig("/ext/agent-config", {"model": "m", "model_params": {"aws_region_name": "us-west-2"}})


def test_honors_custom_endpoint():
    result = A2AAgent.negotiate_agent_config(_card(["model"], endpoint="/custom"), {"model": "m"})
    assert result.endpoint == "/custom"


def test_model_params_reaches_only_opted_in_agents():
    params = {"aws_region_name": "us-west-2", "aws_secret_access_key": "sk-secret"}
    desired = {"model": "bedrock/x", "model_params": params}

    opted_in = A2AAgent.negotiate_agent_config(_card(["model", "model_params"]), desired)
    assert opted_in.fields["model_params"] == params

    result = A2AAgent.negotiate_agent_config(_card(["model"]), desired)
    assert result is not None
    assert "model_params" not in result.fields
    assert "sk-secret" not in str(result.fields)
