"""Which key, URL and headers each place that hands out the model endpoint sends: an agent's deploy, core's own model
call (the direct judge's) and a unit-test verifier's container. The configured key goes only with the configured
endpoint; any other URL needs a key of its own."""

from __future__ import annotations

import asyncio
import json
import shlex
import types

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.a2a_agent.a2a_agent import TAKES_CREDENTIALS, A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.config import get_config
from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelConfig, endpoint_host, same_endpoint
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.task_step.context import DeployedSandbox, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import ResultRows
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)

MODEL = "http://localhost:4000/v1"
# Anthropic's clients append their own /v1/messages, so their variable is the endpoint without its /v1.
MODEL_ROOT = "http://localhost:4000"
HEADERS = {"X-Bill-To": "acme"}
ELSEWHERE = "http://127.0.0.1:5001"
REFUSED = "refused"


@pytest.fixture
def configured(monkeypatch, local_stores):
    monkeypatch.setenv("LITELLM_BASE_URL", MODEL)
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setattr(local_stores, "_model_cfg", ModelConfig(headers=HEADERS))


class _Started(Exception):
    """Raised once the fake provider has seen the container's env."""


class _Provider(SandboxProvider):
    async def create_sandbox(self, *, env, **_):
        self.env = env
        raise _Started


def _deploy(monkeypatch, passed: dict | None, own: dict | None, **metadata) -> dict | str:
    """What the agent's container is started with, or REFUSED when the deploy stops before starting one."""
    provider = _Provider()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)
    agent = A2AAgent(id="solver", version=1, metadata=metadata, default_env_vars=own,
                     docker_image_artifact=DockerImageArtifact(id="img", description="d", image_name="img"))
    try:
        asyncio.run(agent.deploy(sandbox_type="any", env_vars=passed))
    except _Started:
        pass
    except ConfigError as e:
        assert "127.0.0.1:5001 isn't the configured model endpoint" in str(e) and not hasattr(provider, "env")
        return REFUSED
    headers = provider.env.get("LITELLM_EXTRA_HEADERS")
    return {"key": provider.env.get("LITELLM_API_KEY"), "url": provider.env.get("LITELLM_BASE_URL"),
            "headers": json.loads(headers) if headers else None}


AGENT_CASES = {
    "configured": (None, None, {"key": "k", "url": MODEL, "headers": HEADERS}),
    "passed-configured": ({"LITELLM_BASE_URL": MODEL}, None, {"key": "k", "url": MODEL, "headers": HEADERS}),
    "passed-configured-slash": ({"LITELLM_BASE_URL": MODEL + "/"}, None,
                                {"key": "k", "url": MODEL + "/", "headers": HEADERS}),
    "passed-configured-without-v1": ({"LITELLM_BASE_URL": "http://localhost:4000"}, None,
                                     {"key": "k", "url": "http://localhost:4000", "headers": HEADERS}),
    "passed-configured-upper-case": ({"LITELLM_BASE_URL": "HTTP://LOCALHOST:4000/v1"}, None,
                                     {"key": "k", "url": "HTTP://LOCALHOST:4000/v1", "headers": HEADERS}),
    "passed-elsewhere-is-refused": ({"LITELLM_BASE_URL": ELSEWHERE}, None, REFUSED),
    "passed-elsewhere-with-its-key": ({"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "theirs"}, None,
                                      {"key": "theirs", "url": ELSEWHERE, "headers": None}),
    "own-elsewhere-is-refused": (None, {"LITELLM_BASE_URL": ELSEWHERE}, REFUSED),
    "own-elsewhere-with-its-key": (None, {"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "own"},
                                   {"key": "own", "url": ELSEWHERE, "headers": None}),
    "own-elsewhere-with-a-passed-key": ({"LITELLM_API_KEY": "theirs"}, {"LITELLM_BASE_URL": ELSEWHERE},
                                        {"key": "theirs", "url": ELSEWHERE, "headers": None}),
    "passed-configured-over-own-elsewhere": ({"LITELLM_BASE_URL": MODEL}, {"LITELLM_BASE_URL": ELSEWHERE},
                                             {"key": "k", "url": MODEL, "headers": HEADERS}),
    "passed-key-only": ({"LITELLM_API_KEY": "theirs"}, None, {"key": "theirs", "url": MODEL, "headers": HEADERS}),
}


@pytest.mark.parametrize("passed, own, expected", AGENT_CASES.values(), ids=AGENT_CASES.keys())
def test_an_agent_deploy_sends(monkeypatch, configured, passed, own, expected):
    assert _deploy(monkeypatch, passed, own) == expected


@pytest.mark.parametrize("passed, own, expected", [case for case in AGENT_CASES.values() if case[2] != REFUSED],
                         ids=[name for name, case in AGENT_CASES.items() if case[2] != REFUSED])
def test_the_endpoint_an_agent_records_is_the_one_its_deploy_sends(configured, passed, own, expected):
    agent = A2AAgent(id="solver", version=1, default_env_vars=own,
                     docker_image_artifact=DockerImageArtifact(id="img", description="d", image_name="img"))

    assert agent.model_base_url(dict(passed or {}), get_config()) == expected["url"]
    assert agent.on_configured_endpoint(dict(passed or {}), get_config()) is same_endpoint(expected["url"], MODEL)


def test_an_endpoints_recorded_host_carries_no_userinfo_and_an_agent_without_credentials_none(configured):
    agent = A2AAgent(id="solver", version=1, metadata={TAKES_CREDENTIALS: False},
                     docker_image_artifact=DockerImageArtifact(id="img", description="d", image_name="img"))

    assert endpoint_host("https://user:secret@gw.example.com:8443/v1") == "gw.example.com:8443"
    assert agent.model_base_url({"LITELLM_BASE_URL": ELSEWHERE}, get_config()) is None
    assert agent.on_configured_endpoint({}, get_config()) is False


def test_an_agent_that_takes_no_credentials_gets_none(monkeypatch, configured):
    sent = _deploy(monkeypatch, {"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "x"}, None,
                   **{TAKES_CREDENTIALS: False})
    assert sent == {"key": None, "url": None, "headers": None}


GATEWAY = "https://gw.example.com"
OTHER = "https://elsewhere.example.com"
CALL_CASES = {
    "configured": ({}, {}, ("file-key", GATEWAY, True)),
    "configured-env-key": ({"LITELLM_API_KEY": "env-key"}, {}, ("env-key", GATEWAY, True)),
    "elsewhere-env-key-is-refused": ({"LITELLM_API_KEY": "env-key"}, {"base_override": OTHER}, REFUSED),
    "elsewhere-no-key-is-refused": ({}, {"base_override": OTHER}, REFUSED),
    "elsewhere-with-its-key": ({}, {"base_override": OTHER, "default_api_key": "theirs"}, ("theirs", OTHER, False)),
    "configured-with-v1": ({}, {"base_override": GATEWAY + "/v1"}, ("file-key", GATEWAY + "/v1", True)),
    "configured-with-default-port": ({}, {"base_override": "https://GW.example.com:443/"},
                                     ("file-key", "https://GW.example.com:443/", True)),
    "env-url-is-the-configured-one": ({"LITELLM_BASE_URL": OTHER}, {"base_override": OTHER}, ("file-key", OTHER, True)),
    "file-url-under-an-env-url-is-refused": ({"LITELLM_BASE_URL": OTHER}, {"base_override": GATEWAY}, REFUSED),
}


@pytest.mark.parametrize("env, call, expected", CALL_CASES.values(), ids=CALL_CASES.keys())
def test_a_model_call_core_makes_sends(monkeypatch, env, call, expected):
    for name in ("LITELLM_BASE_URL", "LITELLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = ModelConfig(base_url=GATEWAY, api_key="file-key", headers={"X-Bill-To": "acme"})
    try:
        resolved = config.resolve_call("m", secret_resolver=lambda key: None, **call)
    except ConfigError as e:
        assert "isn't the configured model endpoint" in str(e)
        assert expected == REFUSED
        return
    assert (resolved.api_key, resolved.api_base, "extra_headers" in resolved.params) == expected


def test_a_natively_routed_call_gets_the_configured_key(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    resolved = ModelConfig(api_key="file-key").resolve_call("openai/m", secret_resolver=lambda key: None)
    assert (resolved.api_key, resolved.api_base) == ("file-key", None)


class _VerifierSandbox:
    def __init__(self):
        self.type = "modal_vm"
        self.commands: list[str] = []

    def scoped_name(self, name):
        return name

    async def exec_script(self, script: str) -> str:
        self.commands.append(script)
        return ""

    async def exec_with_output(self, *args) -> tuple[int, str, str]:
        self.commands.append(args[-1])
        return 0, "ok", ""


def _verifier_env(monkeypatch, env_vars: dict, step_params: dict | None = None) -> dict | str:
    """The model variables a unit-test verifier's command ran with, or REFUSED when it ran none."""
    sandbox = _VerifierSandbox()
    provider = types.SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=sandbox))
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)
    monkeypatch.setattr(RunContainerUnitTestsVerifierTaskStep, "_upload_text_artifact", staticmethod(
        lambda text, artifact_id, description, object_url: types.SimpleNamespace(
            id=artifact_id, version=1, object_url=object_url)))
    context = TaskStepContext()
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    if step_params:
        context.metadata["user_overrides"] = {"step_params": {"tests": {"env_vars": step_params}}}
    context.deployed_sandboxes.append(
        DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="modal_vm"))
    step = RunContainerUnitTestsVerifierTaskStep(
        id="tests", version=None, sandbox_name="h", container_name="c", command="true", env_vars=env_vars)
    try:
        asyncio.run(step.execute(context))
    except ConfigError as e:
        assert "127.0.0.1:5001 isn't the configured model endpoint" in str(e)
        assert not [c for c in sandbox.commands if "timeout --kill-after" in c]
        return REFUSED
    [run] = [c for c in sandbox.commands if "timeout --kill-after" in c]
    words = shlex.split(run)
    pairs = dict(words[i + 1].split("=", 1) for i, word in enumerate(words) if word == "-e")
    return {name: pairs.get(name) for name in ("LITELLM_API_KEY", "LITELLM_BASE_URL", "ANTHROPIC_API_KEY",
                                               "ANTHROPIC_BASE_URL")}


VERIFIER_CASES = {
    "configured": ({}, None, {"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": MODEL, "ANTHROPIC_API_KEY": "k",
                              "ANTHROPIC_BASE_URL": MODEL_ROOT}),
    "elsewhere-is-refused": ({"LITELLM_BASE_URL": ELSEWHERE}, None, REFUSED),
    "elsewhere-with-its-key": ({"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "theirs"}, None,
                               {"LITELLM_API_KEY": "theirs", "LITELLM_BASE_URL": ELSEWHERE, "ANTHROPIC_API_KEY": "k",
                                "ANTHROPIC_BASE_URL": MODEL_ROOT}),
    "anthropic-elsewhere-is-refused": ({"ANTHROPIC_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "theirs"}, None, REFUSED),
    "anthropic-elsewhere-with-its-key": ({"ANTHROPIC_BASE_URL": ELSEWHERE, "ANTHROPIC_API_KEY": "theirs"}, None,
                                         {"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": MODEL,
                                          "ANTHROPIC_API_KEY": "theirs", "ANTHROPIC_BASE_URL": ELSEWHERE}),
    "elsewhere-by-step-params-is-refused": ({}, {"LITELLM_BASE_URL": ELSEWHERE}, REFUSED),
    "elsewhere-by-step-params-with-the-steps-key": ({"LITELLM_API_KEY": "theirs"}, {"LITELLM_BASE_URL": ELSEWHERE},
                                                    {"LITELLM_API_KEY": "theirs", "LITELLM_BASE_URL": ELSEWHERE,
                                                     "ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": MODEL_ROOT}),
}


@pytest.mark.parametrize("env_vars, step_params, expected", VERIFIER_CASES.values(), ids=VERIFIER_CASES.keys())
def test_a_unit_test_verifier_runs_its_tests_with(monkeypatch, configured, env_vars, step_params, expected):
    assert _verifier_env(monkeypatch, env_vars, step_params) == expected


def _judge_env(monkeypatch, overrides: dict) -> dict | str:
    """The model variables an agent judge is deployed with, or REFUSED when none is deployed."""
    step = RubricsVerifierTaskStep(id="judge", version=1, criteria=[{"id": "done", "description": "It is done"}],
                                   prompt_id="p1", verifier_id="v", use_agent_judge=True, use_trajectory=False,
                                   judge_a2a_agent_id="judge-agent")
    agent = MagicMock()
    agent.deploy = AsyncMock(return_value=types.SimpleNamespace(
        a2a_url="http://judge", sandbox_id="sb-judge", agent_card={}, sandbox_type="modal"))
    monkeypatch.setattr(A2AAgent, "get", MagicMock(return_value=agent))

    async def judged(*, criteria, **_):
        return ResultRows([{**c, "justification": "", "score": 1.0, "result": True} for c in criteria],
                          reasoning="", checks=[]), 0, [], None

    monkeypatch.setattr(step, "_run_judge_with_output_retries", judged)
    provider = MagicMock(get_sandbox=AsyncMock(return_value=AsyncMock()))
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)
    context = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it")],
                              metadata={"user_overrides": overrides})
    try:
        asyncio.run(step.execute(context))
    except ConfigError as e:
        assert "127.0.0.1:5001 isn't the configured model endpoint" in str(e) and not agent.deploy.called
        return REFUSED
    env = agent.deploy.call_args.kwargs["env_vars"]
    return {"key": env.get("LITELLM_API_KEY"), "url": env.get("LITELLM_BASE_URL")}


JUDGE_CASES = {
    "configured": ({}, {"key": "k", "url": MODEL}),
    "judge-elsewhere-is-refused": ({"judge_litellm_base_url": ELSEWHERE}, REFUSED),
    "run-elsewhere-is-refused": ({"litellm_base_url": ELSEWHERE}, REFUSED),
    "judge-elsewhere-with-its-key": ({"judge_litellm_base_url": ELSEWHERE, "judge_litellm_api_key": "theirs"},
                                     {"key": "theirs", "url": ELSEWHERE}),
    "run-elsewhere-with-the-runs-key": ({"litellm_base_url": ELSEWHERE, "litellm_api_key": "theirs"},
                                        {"key": "theirs", "url": ELSEWHERE}),
    "judge-configured-spelled-otherwise": ({"judge_litellm_base_url": "http://LOCALHOST:4000/"},
                                           {"key": "k", "url": "http://LOCALHOST:4000/"}),
}


@pytest.mark.parametrize("overrides, expected", JUDGE_CASES.values(), ids=JUDGE_CASES.keys())
def test_an_agent_judge_is_deployed_with(monkeypatch, configured, overrides, expected):
    assert _judge_env(monkeypatch, overrides) == expected
