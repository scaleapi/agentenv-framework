"""Which key, URL and headers each place that hands out the model endpoint sends: an agent's deploy, core's own model
call (the direct judge's) and a unit-test verifier's container. The configured key goes only with the configured
endpoint; any other URL needs a key of its own."""

from __future__ import annotations

import asyncio
import json
import shlex
import types

import pytest

from agent_env.a2a_agent.a2a_agent import TAKES_CREDENTIALS, A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.config.model import ModelConfig
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)

MODEL = "http://localhost:4000/v1"
HEADERS = {"X-Bill-To": "acme"}
ELSEWHERE = "http://127.0.0.1:5001"


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


def _deploy(monkeypatch, passed: dict | None, own: dict | None, **metadata) -> dict:
    provider = _Provider()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)
    agent = A2AAgent(id="solver", version=1, metadata=metadata, default_env_vars=own,
                     docker_image_artifact=DockerImageArtifact(id="img", description="d", image_name="img"))
    with pytest.raises(_Started):
        asyncio.run(agent.deploy(sandbox_type="any", env_vars=passed))
    headers = provider.env.get("LITELLM_EXTRA_HEADERS")
    return {"key": provider.env.get("LITELLM_API_KEY"), "url": provider.env.get("LITELLM_BASE_URL"),
            "headers": json.loads(headers) if headers else None}


AGENT_CASES = {
    "configured": (None, None, {"key": "k", "url": MODEL, "headers": HEADERS}),
    "passed-configured": ({"LITELLM_BASE_URL": MODEL}, None, {"key": "k", "url": MODEL, "headers": HEADERS}),
    "passed-configured-slash": ({"LITELLM_BASE_URL": MODEL + "/"}, None,
                                {"key": "k", "url": MODEL + "/", "headers": HEADERS}),
    "passed-configured-without-v1": ({"LITELLM_BASE_URL": "http://localhost:4000"}, None,
                                     {"key": "k", "url": "http://localhost:4000", "headers": None}),
    "passed-configured-upper-case": ({"LITELLM_BASE_URL": "HTTP://LOCALHOST:4000/v1"}, None,
                                     {"key": "k", "url": "HTTP://LOCALHOST:4000/v1", "headers": None}),
    "passed-elsewhere-leaks-the-key": ({"LITELLM_BASE_URL": ELSEWHERE}, None,
                                       {"key": "k", "url": ELSEWHERE, "headers": None}),
    "passed-elsewhere-with-its-key": ({"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "theirs"}, None,
                                      {"key": "theirs", "url": ELSEWHERE, "headers": None}),
    "own-elsewhere-leaks-the-key": (None, {"LITELLM_BASE_URL": ELSEWHERE},
                                    {"key": "k", "url": ELSEWHERE, "headers": None}),
    "own-elsewhere-with-its-key": (None, {"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "own"},
                                   {"key": "own", "url": ELSEWHERE, "headers": None}),
    "passed-key-only": ({"LITELLM_API_KEY": "theirs"}, None, {"key": "theirs", "url": MODEL, "headers": HEADERS}),
}


@pytest.mark.parametrize("passed, own, expected", AGENT_CASES.values(), ids=AGENT_CASES.keys())
def test_an_agent_deploy_sends(monkeypatch, configured, passed, own, expected):
    assert _deploy(monkeypatch, passed, own) == expected


def test_an_agent_that_takes_no_credentials_gets_none(monkeypatch, configured):
    sent = _deploy(monkeypatch, {"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "x"}, None,
                   **{TAKES_CREDENTIALS: False})
    assert sent == {"key": None, "url": None, "headers": None}


GATEWAY = "https://gw.example.com"
CALL_CASES = {
    "configured": ({}, {}, ("file-key", GATEWAY, True)),
    "configured-env-key": ({"LITELLM_API_KEY": "env-key"}, {}, ("env-key", GATEWAY, True)),
    "elsewhere-env-key-leaks": ({"LITELLM_API_KEY": "env-key"}, {"base_override": "https://elsewhere.example.com"},
                                ("env-key", "https://elsewhere.example.com", False)),
    "elsewhere-no-key": ({}, {"base_override": "https://elsewhere.example.com"},
                         (None, "https://elsewhere.example.com", False)),
    "elsewhere-with-its-key": ({}, {"base_override": "https://elsewhere.example.com", "default_api_key": "theirs"},
                               ("theirs", "https://elsewhere.example.com", False)),
    "configured-with-v1": ({}, {"base_override": GATEWAY + "/v1"}, (None, GATEWAY + "/v1", False)),
}


@pytest.mark.parametrize("env, call, expected", CALL_CASES.values(), ids=CALL_CASES.keys())
def test_a_model_call_core_makes_sends(monkeypatch, env, call, expected):
    for name in ("LITELLM_BASE_URL", "LITELLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = ModelConfig(base_url=GATEWAY, api_key="file-key", headers={"X-Bill-To": "acme"})
    resolved = config.resolve_call("m", secret_resolver=lambda key: None, **call)
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


def _verifier_env(monkeypatch, env_vars: dict) -> dict:
    """The model variables a unit-test verifier's command ran with."""
    sandbox = _VerifierSandbox()
    provider = types.SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=sandbox))
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)
    monkeypatch.setattr(RunContainerUnitTestsVerifierTaskStep, "_upload_text_artifact", staticmethod(
        lambda text, artifact_id, description, object_url: types.SimpleNamespace(
            id=artifact_id, version=1, object_url=object_url)))
    context = TaskStepContext()
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    context.deployed_sandboxes.append(
        DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="modal_vm"))
    asyncio.run(RunContainerUnitTestsVerifierTaskStep(
        id="tests", version=None, sandbox_name="h", container_name="c", command="true", env_vars=env_vars,
    ).execute(context))
    [run] = [c for c in sandbox.commands if "timeout --kill-after" in c]
    words = shlex.split(run)
    pairs = dict(words[i + 1].split("=", 1) for i, word in enumerate(words) if word == "-e")
    return {name: pairs.get(name) for name in ("LITELLM_API_KEY", "LITELLM_BASE_URL", "ANTHROPIC_API_KEY",
                                               "ANTHROPIC_BASE_URL")}


VERIFIER_CASES = {
    "configured": ({}, {"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": MODEL, "ANTHROPIC_API_KEY": "k",
                        "ANTHROPIC_BASE_URL": MODEL}),
    "elsewhere-leaks-the-key": ({"LITELLM_BASE_URL": ELSEWHERE},
                                {"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": ELSEWHERE, "ANTHROPIC_API_KEY": "k",
                                 "ANTHROPIC_BASE_URL": MODEL}),
    "elsewhere-with-its-key": ({"LITELLM_BASE_URL": ELSEWHERE, "LITELLM_API_KEY": "theirs"},
                               {"LITELLM_API_KEY": "theirs", "LITELLM_BASE_URL": ELSEWHERE, "ANTHROPIC_API_KEY": "k",
                                "ANTHROPIC_BASE_URL": MODEL}),
}


@pytest.mark.parametrize("env_vars, expected", VERIFIER_CASES.values(), ids=VERIFIER_CASES.keys())
def test_a_unit_test_verifier_runs_its_tests_with(monkeypatch, configured, env_vars, expected):
    assert _verifier_env(monkeypatch, env_vars) == expected
