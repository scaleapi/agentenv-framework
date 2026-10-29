"""The agents bundle's proxy: its card, how it installs on a VM's host, and how the episode writes it."""

from __future__ import annotations

import re
import subprocess

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.bundle.installed import write_installed
from agent_env.task.episode.steps import PROXY_AGENT_ID
from agent_env.task_step.task_steps.install_agent import _build_param_resolvers

from tst.unit.agents.proxy.agent import AGENT_DIR, a2a_server


def _extensions() -> dict:
    card = a2a_server.AGENT_CARD.model_dump(mode="json", exclude_none=True, by_alias=True)
    return {ext["uri"]: ext for ext in card["capabilities"]["extensions"]}


def test_the_card_offers_the_proxy_and_an_install_on_a_host_or_into_a_container():
    extensions = _extensions()
    assert {"urn:agentenv:proxy/v1", "urn:agentenv:mcp-config/v1", "urn:agentenv:trajectory/v1",
            "urn:agentenv:install/v1"} <= set(extensions)
    params = extensions["urn:agentenv:install/v1"]["params"]
    assert params["install_commands_host"] and params["install_commands"]


@pytest.mark.parametrize(("commands", "required", "container", "script"), [
    ("install_commands_host", "required_params_host", None,
     "bash {work_dir}/agent/install.sh host {agent_name} {a2a_port} {host_workdir} {host_user}"),
    ("install_commands", "required_params", "task", "bash {work_dir}/agent/install.sh container {container} {a2a_port}"),
])
def test_each_install_asks_only_for_params_install_agent_supplies_and_runs_the_folders_script(commands, required,
                                                                                              container, script):
    params = _extensions()["urn:agentenv:install/v1"]["params"]
    supplied = set(_build_param_resolvers(container_name=container, work_dir="/w", agent_ctx_tar="/w/a.tgz",
                                          a2a_port=8000, config=None, agent_name="a"))
    placeholders = {name for command in params[commands] for name in re.findall(r"\{(\w+)\}", command)}
    assert placeholders == set(params[required]) and placeholders <= supplied
    assert params[commands][-1] == script


def test_the_install_script_parses_and_the_image_keeps_it_in_its_build_context():
    subprocess.run(["bash", "-n", str(AGENT_DIR / "install.sh")], check=True)
    assert "COPY a2a_server.py install.sh ./" in (AGENT_DIR / "Dockerfile").read_text()


def test_the_image_works_in_app_as_agent():
    lines = (AGENT_DIR / "Dockerfile").read_text().splitlines()
    last = {key: next(line.split(maxsplit=1)[1] for line in reversed(lines) if line.startswith(f"{key} "))
            for key in ("USER", "WORKDIR")}
    assert last == {"USER": "agent", "WORKDIR": "/app"}


def test_the_episode_writes_it_from_the_bundle_taking_no_credentials(local_stores, cli_routing):
    version = write_installed(PROXY_AGENT_ID)
    agent = A2AAgent.get(PROXY_AGENT_ID)

    assert (agent.version, agent.metadata) == (version, {"takes_credentials": False})
    assert agent.docker_image_artifact.context_only
    assert write_installed(PROXY_AGENT_ID) == version
