"""Integration test: deploy_agent(enable_docker=True) gives the agent working Docker.

Provisions a real agent sandbox on dev, prompts the agent to run a Docker command
inside its container, and asserts it succeeds — the behavior the unit tests can't
reach (no real VM/daemon). This is what catches regressions in the threading or the
container launch (both of which broke during initial development).

Prereq: the deployed agent image must include the Docker CLI (gateway Dockerfile,
rebuilt + republished). Once the default a2a agent image ships the CLI this runs
against it as-is; until then, point it at a docker-enabled agent via
ENABLE_DOCKER_TEST_A2A_AGENT_ID.

Run (~3-5 min). Use a venv with agent-env's *pinned* deps (the repo venv) — an
unpinned sandbox SDK falls through to a non-Docker fallback sandbox provider:
    [ENABLE_DOCKER_TEST_A2A_AGENT_ID=<docker-enabled a2a-agent id>] \
        venv/bin/python -m pytest tst/integration/task_step/test_enable_docker_e2e.py -v -m integration --log-cli-level=INFO
"""
from __future__ import annotations

import logging
import os

import pytest

from agent_env.task import Task
from agent_env.task_step.context import TaskStepContext
from tst.util.capabilities import skip_without_model_endpoint

# Provisions a real agent sandbox with Docker enabled (~3-5 min).
# Every test here drives an agent (deploy_agent / prompt_agent / install_agent),
# so it needs a model endpoint the resolved config may not provide.
pytestmark = [pytest.mark.int_test_slow, skip_without_model_endpoint()]

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enable_docker_gives_agent_working_docker():
    # This test only succeeds when the deployed agent's container image bundles
    # the Docker CLI. Until the default a2a agent ships the CLI, callers must
    # point at a docker-enabled agent image via ENABLE_DOCKER_TEST_A2A_AGENT_ID.
    # Without it, the agent's shell can't find `docker` and the test would
    # always fail — skip rather than reporting a stale "broken" result.
    agent_id = os.getenv("ENABLE_DOCKER_TEST_A2A_AGENT_ID")
    if not agent_id:
        pytest.skip("Set ENABLE_DOCKER_TEST_A2A_AGENT_ID to an a2a-agent ID whose image bundles the Docker CLI (default agent image does not).")

    deploy_step = {
        "type": "deploy_agent",
        "id": "deploy-agent",
        "agent_name": "default-agent",
        "enable_docker": True,
        "a2a_agent_id": agent_id,
    }

    task = Task.from_dict({
        "id": "enable-docker-smoke",
        "steps": [
            deploy_step,
            {
                "type": "prompt_agent",
                "id": "prompt",
                "agent_name": "default-agent",
                "prompt": (
                    "Run `docker run --rm public.ecr.aws/docker/library/hello-world`. "
                    "Then run `test -S /var/run/docker.sock; echo HOSTSOCK_RC=$?`. "
                    "Report both outputs verbatim."
                ),
                "depends_on": [{"task_step_id": "deploy-agent"}],
            },
        ],
    })

    context = await task.run()
    try:
        assert context.prompt_responses, "prompt_agent produced no response"
        response = context.prompt_responses[0].response or ""
        # Pull from the public ECR mirror, not Docker Hub — the sandbox VM's shared
        # NAT egress trips Docker Hub's unauthenticated pull rate limit.
        assert "Hello from Docker" in response, (
            f"agent could not run docker in its container:\n{response[:600]}"
        )
        # No access to the VM host socket (the isolation this design provides):
        # `test -S` returns non-zero, so HOSTSOCK_RC=1. The token comes from the
        # runtime exit code, not the command text, so it can't false-match on the
        # agent echoing the command back.
        assert "HOSTSOCK_RC=1" in response and "HOSTSOCK_RC=0" not in response, (
            f"VM host Docker socket is exposed to the agent:\n{response[:600]}"
        )
    finally:
        await _terminate_agent_sandboxes(context)


async def _terminate_agent_sandboxes(context: TaskStepContext) -> None:
    from agent_env.providers.sandbox_provider import get_agent_sandbox_provider

    for agent in context.deployed_agents:
        if not agent.sandbox_id:
            continue
        try:
            sandbox = await get_agent_sandbox_provider().get_sandbox(agent.sandbox_id)
            await sandbox.terminate()
            logger.info(f"terminated agent sandbox {agent.sandbox_id}")
        except Exception as e:
            logger.warning(f"agent sandbox {agent.sandbox_id} cleanup failed (TTL will reap): {e}")
