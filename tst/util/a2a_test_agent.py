"""Build and register the echo agent in ``tst/data/a2a_agent`` as an A2A agent.

The build context is staged with this checkout's agentenv-protocol package, so the image
always speaks the protocol version under test; ``build_or_reuse`` content-addresses it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import DockerImageArtifact
from tst.util.image_cache import build_or_reuse

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "tst" / "data" / "a2a_agent"
PROTOCOL_DIR = REPO_ROOT / "packages" / "agentenv-protocol"
IMAGE_TAG = "agentenv-echo-agent"

_PROTOCOL_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples", "*.egg-info")


def build_test_agent_image(artifact_id: str) -> DockerImageArtifact:
    with tempfile.TemporaryDirectory() as staged:
        context = Path(staged)
        shutil.copytree(AGENT_DIR, context, dirs_exist_ok=True)
        shutil.copytree(PROTOCOL_DIR, context / "agentenv-protocol", ignore=_PROTOCOL_IGNORE)
        return build_or_reuse(
            artifact_id=artifact_id,
            description="Echo A2A agent for the integration suites",
            dockerfile=context / "Dockerfile",
            context=context,
            tag=IMAGE_TAG,
        )


# The echo agent calls no model, so it declares the model variables itself instead of having
# A2AAgent.deploy resolve them from a [model] section the profile may not have.
_NO_MODEL_ENV = {"LITELLM_API_KEY": "unused", "LITELLM_BASE_URL": "http://unused.invalid"}


def put_test_agent(agent_id: str) -> A2AAgent:
    return A2AAgent.put(
        id=agent_id,
        docker_image_artifact=build_test_agent_image(f"{agent_id}-image"),
        default_env_vars=_NO_MODEL_ENV,
    )
