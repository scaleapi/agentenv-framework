"""VMImageArtifact.sandbox_type is additive: artifacts written before it keep the old behaviour."""

from __future__ import annotations

from agent_env.artifact import VMImageArtifact


def test_sandbox_type_defaults_to_none_and_round_trips():
    # None is what makes CuaEnv fall through to the configured default, so every
    # pre-existing artifact deploys exactly as it did before this field existed.
    legacy = VMImageArtifact(id="cua-vm-test", version=1, description="test", ecr_url="ecr/img:1")
    assert legacy.sandbox_type is None

    pinned = VMImageArtifact(**{**legacy.model_dump(), "sandbox_type": "modal"})
    assert pinned.sandbox_type == "modal"
