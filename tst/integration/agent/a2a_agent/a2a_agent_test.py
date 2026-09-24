"""Integration test suite for A2A agent validation.

Registers the echo agent (tst/data/a2a_agent) as a test A2A agent, then runs the validation
pipeline and the sandbox-provider deploy paths against it.

Takes ~5-10 minutes (image build + deploy env + deploy agent + validations).
"""

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent
from tst.util.a2a_test_agent import put_test_agent
from tst.util.capabilities import skip_without_model_endpoint, skip_without_remote_sandbox

# Every test in this module deploys a real A2A agent on a sandbox VM
# (~3-5 min each). Mark them all as slow so `make int-test-fast` skips.
pytestmark = [pytest.mark.int_test_slow]

TEST_AGENT_ID = "test-a2a-agent"


@pytest.fixture(scope="module")
def a2a_agent() -> A2AAgent:
    return put_test_agent(TEST_AGENT_ID)


@pytest.mark.asyncio
@pytest.mark.integration
# A2AAgentValidator stores its artefacts in S3; the profiles that have it are the model-endpoint ones.
@skip_without_model_endpoint()
@pytest.mark.xfail(
    reason=(
        "The echo agent has no model or tools: the MCP check needs a tool call and "
        "the snapshot extension is not implemented."
    ),
    strict=False,
)
async def test_a2a_agent_validate(a2a_agent):
    """Full A2A agent validation should pass for the test agent."""
    verifications = await a2a_agent.validate(on_progress=print)

    # Agent card should be accessible
    card_result = verifications.get("a2a_agent_card", {})
    validated_card = card_result.get("validated_agent_card", {})
    assert validated_card.get("accessible") is True

    # Core protocol should work
    protocol = verifications.get("a2a_core_protocol", {})
    assert protocol.get("message/send", {}).get("supported") is True
    assert protocol.get("tasks/get", {}).get("supported") is True

    # MCP extension should pass
    mcp_result = verifications.get("a2a_agent_mcp", {})
    assert mcp_result.get("passed") is True
    assert mcp_result.get("tools_invoked", 0) > 0

    # Trajectory extension should work (at least inline)
    traj_result = verifications.get("a2a_trajectory", {})
    assert traj_result.get("inline") is True

    # Skill-config extension should work (at least inline)
    skill_result = verifications.get("a2a_skill_config", {})
    assert skill_result.get("inline") is True
    assert skill_result.get("list") is True

    # Snapshot extension should pass end-to-end (save → load on a fresh agent → recall)
    snapshot_result = verifications.get("a2a_snapshot", {})
    assert snapshot_result.get("extension_advertised") is True
    assert snapshot_result.get("save") is True
    assert snapshot_result.get("load") is True
    assert snapshot_result.get("recall") is True

    # Re-fetch agent to verify metadata was persisted
    agent = A2AAgent.get(TEST_AGENT_ID)
    metadata = agent.metadata

    # validated_agent_card persisted
    assert metadata["validated_agent_card"]["accessible"] is True

    # validated_a2a_protocol persisted
    assert metadata["validated_a2a_protocol"]["message/send"]["supported"] is True
    assert metadata["validated_a2a_protocol"]["tasks/get"]["supported"] is True

    # validated_a2a_extensions persisted
    ext = metadata["validated_a2a_extensions"]
    assert ext[A2AAgent.EXT_MCP_CONFIG]["supported"] is True
    assert ext[A2AAgent.EXT_TRAJECTORY]["supported"] is True
    assert ext[A2AAgent.EXT_SKILL_CONFIG]["supported"] is True
    assert ext[A2AAgent.EXT_SNAPSHOT]["supported"] is True

    # validated_data_extensions persisted
    assert metadata["validated_data_extensions"]["tool_call_count"]["supported"] is True



@pytest.mark.asyncio
@pytest.mark.integration
@skip_without_remote_sandbox("modal")
async def test_a2a_agent_deploy_via_modal(a2a_agent):
    """Deploy the test A2A agent through ModalSandboxProvider and hit /.well-known/agent.json."""
    from agent_env.providers import ModalSandboxProvider, reset_agent_sandbox_provider, set_agent_sandbox_provider

    set_agent_sandbox_provider(ModalSandboxProvider())

    deployed = None
    try:
        deployed = await a2a_agent.deploy(ttl_seconds=900)
        assert deployed.a2a_url.startswith("https://")
        assert deployed.sandbox_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{deployed.a2a_url}/.well-known/agent.json")
            assert resp.status_code == 200, resp.text
            card = resp.json()
            assert card.get("name")
    finally:
        if deployed is not None:
            try:
                sandbox = await ModalSandboxProvider().get_sandbox(deployed.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                print(f"Cleanup warning: {e}")
        reset_agent_sandbox_provider()


@pytest.mark.asyncio
@pytest.mark.integration
@skip_without_remote_sandbox("modal_vm")
async def test_a2a_agent_deploy_via_modal_vm(a2a_agent):
    """Deploy the test A2A agent on a Modal VM (modal_vm) and hit /.well-known/agent.json.

    VM-mode twin of test_a2a_agent_deploy_via_modal: exercises the agent deploy path in
    VM mode, where create_sandbox returns a bare VM and A2AAgent.deploy loads + runs the
    agent image inside it (the mode == "vm" branch), rather than a pre-running container.
    """
    from agent_env.providers import ModalVmSandboxProvider, reset_agent_sandbox_provider, set_agent_sandbox_provider

    set_agent_sandbox_provider(ModalVmSandboxProvider())

    deployed = None
    try:
        deployed = await a2a_agent.deploy(ttl_seconds=900)
        assert deployed.a2a_url.startswith("https://")
        assert deployed.sandbox_id
        assert deployed.sandbox_type == "modal_vm", f"expected sandbox_type=modal_vm, got {deployed.sandbox_type!r}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{deployed.a2a_url}/.well-known/agent.json")
            assert resp.status_code == 200, resp.text
            card = resp.json()
            assert card.get("name")
    finally:
        if deployed is not None:
            try:
                sandbox = await ModalVmSandboxProvider().get_sandbox(deployed.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                print(f"Cleanup warning: {e}")
        reset_agent_sandbox_provider()


@pytest.mark.asyncio
@pytest.mark.integration
@skip_without_remote_sandbox("modal")
async def test_a2a_agent_deploy_chain_falls_back_to_modal(a2a_agent):
    """Chain [failing primary, Modal]: the stubbed first provider always fails, so the deploy falls through to Modal."""
    from agent_env.providers import (
        ChainedSandboxProvider,
        ModalSandboxProvider,
        SandboxProvider,
        reset_agent_sandbox_provider,
        set_agent_sandbox_provider,
    )

    class _AlwaysFailingProvider(SandboxProvider):
        async def create_sandbox(self, **kwargs):
            raise RuntimeError("simulated primary provisioning failure")

    failing_primary = _AlwaysFailingProvider()
    modal = ModalSandboxProvider()
    set_agent_sandbox_provider(ChainedSandboxProvider([failing_primary, modal]))

    deployed = None
    try:
        deployed = await a2a_agent.deploy(ttl_seconds=900)
        assert deployed.a2a_url.startswith("https://")
        assert deployed.sandbox_id
        assert deployed.sandbox_type == "modal", f"expected fallback to Modal, got {deployed.sandbox_type!r}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{deployed.a2a_url}/.well-known/agent.json")
            assert resp.status_code == 200, resp.text
            card = resp.json()
            assert card.get("name")
    finally:
        if deployed is not None:
            try:
                sandbox = await ModalSandboxProvider().get_sandbox(deployed.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                print(f"Cleanup warning: {e}")
        reset_agent_sandbox_provider()


@pytest.mark.asyncio
@pytest.mark.integration
@skip_without_remote_sandbox("modal_vm")
async def test_a2a_agent_deploy_chain_falls_back_to_modal_vm(a2a_agent):
    """Chain [failing primary, ModalVm]: the stubbed first provider always fails, so the deploy falls
    through to modal_vm — verifies the bare-VM agent deploy resolves through ChainedSandboxProvider."""
    from agent_env.providers import (
        ChainedSandboxProvider,
        ModalVmSandboxProvider,
        SandboxProvider,
        reset_agent_sandbox_provider,
        set_agent_sandbox_provider,
    )

    class _AlwaysFailingProvider(SandboxProvider):
        async def create_sandbox(self, **kwargs):
            raise RuntimeError("simulated primary provisioning failure")

    modal_vm = ModalVmSandboxProvider()
    set_agent_sandbox_provider(ChainedSandboxProvider([_AlwaysFailingProvider(), modal_vm]))

    deployed = None
    try:
        deployed = await a2a_agent.deploy(ttl_seconds=900)
        assert deployed.a2a_url.startswith("https://")
        assert deployed.sandbox_id
        assert deployed.sandbox_type == "modal_vm", f"expected fallback to modal_vm, got {deployed.sandbox_type!r}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{deployed.a2a_url}/.well-known/agent.json")
            assert resp.status_code == 200, resp.text
            card = resp.json()
            assert card.get("name")
    finally:
        if deployed is not None:
            try:
                sandbox = await ModalVmSandboxProvider().get_sandbox(deployed.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                print(f"Cleanup warning: {e}")
        reset_agent_sandbox_provider()
