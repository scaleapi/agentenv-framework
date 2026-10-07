import asyncio
import json
import logging
import subprocess
import uuid
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Module-scope fixtures build 5-7 docker images + push to ECR, and every test
# deploys + drives a real sandbox VM.
pytestmark = [pytest.mark.int_test_slow]

from agent_env.artifact import FileArtifact, FileArtifactUniverse, EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.env import Env, GatewayEnv, MCPServerEnv, MultiEnv
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.config import get_config
from agent_env.task import Task, TaskStepStatus
from agent_env.task_step import AddSkillsTaskStep, BuildMcpCliTaskStep, DeployAgentTaskStep, DeployEnvTaskStep, EnvOutcomeVerifierTaskStep, LoadArtifactTaskStep, PromptAgentTaskStep, RubricsVerifierTaskStep, TaskStep, TaskStepContext, VerifyMCPToolSchemaTaskStep
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.a2a_agent import A2AAgent, conversation_store
from agent_env.a2a_agent.object_transfer import trajectory_mode
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_agent_sandbox_provider

from tst.task import journal_invariants as journal
from tst.util.a2a_test_agent import put_test_agent
from tst.util.capabilities import skip_without_default_a2a_agent, skip_without_model_endpoint, skip_without_remote_sandbox
from tst.util.image_cache import build_or_reuse

logger = logging.getLogger(__name__)



REPO_ROOT = Path(__file__).resolve().parents[4]
TST_DATA_DIR = REPO_ROOT / "tst" / "data"
ENV_DIR = REPO_ROOT / "src" / "agent_env" / "env"
ENVS_DIR = ENV_DIR / "envs"


TEST_RUN_SUFFIX = uuid.uuid4().hex[:8]


def _run_id(base: str) -> str:
    return f"{base}-{TEST_RUN_SUFFIX}"


@dataclass
class DockerImageConfig:
    name: str
    dockerfile: Path
    tag: str
    context: Path
    service_name: str


GATEWAY_IMAGE = DockerImageConfig(
    name="gateway",
    dockerfile=ENV_DIR / "gateway" / "Dockerfile",
    tag="env-gateway",
    context=ENV_DIR,
    service_name="gateway",
)

SERVICE_DB_IMAGE = DockerImageConfig(
    name="service-db",
    dockerfile=ENVS_DIR / "service_db" / "Dockerfile",
    tag="agent-env-service-db",
    context=ENVS_DIR / "service_db",
    service_name="servicedb",
)

DB_WEB_IMAGE = DockerImageConfig(
    name="db-web",
    dockerfile=ENVS_DIR / "service_db" / "Dockerfile.db-web",
    tag="agent-env-db-web",
    context=ENVS_DIR / "service_db",
    service_name="pgweb",
)

EMAIL_MCP_IMAGE = DockerImageConfig(
    name="email",
    dockerfile=TST_DATA_DIR / "email_mcp" / "Dockerfile",
    tag="mcp-email",
    context=TST_DATA_DIR,
    service_name="email",
)

SLACK_MCP_IMAGE = DockerImageConfig(
    name="slack",
    dockerfile=TST_DATA_DIR / "slack_mcp" / "Dockerfile",
    tag="mcp-slack",
    context=TST_DATA_DIR,
    service_name="slack",
)

def _scheme(url: str) -> str:
    return urlparse(url).scheme

@pytest.fixture(scope="module")
def gateway_env() -> GatewayEnv:
    img = GATEWAY_IMAGE
    artifact = build_or_reuse(
        artifact_id=f"task-step-test-gateway-{img.name}",
        description=f"Gateway {img.name} server image",
        dockerfile=img.dockerfile,
        context=img.context,
        tag=img.tag,
    )
    env = GatewayEnv.put(id=_run_id("task-step-test-gateway"), docker_image_artifact=artifact)
    logger.info(f"Created GatewayEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def service_db_env() -> ServiceDBEnv:
    img = SERVICE_DB_IMAGE
    db_artifact = build_or_reuse(
        artifact_id=f"task-step-test-service-db-{img.name}",
        description=f"ServiceDB {img.name} image",
        dockerfile=img.dockerfile,
        context=img.context,
        tag=img.tag,
    )

    db_web = DB_WEB_IMAGE
    db_web_artifact = build_or_reuse(
        artifact_id="task-step-test-db-web-service-db",
        description="db-web lightweight web UI for database inspection",
        dockerfile=db_web.dockerfile,
        context=db_web.context,
        tag=db_web.tag,
    )

    db_mcp_dockerfile = ENVS_DIR / "service_db" / "Dockerfile.db-mcp"
    db_mcp_artifact = build_or_reuse(
        artifact_id="task-step-test-db-mcp-service-db",
        description="db-mcp PostgreSQL MCP server for direct DB access",
        dockerfile=db_mcp_dockerfile,
        context=db_mcp_dockerfile.parent,
        tag="agent-env-db-mcp",
    )

    env = ServiceDBEnv.put(
        id=_run_id("task-step-test-service-db"),
        db_docker_image_artifact=db_artifact,
        db_web_docker_image_artifact=db_web_artifact,
        db_mcp_docker_image_artifact=db_mcp_artifact,
    )
    logger.info(f"Created ServiceDBEnv: {env.id} version={env.version}")
    return env


# Function-scoped so the ids survive the per-test config reset; the env fixtures it
# depends on stay module-scoped, so the deploys still happen once.
@pytest.fixture(autouse=True)
def configure_default_envs(gateway_env, service_db_env):
    config = get_config()
    config.default_gateway_env_id = gateway_env.id
    config.default_service_db_env_id = service_db_env.id
    yield


@pytest.fixture(scope="module")
def mcp_server_env() -> MCPServerEnv:
    img = EMAIL_MCP_IMAGE
    artifact = build_or_reuse(
        artifact_id=f"task-step-test-mcp-{img.name}",
        description=f"MCP {img.name} server image",
        dockerfile=img.dockerfile,
        context=img.context,
        tag=img.tag,
    )
    env = MCPServerEnv.put(
        id=_run_id(f"task-step-test-mcp-server-{img.name}"),
        docker_image_artifact=artifact,
        environment_name=img.service_name,
    )
    logger.info(f"Created MCPServerEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def slack_mcp_server_env() -> MCPServerEnv:
    img = SLACK_MCP_IMAGE
    artifact = build_or_reuse(
        artifact_id=f"task-step-test-mcp-{img.name}",
        description=f"MCP {img.name} server image",
        dockerfile=img.dockerfile,
        context=img.context,
        tag=img.tag,
    )
    env = MCPServerEnv.put(
        id=_run_id(f"task-step-test-mcp-server-{img.name}"),
        docker_image_artifact=artifact,
        environment_name=img.service_name,
    )
    logger.info(f"Created MCPServerEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def multi_env(mcp_server_env, slack_mcp_server_env) -> MultiEnv:
    env = MultiEnv.put(
        id=_run_id("task-step-test-multi-slack-email"),
        mcp_server_envs=[slack_mcp_server_env, mcp_server_env],
    )
    logger.info(f"Created MultiEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def email_service_artifact() -> EnvironmentArtifact:
    file_artifact = FileArtifact.put(
        id=_run_id("task-step-test-email-data"),
        description="Generated email data with at least 100 emails",
        file_path=str(TST_DATA_DIR / "email_mcp" / "generated_data.json"),
    )
    return EnvironmentArtifact.put(
        id=_run_id("task-step-test-email-service-data"),
        environment_name="email",
        file_artifact=file_artifact,
    )


@pytest.fixture(scope="module")
def slack_service_artifact() -> EnvironmentArtifact:
    file_artifact = FileArtifact.put(
        id=_run_id("task-step-test-slack-data"),
        description="Generated slack data with users, channels, and messages",
        file_path=str(TST_DATA_DIR / "slack_mcp" / "generated_data.json"),
    )
    return EnvironmentArtifact.put(
        id=_run_id("task-step-test-slack-service-data"),
        environment_name="slack",
        file_artifact=file_artifact,
    )


@pytest.fixture(scope="module")
def environment_universe_artifact(slack_service_artifact, email_service_artifact) -> EnvironmentUniverseArtifact:
    return EnvironmentUniverseArtifact.put(
        id=_run_id("task-step-test-universe"),
        environment_artifacts=[slack_service_artifact, email_service_artifact],
    )


@pytest.fixture(scope="module")
def a2a_agent() -> A2AAgent:
    """The configured default agent: what a deploy_agent step without an id deploys."""
    return A2AAgent.get(get_config().get_default_a2a_agent_id())


@pytest.fixture(scope="module")
def echo_agent() -> A2AAgent:
    return put_test_agent("task-step-test-echo-agent")


@pytest.fixture(params=[
    pytest.param("default", id="default"),
    pytest.param(
        "modal",
        id="modal",
        marks=skip_without_remote_sandbox("modal"),
    ),
    pytest.param(
        "modal_vm",
        id="modal_vm",
        marks=skip_without_remote_sandbox("modal_vm"),
    ),
    pytest.param(
        "sail_vm",
        id="sail_vm",
        marks=skip_without_remote_sandbox("sail_vm"),
    ),
])
def sandbox_provider(request):
    from agent_env.providers import (
        ModalSandboxProvider,
        ModalVmSandboxProvider,
        build_sandbox_provider,
        reset_agent_sandbox_provider,
        reset_env_sandbox_provider,
        reset_sandbox_provider,
        set_agent_sandbox_provider,
        set_env_sandbox_provider,
        set_sandbox_provider,
    )

    if request.param == "modal":
        set_sandbox_provider(ModalSandboxProvider())
        set_env_sandbox_provider(ModalSandboxProvider())
        set_agent_sandbox_provider(ModalSandboxProvider())
    elif request.param == "modal_vm":
        set_sandbox_provider(ModalVmSandboxProvider())
        set_env_sandbox_provider(ModalVmSandboxProvider())
        set_agent_sandbox_provider(ModalVmSandboxProvider())
    elif request.param == "sail_vm":
        set_sandbox_provider(build_sandbox_provider("sail_vm"))
        set_env_sandbox_provider(build_sandbox_provider("sail_vm"))
        set_agent_sandbox_provider(build_sandbox_provider("sail_vm"))
    try:
        yield request.param
    finally:
        reset_sandbox_provider()
        reset_env_sandbox_provider()
        reset_agent_sandbox_provider()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.dependency(name="test_task_steps_e2e")
@skip_without_model_endpoint()
@skip_without_default_a2a_agent()
async def test_task_steps_e2e(sandbox_provider, multi_env, email_service_artifact, environment_universe_artifact, a2a_agent):
    """Test full task step pipeline: deploy env → load service artifact → load universe artifact → deploy agent."""

    # 1. Put DeployEnvTaskStep into control plane
    deploy_step = DeployEnvTaskStep.put(
        id=_run_id("task-step-test-deploy-multi"),
        env_id=multi_env.id,
        env_version=multi_env.version,
    )
    assert deploy_step.version is not None
    assert deploy_step.type == "deploy_env"
    logger.info(f"Created DeployEnvTaskStep: id={deploy_step.id} version={deploy_step.version}")

    # 2. Get task step from control plane (via base class for type dispatch)
    retrieved = TaskStep.get(deploy_step.id, deploy_step.version)
    assert isinstance(retrieved, DeployEnvTaskStep)
    assert retrieved.id == deploy_step.id
    assert retrieved.version == deploy_step.version
    assert retrieved.env_id == multi_env.id
    assert retrieved.env_version == multi_env.version
    assert retrieved.ttl_seconds == TaskStep.DEFAULT_TTL_SECONDS
    assert retrieved.type == "deploy_env"
    logger.info(f"Retrieved DeployEnvTaskStep: id={retrieved.id} env_id={retrieved.env_id}")

    # 3. Execute deploy step
    context = await retrieved.execute(TaskStepContext())
    assert len(context.deployed_envs) == 1
    deployed = context.deployed_envs[0]
    assert deployed.env_id == multi_env.id
    assert deployed.env_version == multi_env.version
    assert deployed.gateway_url.startswith("http")
    logger.info(f"DeployEnvTaskStep.execute() returned context with gateway_url={deployed.gateway_url}")

    # 4. Verify MCP tools available with empty databases
    async def assert_empty_databases(session, tools_result, tool_names):
        assert "list_emails" in tool_names, f"Expected list_emails tool, got {tool_names}"
        assert "channels_list" in tool_names, f"Expected channels_list tool, got {tool_names}"

        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"BEFORE load - list_emails result: {content_text[:300]}...")
        assert "total_emails\": 0" in content_text or '"total_emails": 0' in content_text, \
            f"Expected empty email database, got {content_text}"

        result = await session.call_tool("channels_list", {"channel_types": "public_channel"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"BEFORE load - channels_list result: {content_text[:300]}...")
        assert '"channels": []' in content_text, f"Expected empty slack channels, got {content_text}"

    await _verify_list_tools(mcp_url=deployed.mcp_url, tool_verifier_callback=assert_empty_databases)

    # 5. Put and get LoadArtifactTaskStep for service artifact (email)
    load_service_step = LoadArtifactTaskStep.put(
        id=_run_id("task-step-test-load-email-service"),
        env_id=multi_env.id,
        artifact_id=email_service_artifact.id,
        artifact_version=email_service_artifact.version,
    )
    retrieved_load = TaskStep.get(load_service_step.id, load_service_step.version)
    assert isinstance(retrieved_load, LoadArtifactTaskStep)
    assert retrieved_load.env_id == multi_env.id
    assert retrieved_load.artifacts[0]["id"] == email_service_artifact.id
    assert retrieved_load.type == "load_artifact"
    logger.info(f"Retrieved LoadArtifactTaskStep (service): id={retrieved_load.id}")

    # 6. Execute load service artifact step
    context = await retrieved_load.execute(context)

    # 7. Verify email data loaded
    async def assert_email_data_loaded(session, tools_result, tool_names):
        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"AFTER load service artifact - list_emails result: {content_text[:300]}...")
        assert "alex.chen@techcorp.com" in content_text, \
            f"Expected alex.chen@techcorp.com (generated_data), got {content_text}"

    await _verify_list_tools(mcp_url=deployed.mcp_url, tool_verifier_callback=assert_email_data_loaded)

    # 8. Put and get LoadArtifactTaskStep for universe artifact (slack + email)
    load_universe_step = LoadArtifactTaskStep.put(
        id=_run_id("task-step-test-load-universe"),
        env_id=multi_env.id,
        artifact_id=environment_universe_artifact.id,
        artifact_version=environment_universe_artifact.version,
    )
    retrieved_universe = TaskStep.get(load_universe_step.id, load_universe_step.version)
    assert isinstance(retrieved_universe, LoadArtifactTaskStep)
    assert retrieved_universe.env_id == multi_env.id
    assert retrieved_universe.artifacts[0]["id"] == environment_universe_artifact.id
    logger.info(f"Retrieved LoadArtifactTaskStep (universe): id={retrieved_universe.id}")

    # 9. Execute load universe artifact step
    context = await retrieved_universe.execute(context)

    # 10. Verify all data loaded (both slack and email)
    async def assert_universe_data_loaded(session, tools_result, tool_names):
        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"AFTER load universe - list_emails result: {content_text[:300]}...")
        assert "alex.chen@techcorp.com" in content_text, \
            f"Expected alex.chen@techcorp.com (email generated_data), got {content_text}"

        result = await session.call_tool("channels_list", {"channel_types": "public_channel"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"AFTER load universe - channels_list result: {content_text[:300]}...")
        assert "general" in content_text, \
            f"Expected general channel (slack generated_data), got {content_text}"

    await _verify_list_tools(mcp_url=deployed.mcp_url, tool_verifier_callback=assert_universe_data_loaded)

    # 11. Put and get DeployAgentTaskStep
    deploy_agent_step = DeployAgentTaskStep.put(
        id=_run_id("task-step-test-deploy-agent"),
        env_ids=[multi_env.id],
        a2a_agent_id=a2a_agent.id,
        a2a_agent_version=a2a_agent.version,
    )
    retrieved_agent = TaskStep.get(deploy_agent_step.id, deploy_agent_step.version)
    assert isinstance(retrieved_agent, DeployAgentTaskStep)
    assert retrieved_agent.env_ids == [multi_env.id]
    assert retrieved_agent.agent_name == TaskStep.DEFAULT_AGENT_NAME
    assert retrieved_agent.type == "deploy_agent"
    logger.info(f"Retrieved DeployAgentTaskStep: id={retrieved_agent.id}")

    # 12. Execute deploy agent step
    context = await retrieved_agent.execute(context)
    assert len(context.deployed_agents) == 1
    agent = context.deployed_agents[0]
    assert agent.agent_name == TaskStep.DEFAULT_AGENT_NAME
    assert agent.api_url.startswith("http")
    logger.info(f"DeployAgentTaskStep.execute() returned agent api_url={agent.api_url}")

    # 13. Verify agent health and MCP registration
    async with httpx.AsyncClient() as client:
        health = await client.get(f"{agent.api_url}/health", timeout=15)
        health.raise_for_status()
        assert health.json()["status"] == "ok"
        logger.info("Agent health: ok")

        mcps = await client.get(f"{agent.api_url}/mcps", timeout=15)
        mcps.raise_for_status()
        mcp_servers = mcps.json()["mcp_servers"]
        assert len(mcp_servers) == 1, f"Expected 1 MCP server registered, got {len(mcp_servers)}"
        logger.info(f"Agent MCP servers: {list(mcp_servers.keys())}")

    # 14. Put and get PromptAgentTaskStep
    prompt_step = PromptAgentTaskStep.put(
        id=_run_id("task-step-test-prompt-agent"),
        prompt="Use the list_emails tool to list emails in the INBOX folder. Who sent the email with subject 'Weekly standup reminder'? Reply with just their email address.",
    )
    retrieved_prompt = TaskStep.get(prompt_step.id, prompt_step.version)
    assert isinstance(retrieved_prompt, PromptAgentTaskStep)
    assert retrieved_prompt.prompt_id is not None
    assert retrieved_prompt.agent_name == TaskStep.DEFAULT_AGENT_NAME
    assert retrieved_prompt.timeout_seconds == PromptAgentTaskStep.DEFAULT_PROMPT_TIMEOUT_SECONDS
    assert retrieved_prompt.type == "prompt_agent"
    assert retrieved_prompt.model is None
    assert retrieved_prompt.system_prompt is None
    assert retrieved_prompt.max_turns is None
    assert retrieved_prompt.output_format is None
    logger.info(f"Retrieved PromptAgentTaskStep: id={retrieved_prompt.id} prompt_id={retrieved_prompt.prompt_id}")

    # 14b. Verify optional fields round-trip through put/get
    test_output_format = {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}
    prompt_step_with_opts = PromptAgentTaskStep.put(
        id=_run_id("task-step-test-prompt-agent-opts"),
        prompt="Say hello.",
        model="gemini-2.5-flash",
        system_prompt="You are a helpful assistant.",
        max_turns=10,
        output_format=test_output_format,
    )
    retrieved_opts = TaskStep.get(prompt_step_with_opts.id, prompt_step_with_opts.version)
    assert isinstance(retrieved_opts, PromptAgentTaskStep)
    assert retrieved_opts.model == "gemini-2.5-flash"
    assert retrieved_opts.system_prompt == "You are a helpful assistant."
    assert retrieved_opts.max_turns == 10
    assert retrieved_opts.output_format == test_output_format
    logger.info(f"PromptAgentTaskStep optional fields round-trip: model={retrieved_opts.model} system_prompt={retrieved_opts.system_prompt} max_turns={retrieved_opts.max_turns} output_format={retrieved_opts.output_format}")

    # 15. Execute prompt agent step
    context = await retrieved_prompt.execute(context)
    assert len(context.prompt_responses) == 1
    prompt_response = context.prompt_responses[0]
    assert prompt_response.prompt_id == retrieved_prompt.prompt_id
    assert "sarah.johnson" in prompt_response.response.lower(), \
        f"Expected 'sarah.johnson' in response, got: {prompt_response.response}"
    logger.info(f"PromptAgentTaskStep response: {prompt_response.response[:200]}")

    # 16. Verify trajectory was uploaded to the configured object store
    assert prompt_response.agent_trajectory_s3_uri is not None, "Expected agent_trajectory_s3_uri to be set"
    object_store = get_config().get_object_store()
    assert _scheme(prompt_response.agent_trajectory_s3_uri) == _scheme(object_store.object_url("probe")), \
        f"trajectory URI not on the configured object store: {prompt_response.agent_trajectory_s3_uri}"
    logger.info(f"Trajectory URI: {prompt_response.agent_trajectory_s3_uri}")

    trajectory = json.loads(object_store.get(prompt_response.agent_trajectory_s3_uri))
    assert isinstance(trajectory, list), f"Expected trajectory to be a list, got {type(trajectory)}"
    assert len(trajectory) > 0, "Expected non-empty trajectory"
    logger.info(f"Trajectory has {len(trajectory)} messages")

    # 17. Put and get EnvOutcomeVerifierTaskStep
    verifier_step = EnvOutcomeVerifierTaskStep.put(
        id=_run_id("task-step-test-env-outcome-verifier"),
        env_id=multi_env.id,
        verify_script_file_path=str(TST_DATA_DIR / "test_outcome_verifier.py"),
        score_aggregator="all_pass",
    )
    retrieved_verifier = TaskStep.get(verifier_step.id, verifier_step.version)
    assert isinstance(retrieved_verifier, EnvOutcomeVerifierTaskStep)
    assert retrieved_verifier.env_id == multi_env.id
    assert retrieved_verifier.file_artifact_id is not None
    assert retrieved_verifier.type == "env_outcome_verifier"
    logger.info(f"Retrieved EnvOutcomeVerifierTaskStep: id={retrieved_verifier.id}")

    # 18. Execute env outcome verifier step
    context = await retrieved_verifier.execute(context)
    env_verifier_id = retrieved_verifier.verifier_id
    assert env_verifier_id in context.metadata.get("verifications", {}), \
        f"Expected verifier_id '{env_verifier_id}' in verifications"
    env_vdata = context.metadata["verifications"][env_verifier_id]
    verification_results = env_vdata["results"]
    assert len(verification_results) == 2, f"Expected 2 criteria, got {len(verification_results)}"
    for criterion in verification_results:
        assert criterion["result"] is True, f"Criterion '{criterion['id']}' failed: {criterion['description']}"
    assert env_vdata["score"] == 1.0, f"Expected score 1.0, got {env_vdata['score']}"
    logger.info(f"Verification [{env_verifier_id}] results: {verification_results}, score: {env_vdata['score']}")

    # 19. Put and get RubricsVerifierTaskStep
    rubrics_step = RubricsVerifierTaskStep.put(
        id=_run_id("task-step-test-rubrics-verifier"),
        criteria=[
            {
                "id": "criterion-1",
                "title": "The agent used the list_emails tool to list emails in the INBOX folder",
                "rubric_category": "Tool Selection",
                "rubric_target": "Process/Reasoning",
            },
            {
                "id": "criterion-2",
                "title": "The agent's final response includes the email address sarah.johnson@techcorp.com",
                "rubric_category": "Outcome",
                "rubric_target": "Outcome",
            },
        ],
        prompt_id=retrieved_prompt.prompt_id,
        default_model="claude-sonnet-4-6",
        # Follow the parametrized provider instead of the configured agent default chain: its
        # first leg is unreachable from CI, and the 180s per-provider cap then times out Modal's
        # cold image build. The judge is the only step that escaped the fixture's override.
        judge_sandbox_type=None,
    )
    retrieved_rubrics = TaskStep.get(rubrics_step.id, rubrics_step.version)
    assert isinstance(retrieved_rubrics, RubricsVerifierTaskStep)
    assert retrieved_rubrics.prompt_id == retrieved_prompt.prompt_id
    assert len(retrieved_rubrics.criteria) == 2
    assert retrieved_rubrics.type == "rubrics_verifier"
    logger.info(f"Retrieved RubricsVerifierTaskStep: id={retrieved_rubrics.id}")

    # 20. Execute rubrics verifier step
    context = await retrieved_rubrics.execute(context)
    rubrics_verifier_id = retrieved_rubrics.verifier_id
    assert rubrics_verifier_id in context.metadata.get("verifications", {}), \
        f"Expected verifier_id '{rubrics_verifier_id}' in verifications"
    rubrics_vdata = context.metadata["verifications"][rubrics_verifier_id]
    rubrics_results = rubrics_vdata["results"]
    assert len(rubrics_results) == 2, f"Expected 2 criteria, got {len(rubrics_results)}"
    for result in rubrics_results:
        assert "id" in result, f"Expected 'id' field in result: {result}"
        assert "score" in result, f"Expected 'score' field in result: {result}"
        assert "result" in result, f"Expected 'result' field in result: {result}"
    assert rubrics_vdata["score"] == 1.0, f"Expected score 1.0, got {rubrics_vdata['score']}"
    # Verify env outcome verifier results still exist
    assert env_verifier_id in context.metadata["verifications"], \
        "Env outcome verifier results were overwritten by rubrics verifier"
    logger.info(f"Verification [{rubrics_verifier_id}] results: {rubrics_results}, score: {rubrics_vdata['score']}")

    # 21. Put and get RubricsVerifierTaskStep with agent_name (reusing deployed agent)
    rubrics_with_agent_step = RubricsVerifierTaskStep.put(
        id=_run_id("task-step-test-rubrics-verifier-with-agent"),
        criteria=[
            {
                "id": "criterion-1",
                "title": "The agent used the list_emails tool to list emails in the INBOX folder",
                "rubric_category": "Tool Selection",
                "rubric_target": "Process/Reasoning",
            },
            {
                "id": "criterion-2",
                "title": "The agent's final response includes the email address sarah.johnson@techcorp.com",
                "rubric_category": "Outcome",
                "rubric_target": "Outcome",
            },
        ],
        prompt_id=retrieved_prompt.prompt_id,
        default_model="claude-sonnet-4-6",
        agent_name=TaskStep.DEFAULT_AGENT_NAME,
    )
    retrieved_rubrics_agent = TaskStep.get(rubrics_with_agent_step.id, rubrics_with_agent_step.version)
    assert isinstance(retrieved_rubrics_agent, RubricsVerifierTaskStep)
    assert retrieved_rubrics_agent.agent_name == TaskStep.DEFAULT_AGENT_NAME
    logger.info(f"Retrieved RubricsVerifierTaskStep (with agent): id={retrieved_rubrics_agent.id}")

    # 22. Execute rubrics verifier with reused agent
    context = await retrieved_rubrics_agent.execute(context)
    rubrics_agent_verifier_id = retrieved_rubrics_agent.verifier_id
    assert rubrics_agent_verifier_id in context.metadata.get("verifications", {})
    rubrics_agent_vdata = context.metadata["verifications"][rubrics_agent_verifier_id]
    rubrics_agent_results = rubrics_agent_vdata["results"]
    assert len(rubrics_agent_results) == 2
    for result in rubrics_agent_results:
        assert "id" in result
        assert "score" in result
        assert "result" in result
    # Verify all previous verifier results still exist
    assert env_verifier_id in context.metadata["verifications"]
    assert rubrics_verifier_id in context.metadata["verifications"]
    logger.info(f"Verification [{rubrics_agent_verifier_id}] (with agent) results: {rubrics_agent_results}, score: {rubrics_agent_vdata['score']}")

    if agent.sandbox_id:
        try:
            from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_agent_sandbox_provider

            provider = build_sandbox_provider(agent.sandbox_type) if agent.sandbox_type else get_agent_sandbox_provider()
            agent_sandbox = await provider.get_sandbox(agent.sandbox_id)
            await agent_sandbox.terminate()
        except Exception as e:
            logger.warning(f"Task steps e2e agent cleanup failed for sandbox {agent.sandbox_id}: {e}")

    env = await type(Env.get(deployed.env_id, deployed.env_version)).from_deployed_env(deployed)
    await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_load_universe_on_single_mcp_server_env(sandbox_provider, mcp_server_env, environment_universe_artifact):
    """A single MCPServerEnv loads its own service out of a (multi-service) universe.

    Regression for the prod AttributeError — before the fix, loading an EnvironmentUniverseArtifact
    into a single-server env raised 'MCPServerEnv has no load_environment_universe_artifact'. Here the
    email server loads the email service from the slack+email universe (slack is skipped).
    """
    # 1. Deploy the single email MCP server env
    deploy_step = DeployEnvTaskStep.put(
        id=_run_id("task-step-test-deploy-single-mcp"),
        env_id=mcp_server_env.id,
        env_version=mcp_server_env.version,
    )
    context = await deploy_step.execute(TaskStepContext())
    assert len(context.deployed_envs) == 1
    deployed = context.deployed_envs[0]
    assert deployed.env_id == mcp_server_env.id

    try:
        # 2. Load the slack+email universe into the single email server (slack is skipped)
        load_universe_step = LoadArtifactTaskStep.put(
            id=_run_id("task-step-test-load-universe-single-mcp"),
            env_id=mcp_server_env.id,
            artifact_id=environment_universe_artifact.id,
            artifact_version=environment_universe_artifact.version,
        )
        context = await load_universe_step.execute(context)

        # 3. Verify email data landed via the real MCP endpoint
        async def assert_email_data_loaded(session, tools_result, tool_names):
            result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
            content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
            logger.info(f"AFTER load single-service universe - list_emails: {content_text[:300]}...")
            assert "alex.chen@techcorp.com" in content_text, \
                f"Expected email generated_data, got {content_text}"

        await _verify_list_tools(mcp_url=deployed.mcp_url, tool_verifier_callback=assert_email_data_loaded)
    finally:
        env = await type(Env.get(deployed.env_id, deployed.env_version)).from_deployed_env(deployed)
        await env.close()


async def _agent_sandbox(agent):
    provider = build_sandbox_provider(agent.sandbox_type) if agent.sandbox_type else get_agent_sandbox_provider()
    return await provider.get_sandbox(agent.sandbox_id)


async def _read_in_agent(sandbox, path):
    if isinstance(sandbox, VmSandbox):
        return await sandbox.exec_with_output("docker", "exec", sandbox.container_name, "cat", path)
    return await sandbox.exec_with_output("cat", path)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_load_artifact_stages_files_into_env_and_agent(mcp_server_env, echo_agent, tmp_path):
    """A single file and a file-artifact universe land at ``<destination>/<name>`` in a deployed env
    and in a deployed agent, through the real loaders and sandboxes."""
    suffix = uuid.uuid4().hex[:8]
    expected = {"note.txt": "a single file\n", "a.txt": "a universe member\n", "sub/b.txt": "a nested member\n"}

    def put_file(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected[name])
        return FileArtifact.put(id=_run_id(f"load-files-{path.stem}"), description=name, file_path=str(path))

    note = put_file("note.txt")
    universe = FileArtifactUniverse.put(
        id=_run_id("load-files-universe"), file_artifacts={name: put_file(name) for name in ("a.txt", "sub/b.txt")},
    )
    refs = [{"id": artifact.id, "version": artifact.version} for artifact in (note, universe)]

    context = await DeployEnvTaskStep(
        id=f"load-files-env-{suffix}", version=None, env_id=mcp_server_env.id, env_version=mcp_server_env.version,
    ).execute(TaskStepContext())
    deployed_env = context.deployed_envs[0]
    try:
        await DeployAgentTaskStep(
            id=f"load-files-agent-{suffix}", version=None, env_ids=[mcp_server_env.id],
            a2a_agent_id=echo_agent.id, a2a_agent_version=echo_agent.version,
        ).execute(context)
        agent = context.deployed_agents[0]
        await LoadArtifactTaskStep(
            id=f"load-files-into-env-{suffix}", version=None, env_id=mcp_server_env.id,
            artifacts=refs, destination_path="/app/loaded",
        ).execute(context)
        await LoadArtifactTaskStep(
            id=f"load-files-into-agent-{suffix}", version=None, agent_name=agent.agent_name, artifacts=refs,
        ).execute(context)

        env = await type(Env.get(deployed_env.env_id, deployed_env.env_version)).from_deployed_env(deployed_env)
        agent_sandbox = await _agent_sandbox(agent)
        for name, text in expected.items():
            assert await env._sandbox.exec_with_output("cat", f"/app/loaded/{name}") == (0, text, "")
            assert await _read_in_agent(agent_sandbox, f"/tmp/file_artifacts/{name}") == (0, text, "")
        loaded = context.metadata["loaded_file_artifact_universes"]
        assert [(e["artifact_type"], e["env_id"], e["agent_name"], sorted(e["files"])) for e in loaded] == [
            ("file", mcp_server_env.id, None, ["note.txt"]),
            ("file_artifact_universe", mcp_server_env.id, None, ["a.txt", "sub/b.txt"]),
            ("file", None, agent.agent_name, ["note.txt"]),
            ("file_artifact_universe", None, agent.agent_name, ["a.txt", "sub/b.txt"]),
        ]
    finally:
        if context.deployed_agents:
            try:
                await (await _agent_sandbox(context.deployed_agents[0])).terminate()
            except Exception as e:
                logger.warning(f"load-files test agent cleanup: {e}")
        env = await type(Env.get(deployed_env.env_id, deployed_env.env_version)).from_deployed_env(deployed_env)
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_an_agent_uploads_its_trajectory_through_a_local_grant(mcp_server_env, echo_agent):
    """On the local defaults an agent that takes the object form uploads its own trajectory: its container
    trusts the local transfer CA and reaches the grant server on this host."""
    suffix = uuid.uuid4().hex[:8]
    store = get_config().get_object_store()
    context = await DeployEnvTaskStep(
        id=f"grant-traj-env-{suffix}", version=None, env_id=mcp_server_env.id, env_version=mcp_server_env.version,
    ).execute(TaskStepContext())
    deployed_env = context.deployed_envs[0]
    try:
        await DeployAgentTaskStep(
            id=f"grant-traj-agent-{suffix}", version=None, env_ids=[mcp_server_env.id],
            a2a_agent_id=echo_agent.id, a2a_agent_version=echo_agent.version,
        ).execute(context)
        agent = context.deployed_agents[0]
        get_method, _ = A2AAgent.operation(A2AAgent.find_extension(agent.a2a_card, A2AAgent.EXT_TRAJECTORY), "get")
        assert trajectory_mode(get_method, store, by="task_id", sandbox_type=agent.sandbox_type) == "objects"

        await PromptAgentTaskStep(
            id=f"grant-traj-prompt-{suffix}", version=None, prompt="hello through a grant", timeout_seconds=120,
        ).execute(context)
        uri = context.prompt_responses[-1].agent_trajectory_s3_uri
        assert uri, "the agent did not upload its trajectory through the grant"
        assert "hello through a grant" in store.get(uri).decode()
    finally:
        if context.deployed_agents:
            try:
                await (await _agent_sandbox(context.deployed_agents[0])).terminate()
            except Exception as e:
                logger.warning(f"grant trajectory test agent cleanup: {e}")
        env = await type(Env.get(deployed_env.env_id, deployed_env.env_version)).from_deployed_env(deployed_env)
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
@skip_without_model_endpoint()
@skip_without_default_a2a_agent()
async def test_cli_install_e2e(sandbox_provider, multi_env, a2a_agent):
    """End-to-end: deploy env + agent, build CliArtifact, install into agent, register skill via cli_artifact_ids."""
    from agent_env.a2a_agent.store import get_a2a_agent_instance_store
    from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider

    async def exec_in(sandbox, wrapper_cmd, *cmd):
        # Container mode: the agent IS the sandbox; exec directly. VM mode: docker exec into the named container.
        if sandbox.mode == "container":
            _, stdout, _ = await sandbox.exec_with_output("sh", "-c", " ".join(cmd))
            return stdout
        return await sandbox.exec_script(f"{wrapper_cmd} sh -c {repr(' '.join(cmd))}")

    suffix = uuid.uuid4().hex[:8]

    deploy_env_step = DeployEnvTaskStep(id=f"cli-e2e-deploy-env-{suffix}", version=None, env_id=multi_env.id, env_version=multi_env.version)
    context = await deploy_env_step.execute(TaskStepContext())
    deployed_env = context.deployed_envs[0]
    logger.info(f"CLI e2e: deployed env at {deployed_env.gateway_url}")

    try:
        # env_ids=[] — agent does NOT register the MCP server, so the CLI is its only path
        # to the env's tools. Makes the trajectory check below deterministic: if the agent
        # answers correctly, it must have invoked the CLI.
        deploy_agent_step = DeployAgentTaskStep(
            id=f"cli-e2e-deploy-agent-{suffix}", version=None, env_ids=[],
            a2a_agent_id=a2a_agent.id, a2a_agent_version=a2a_agent.version,
        )
        await deploy_agent_step.execute(context)
        deployed_agent_ref = context.deployed_agents[0]
        logger.info(f"CLI e2e: deployed agent at {deployed_agent_ref.api_url}")

        build_step = BuildMcpCliTaskStep(id=f"cli-e2e-build-{suffix}", version=None, env_id=multi_env.id, command_name=multi_env.id)
        await build_step.execute(context)
        cli_ref = context.metadata["cli_artifact"]
        assert cli_ref["id"] == f"{multi_env.id}__cli"
        logger.info(f"CLI e2e: built CliArtifact {cli_ref['id']} v{cli_ref['version']}")

        load_step = LoadArtifactTaskStep(
            id=f"cli-e2e-load-{suffix}", version=None,
            env_id=multi_env.id, artifact_id=cli_ref["id"], artifact_version=cli_ref["version"],
            agent_name=TaskStep.DEFAULT_AGENT_NAME,
        )
        await load_step.execute(context)

        installed = context.metadata["installed_clis"][TaskStep.DEFAULT_AGENT_NAME][cli_ref["id"]]
        assert installed["command_name"] == multi_env.id
        assert installed["install_path"] == f"/opt/cli/{multi_env.id}/bin/{multi_env.id}"

        deployed_a2a = get_a2a_agent_instance_store().get(deployed_agent_ref.instance_id)
        sandbox = await build_sandbox_provider(deployed_a2a.sandbox_type).get_sandbox(deployed_a2a.sandbox_id)
        ls_output = await exec_in(sandbox, "docker exec agent-api", "ls", "-a", f"/opt/cli/{multi_env.id}/bin/")
        assert multi_env.id in ls_output
        assert ".env" in ls_output
        env_content = await exec_in(sandbox, "docker exec agent-api", "cat", f"/opt/cli/{multi_env.id}/bin/.env")
        assert f"AGENT_ENV_GATEWAY_URL={deployed_env.gateway_url}" in env_content

        add_skills_step = AddSkillsTaskStep(id=f"cli-e2e-skill-{suffix}", version=None, cli_artifact_ids=[cli_ref["id"]])
        await add_skills_step.execute(context)
        logger.info(f"CLI e2e: registered skill {multi_env.id}-cli on agent")

        prompt_step = PromptAgentTaskStep(
            id=f"cli-e2e-prompt-{suffix}", version=None,
            prompt=(
                "Use the available CLI skill to list public Slack channels (limit 3). "
                "Report exactly what the CLI prints."
            ),
            timeout_seconds=300,
        )
        await prompt_step.execute(context)
        prompt_response = context.prompt_responses[-1]
        logger.info(f"CLI e2e prompt response: {prompt_response.response[:300]}")
        assert prompt_response.tool_call_count and prompt_response.tool_call_count > 0, \
            f"Expected agent to make at least one tool call; got {prompt_response.tool_call_count}"
        assert prompt_response.agent_trajectory_s3_uri, "Expected trajectory to be uploaded"

        trajectory_json = get_config().get_object_store().get(prompt_response.agent_trajectory_s3_uri).decode()
        assert installed["install_path"] in trajectory_json, (
            f"Expected install_path '{installed['install_path']}' to appear in agent trajectory "
            f"(would prove CLI was invoked); not found in {len(trajectory_json)} bytes of trajectory"
        )
        logger.info(f"CLI e2e: confirmed agent invoked CLI at {installed['install_path']} via trajectory inspection")

        # Env gateway trajectory: stronger receipt — proves the /step call landed and was dispatched.
        # Since env_ids=[] for the agent (no MCP), the only path producing a step_api tool_call is the CLI.
        # Path is set by the gateway container's GATEWAY_TRAJECTORY_FILE env var.
        sandbox_env = await build_sandbox_provider(deployed_env.sandbox_type).get_sandbox(deployed_env.sandbox_id)
        gateway_log_raw = await exec_in(sandbox_env, "sudo docker exec app-gateway-1", "cat ${GATEWAY_TRAJECTORY_FILE:-/tmp/agentenv/*-trajectory.jsonl}")
        gateway_events = [json.loads(line) for line in gateway_log_raw.strip().split("\n") if line.strip()]
        step_tool_calls = [e for e in gateway_events if e.get("event_type") == "tool_call" and e.get("source") == "step_api"]
        assert step_tool_calls, f"Expected at least one /step tool_call in env gateway trajectory; got {len(gateway_events)} events: {gateway_events[:3]}"
        called_tools = {e["tool_call"]["function_name"] for e in step_tool_calls}
        assert "channels_list" in called_tools, f"Expected channels_list call from CLI; got {called_tools}"
        results = [e for e in gateway_events if e.get("event_type") == "tool_call_result"]
        assert any(not r.get("tool_call_result", {}).get("isError") for r in results), \
            "Expected at least one successful tool_call_result"
        logger.info(f"CLI e2e: env gateway saw step_api tool calls for tools={called_tools}")
    finally:
        try:
            env_obj = await type(Env.get(deployed_env.env_id, deployed_env.env_version)).from_deployed_env(deployed_env)
            await env_obj.close()
        except Exception as e:
            logger.warning(f"CLI e2e env cleanup: {e}")
        if context.deployed_agents:
            try:
                deployed_a2a = get_a2a_agent_instance_store().get(context.deployed_agents[0].instance_id)
                sandbox = await build_sandbox_provider(deployed_a2a.sandbox_type).get_sandbox(deployed_a2a.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                logger.warning(f"CLI e2e agent cleanup: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
# prompt_agent stores trajectories in S3 (_trajectory_prefix); the profiles that have it are the model-endpoint ones.
@skip_without_model_endpoint()
async def test_prompt_agent_model_params_reaches_agent(sandbox_provider, multi_env, echo_agent):
    """prompt_agent negotiates model_params to a deployed agent over the agent-config
    extension: the echo agent advertises it as a write-only field, so it must arrive and
    read back redacted (never echoing provider secrets)."""
    from agent_env.a2a_agent.store import get_a2a_agent_instance_store
    from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider

    suffix = uuid.uuid4().hex[:8]
    deploy_env_step = DeployEnvTaskStep(id=f"mp-env-{suffix}", version=None, env_id=multi_env.id, env_version=multi_env.version)
    context = await deploy_env_step.execute(TaskStepContext())
    deployed_env = context.deployed_envs[0]
    try:
        deploy_agent_step = DeployAgentTaskStep(
            id=f"mp-agent-{suffix}", version=None, env_ids=[multi_env.id],
            a2a_agent_id=echo_agent.id, a2a_agent_version=echo_agent.version,
        )
        await deploy_agent_step.execute(context)
        agent = context.deployed_agents[0]
        target_url = agent.a2a_url or agent.api_url

        # The agent must ADVERTISE model_params, else the negotiation filter drops it.
        async with httpx.AsyncClient(timeout=30) as client:
            card = (await client.get(f"{target_url}/.well-known/agent.json")).json()
        cfg_ext = next(e for e in card["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:agent-config/v1")
        assert "model_params" in cfg_ext["params"]["methods"]["set"]["request"]["supported"], \
            f"agent card does not advertise model_params: {cfg_ext}"

        prompt_step = PromptAgentTaskStep(
            id=f"mp-prompt-{suffix}", version=None,
            prompt="Reply with exactly this token and nothing else: MP_OK",
            model_params={"aws_region_name": "us-west-2"},
            timeout_seconds=300,
        )
        await prompt_step.execute(context)

        # prompt_agent POSTed the negotiated model_params to /ext/agent-config; read it
        # back — it must be present (received) but redacted (never echo provider secrets).
        async with httpx.AsyncClient(timeout=30) as client:
            agent_cfg = (await client.get(f"{target_url}/ext/agent-config")).json()["config"]
        assert agent_cfg.get("model_params") == "***", \
            f"expected received-and-redacted model_params, got {agent_cfg}"
        logger.info("model_params reached the agent and read back redacted")
    finally:
        if context.deployed_agents:
            try:
                inst = get_a2a_agent_instance_store().get(context.deployed_agents[0].instance_id)
                sandbox = await build_sandbox_provider(inst.sandbox_type).get_sandbox(inst.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                logger.warning(f"model_params test agent cleanup: {e}")
        try:
            env_obj = await type(Env.get(deployed_env.env_id, deployed_env.env_version)).from_deployed_env(deployed_env)
            await env_obj.close()
        except Exception as e:
            logger.warning(f"model_params test env cleanup: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.dependency(depends=["test_task_steps_e2e"])
@skip_without_model_endpoint()
@skip_without_default_a2a_agent()
async def test_task_e2e(multi_env, environment_universe_artifact):
    """Test full Task lifecycle: create from stored steps, persist, retrieve, run, verify."""

    # 1. Retrieve the steps already put by test_task_steps_e2e
    steps = [
        TaskStep.get(_run_id("task-step-test-deploy-multi")),
        TaskStep.get(_run_id("task-step-test-load-email-service")),
        TaskStep.get(_run_id("task-step-test-load-universe")),
        TaskStep.get(_run_id("task-step-test-deploy-agent")),
        TaskStep.get(_run_id("task-step-test-prompt-agent")),
        TaskStep.get(_run_id("task-step-test-env-outcome-verifier")),
        TaskStep.get(_run_id("task-step-test-rubrics-verifier")),
    ]
    logger.info(f"Retrieved {len(steps)} steps from store")

    # 2. Create and persist a Task
    task = Task.put(id=_run_id("task-test-e2e"), steps=steps)
    assert task.version is not None
    assert len(task.steps) == 7
    logger.info(f"Created Task: id={task.id} version={task.version}")

    # 3. Retrieve the Task from the store
    retrieved = Task.get(task.id, task.version)
    assert retrieved.id == task.id
    assert retrieved.version == task.version
    assert len(retrieved.steps) == 7
    assert isinstance(retrieved.steps[0], DeployEnvTaskStep)
    assert isinstance(retrieved.steps[1], LoadArtifactTaskStep)
    assert isinstance(retrieved.steps[2], LoadArtifactTaskStep)
    assert isinstance(retrieved.steps[3], DeployAgentTaskStep)
    assert isinstance(retrieved.steps[4], PromptAgentTaskStep)
    assert isinstance(retrieved.steps[5], EnvOutcomeVerifierTaskStep)
    assert isinstance(retrieved.steps[6], RubricsVerifierTaskStep)
    logger.info(f"Retrieved Task: id={retrieved.id} version={retrieved.version}")

    # 4. Run the task, verifying task instance on each step
    def verify_instance_on_step(i, total, step, context, duration):
        assert context.instance_id is not None
        inst = Task.get_instance(context.instance_id)
        expected_status = "completed" if i + 1 == total else "running"
        assert inst.status == expected_status
        assert inst.current_step == i + 1
        assert inst.total_steps == total
        assert inst.context is not None
        logger.info(f"Task instance after step {i+1}/{total}: current_step={inst.current_step} status={inst.status}")

    context = await retrieved.run(on_step_complete=verify_instance_on_step)

    # 5. Verify results
    assert len(context.deployed_envs) == 1
    deployed = context.deployed_envs[0]
    assert deployed.env_id == multi_env.id
    assert deployed.gateway_url.startswith("http")
    logger.info(f"Task deployed env: {deployed.env_id} gateway={deployed.gateway_url}")

    assert len(context.deployed_agents) == 1
    assert context.deployed_agents[0].api_url.startswith("http")
    logger.info(f"Task deployed agent: {context.deployed_agents[0].agent_name}")

    assert len(context.prompt_responses) == 1
    prompt_response = context.prompt_responses[0]
    assert "sarah.johnson" in prompt_response.response.lower(), \
        f"Expected 'sarah.johnson' in response, got: {prompt_response.response}"
    logger.info(f"Task prompt response: {prompt_response.response}")

    # Verify the trajectory exists in the configured object store
    assert prompt_response.agent_trajectory_s3_uri is not None, "Expected agent_trajectory_s3_uri to be set"
    trajectory = json.loads(get_config().get_object_store().get(prompt_response.agent_trajectory_s3_uri))
    assert isinstance(trajectory, list) and len(trajectory) > 0, "Expected non-empty trajectory list"
    logger.info(f"Task trajectory: {len(trajectory)} messages at {prompt_response.agent_trajectory_s3_uri}")

    # Verify both verifier results coexist in context.metadata["verifications"]
    verifications = context.metadata.get("verifications", {})
    assert len(verifications) == 2, f"Expected 2 verifiers, got {len(verifications)}: {list(verifications.keys())}"
    for vid, vdata in verifications.items():
        results = vdata["results"]
        assert isinstance(results, list) and len(results) > 0, \
            f"Expected non-empty results for verifier '{vid}'"
        for result in results:
            assert "id" in result, f"Expected 'id' field in result: {result}"
            assert "result" in result or "score" in result, f"Expected 'result' or 'score' field in result: {result}"
        assert vdata["score"] == 1.0, f"Expected score 1.0 for verifier '{vid}', got {vdata['score']}"
    logger.info(f"Task verifications: {list(verifications.keys())}")

    # 6. Verify task instance was created and matches context
    assert context.instance_id is not None, "Expected instance_id to be set on context"
    task_instance = Task.get_instance(context.instance_id)
    assert task_instance.task_id == retrieved.id
    assert task_instance.task_version == retrieved.version
    assert task_instance.status == "completed"
    assert task_instance.current_step == len(retrieved.steps)
    assert task_instance.total_steps == len(retrieved.steps)
    assert task_instance.completed_at_utc is not None
    assert task_instance.context is not None
    restored_context = TaskStepContext.from_dict(task_instance.context)
    assert len(restored_context.deployed_envs) == len(context.deployed_envs)
    assert len(restored_context.prompt_responses) == len(context.prompt_responses)
    assert restored_context.metadata.get("verifications", {}).keys() == context.metadata.get("verifications", {}).keys()
    logger.info(f"Task instance verified: {task_instance.instance_id} status={task_instance.status}")

    # 7. Write context to temp file for downstream tests
    context_path = Path(f"/tmp/test_task_e2e_context_{TEST_RUN_SUFFIX}.json")
    context_path.write_text(json.dumps(dataclasses.asdict(context), indent=2))
    logger.info(f"Wrote task context to {context_path}")

    # 8. Teardown
    env = await type(Env.get(deployed.env_id, deployed.env_version)).from_deployed_env(deployed)
    await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.dependency(depends=["test_task_e2e"])
@skip_without_model_endpoint()
@skip_without_default_a2a_agent()
async def test_task_run_with_start_step_and_context():
    """Test re-running rubrics verifier from a prior run's context."""
    # 1. Load context written by test_task_e2e
    context_path = Path(f"/tmp/test_task_e2e_context_{TEST_RUN_SUFFIX}.json")
    context_dict = json.loads(context_path.read_text())
    restored = TaskStepContext.from_dict(context_dict)

    # 2. Clear verifications to prove the verifier actually re-runs
    restored.metadata.pop("verifications", None)

    # 3. Retrieve the same task, run from step 6 (rubrics_verifier)
    task = Task.get(_run_id("task-test-e2e"))
    result = await task.run(start_step=6, context=restored)

    # 4. Verify rubrics verifier produced fresh results
    verifications = result.metadata.get("verifications", {})
    assert len(verifications) == 1, f"Expected 1 verifier, got {len(verifications)}"
    for vid, vdata in verifications.items():
        assert len(vdata["results"]) == 2, f"Expected 2 criteria, got {len(vdata['results'])}"
        assert "score" in vdata
    logger.info(f"start_step test verifications: {verifications}")


async def _verify_list_tools(mcp_url: str, tool_verifier_callback: callable):
    max_retries = 10
    for attempt in range(max_retries):
        logger.info(f"Testing gateway via tunnel (attempt {attempt + 1}/{max_retries})...")
        verified = False
        try:
            async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    logger.info(f"Available tools: {tool_names}")
                    await tool_verifier_callback(session, tools_result, tool_names)
                    verified = True
        except BaseExceptionGroup as eg:
            assertion_errors, _ = eg.split(AssertionError)
            if assertion_errors:
                raise
            if verified:
                logger.info(f"  Verified (ignoring cleanup error)")
                break
            logger.info(f"  Attempt {attempt + 1} failed: {type(eg).__name__}")
            if attempt < max_retries - 1:
                await asyncio.sleep(10)
            else:
                raise RuntimeError(f"MCP client failed after {max_retries} attempts") from eg
        except Exception as e:
            logger.info(f"  Attempt {attempt + 1} failed: {type(e).__name__}")
            if attempt < max_retries - 1:
                await asyncio.sleep(10)
            else:
                raise RuntimeError(f"MCP client failed after {max_retries} attempts") from e
        else:
            break

    logger.info("Test passed!")


class _MetadataWriterStep(TaskStep):
    """Test-only step that applies a caller-provided mutation to context.

    Drives the full Task.run scheduler + path-level Mongo write pipeline with
    no infrastructure — so tests can pin concurrent-metadata-write semantics
    (shared nested parents, deletions, subtree replacements, $push chains)
    without paying sandbox/agent/judge cost.
    """
    type = "_synthetic_metadata_writer"

    def __init__(self, id, mutate_fn, depends_on=None, fail_task_on_error=True):
        super().__init__(id=id, version=None, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self._mutate = mutate_fn

    async def execute(self, context):
        self._mutate(context)
        return context


class _AsyncBlockingStep(TaskStep):
    """Test-only step that blocks on an asyncio.Event for cancel-path tests.

    Lets a test wait until the step is in-flight, then trigger a cancel from
    outside `Task.run()` to exercise the scheduler's external-interrupt
    handling (the try/except around the dispatch loop in task.py).
    """
    type = "_synthetic_async_blocker"

    def __init__(self, id, started_event, mutate_fn=None, depends_on=None, fail_task_on_error=True):
        super().__init__(id=id, version=None, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self._started = started_event
        self._mutate = mutate_fn

    async def execute(self, context):
        if self._mutate:
            self._mutate(context)
        # Signal that we're inside execute() so the test can cancel us.
        self._started.set()
        # Block until cancelled. asyncio.Event.wait() never resolves on its
        # own here — the only way out is for the surrounding task to cancel
        # us, which propagates as CancelledError up through Task.run().
        await asyncio.Event().wait()
        return context


class _RaisingStep(TaskStep):
    """Test-only step that always raises — exercises fail_task_on_error=False."""
    type = "_synthetic_raising_step"

    def __init__(self, id, depends_on=None, error_msg="synthetic-failure", fail_task_on_error=True):
        super().__init__(id=id, version=None, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self._error_msg = error_msg

    async def execute(self, context):
        raise RuntimeError(self._error_msg)


@pytest.mark.asyncio
@pytest.mark.integration
# The only test in this module that prompts an agent, so the only one that needs
# a model endpoint. The rest fail on other grounds under the local defaults.
@skip_without_model_endpoint()
@skip_without_default_a2a_agent()
async def test_task_dag_parallel_e2e(
    mcp_server_env, slack_mcp_server_env,
    email_service_artifact, slack_service_artifact,
    a2a_agent,
):
    """End-to-end DAG task: parallel envs + loads + agents + prompts, single verifier.

    Also drives a synthetic fan-out of metadata-writer steps that exercise the
    path-level Mongo write semantics the real-infra steps don't: concurrent
    shared-nested-parent writes (metadata.verifications.<id>), key deletion,
    subtree replacement, and sequential $push chains.

    Plus a tolerated-failure branch: one synthetic step always raises, the
    task runs with fail_task_on_error=False, and we assert the failure is
    recorded in context.metadata['failed_steps'] with is_fatal=False while
    its dependent still ran.

    Verifies that independent roots overlap, dep edges are respected, and
    non-failed steps appear in inst.completed_steps with status == success.
    """
    import threading
    import time as _time
    from agent_env.task_step.task_step import TaskStepDependency as Dep

    steps = [
        DeployEnvTaskStep(
            id="dag-deploy-env-email", version=None,
            env_id=mcp_server_env.id, env_version=mcp_server_env.version,
            depends_on=[],
        ),
        DeployEnvTaskStep(
            id="dag-deploy-env-slack", version=None,
            env_id=slack_mcp_server_env.id, env_version=slack_mcp_server_env.version,
            depends_on=[],
        ),
        LoadArtifactTaskStep(
            id="dag-load-email", version=None,
            env_id=mcp_server_env.id,
            artifact_id=email_service_artifact.id,
            artifact_version=email_service_artifact.version,
            depends_on=[Dep(task_step_id="dag-deploy-env-email")],
        ),
        LoadArtifactTaskStep(
            id="dag-load-slack", version=None,
            env_id=slack_mcp_server_env.id,
            artifact_id=slack_service_artifact.id,
            artifact_version=slack_service_artifact.version,
            depends_on=[Dep(task_step_id="dag-deploy-env-slack")],
        ),
        DeployAgentTaskStep(
            id="dag-deploy-agent-1", version=None,
            agent_name="agent-1",
            a2a_agent_id=a2a_agent.id,
            env_ids=[mcp_server_env.id, slack_mcp_server_env.id],
            depends_on=[Dep(task_step_id="dag-load-email"), Dep(task_step_id="dag-load-slack")],
        ),
        DeployAgentTaskStep(
            id="dag-deploy-agent-2", version=None,
            agent_name="agent-2",
            a2a_agent_id=a2a_agent.id,
            env_ids=[mcp_server_env.id, slack_mcp_server_env.id],
            depends_on=[Dep(task_step_id="dag-load-email"), Dep(task_step_id="dag-load-slack")],
        ),
        PromptAgentTaskStep(
            id="dag-prompt-1", version=None,
            prompt_id="dag-prompt-1",
            agent_name="agent-1",
            prompt="Use the list_emails tool on the INBOX folder. Who sent the email with subject 'Weekly standup reminder'? Reply with only their email address.",
            depends_on=[Dep(task_step_id="dag-deploy-agent-1")],
        ),
        PromptAgentTaskStep(
            id="dag-prompt-2", version=None,
            prompt_id="dag-prompt-2",
            agent_name="agent-2",
            prompt="Use the channels_list tool with channel_types='public_channel'. Reply with a comma-separated list of the channel names.",
            depends_on=[Dep(task_step_id="dag-deploy-agent-2")],
        ),
        RubricsVerifierTaskStep(
            id="dag-verify", version=None,
            verifier_id="dag-verify-prompt-1",
            prompt_id="dag-prompt-1",
            agent_name="agent-1",
            criteria=[{
                "id": "criterion-1",
                "title": "The agent's final response contains the email address sarah.johnson@techcorp.com",
                "rubric_category": "Outcome",
                "rubric_target": "Outcome",
            }],
            default_model="claude-sonnet-4-6",
            depends_on=[Dep(task_step_id="dag-prompt-1")],
        ),
    ]

    # Synthetic fan-out to exercise path-level Mongo write semantics that the
    # real-infra steps above don't hit. Runs independently of the sandbox/agent
    # pipeline (roots at synth-seed, no deps on dag-* steps).
    def _seed(ctx):
        ctx.metadata["to_delete"] = "stale"
        ctx.metadata["seed"] = {"old": 1}

    def _write_verification(i):
        def fn(ctx):
            ctx.metadata.setdefault("verifications", {})[f"v_{i}"] = {"score": i / 10}
        return fn

    def _delete_stale(ctx):
        ctx.metadata.pop("to_delete", None)

    def _replace_seed(ctx):
        ctx.metadata["seed"] = {"new": 2}

    def _append_run(run_id):
        def fn(ctx):
            ctx.metadata.setdefault("completed_runs", []).append({"run_id": run_id})
        return fn

    after_seed = [Dep(task_step_id="synth-seed")]
    steps += [
        _MetadataWriterStep(id="synth-seed", mutate_fn=_seed, depends_on=[]),
        *[
            _MetadataWriterStep(
                id=f"synth-verify-{i}", mutate_fn=_write_verification(i),
                depends_on=after_seed,
            )
            for i in range(6)
        ],
        _MetadataWriterStep(id="synth-delete", mutate_fn=_delete_stale, depends_on=after_seed),
        _MetadataWriterStep(id="synth-replace", mutate_fn=_replace_seed, depends_on=after_seed),
        # Both appenders run concurrently after seed. Each emits $addToSet for
        # its tail items; even though the second-running appender's `post`
        # snapshot also includes the first's item (via shared in-memory ctx),
        # Mongo dedupes by BSON value so no item lands twice.
        _MetadataWriterStep(id="synth-append-1", mutate_fn=_append_run("r1"), depends_on=after_seed),
        _MetadataWriterStep(id="synth-append-2", mutate_fn=_append_run("r2"), depends_on=after_seed),
        # Tolerated-failure branch: independent failing root + dependent.
        _RaisingStep(id="synth-fail-root", depends_on=[], fail_task_on_error=False),
        _MetadataWriterStep(
            id="synth-fail-dependent",
            mutate_fn=lambda ctx: ctx.metadata.setdefault("tolerated_chain", []).append("ran"),
            depends_on=[Dep(task_step_id="synth-fail-root")],
        ),
    ]

    task = Task.put(id="task-test-dag-parallel-e2e", steps=steps)
    logger.info(f"Created DAG task: id={task.id} version={task.version} steps={len(task.steps)}")

    # Capture per-step monotonic timestamps for DAG-edge assertions.
    timeline: list[tuple[str, str, float]] = []
    timeline_lock = threading.Lock()

    def on_start(i, total, step, ctx):
        with timeline_lock:
            timeline.append((step.id, "start", _time.monotonic()))
        logger.info(f"DAG step started [{i+1}/{total}]: {step.id}")

    def on_complete(i, total, step, ctx, dur):
        with timeline_lock:
            timeline.append((step.id, "end", _time.monotonic()))
        logger.info(f"DAG step completed [{i+1}/{total}]: {step.id} [{dur:.1f}s]")

    context = await task.run(on_step_start=on_start, on_step_complete=on_complete)

    # --- Shape assertions (real-infra steps) ---
    assert len(context.deployed_envs) == 2
    assert {e.env_id for e in context.deployed_envs} == {mcp_server_env.id, slack_mcp_server_env.id}
    assert len(context.deployed_agents) == 2
    assert {a.agent_name for a in context.deployed_agents} == {"agent-1", "agent-2"}
    assert len(context.prompt_responses) == 2
    verifications = context.metadata.get("verifications", {})
    expected_verifiers = {"dag-verify-prompt-1"} | {f"v_{i}" for i in range(6)}
    assert set(verifications.keys()) == expected_verifiers, (
        f"Expected {sorted(expected_verifiers)}, got {sorted(verifications.keys())}"
    )

    # --- Shape assertions (synthetic metadata semantics) ---
    assert context.metadata.get("to_delete") is None, "synth-delete didn't remove key"
    assert context.metadata["seed"] == {"new": 2}, "synth-replace didn't overwrite subtree"
    for i in range(6):
        assert verifications[f"v_{i}"] == {"score": i / 10}
    assert [r["run_id"] for r in context.metadata["completed_runs"]] == ["r1", "r2"]

    # --- Timeline / DAG-edge assertions ---
    starts = {sid: ts for sid, ev, ts in timeline if ev == "start"}
    ends = {sid: ts for sid, ev, ts in timeline if ev == "end"}

    # env1 || env2 overlap
    assert min(ends["dag-deploy-env-email"], ends["dag-deploy-env-slack"]) \
         > max(starts["dag-deploy-env-email"], starts["dag-deploy-env-slack"]), \
         "Expected env deploys to overlap"

    # load_email waits only on env-email (not env-slack)
    assert starts["dag-load-email"] >= ends["dag-deploy-env-email"]
    assert starts["dag-load-slack"] >= ends["dag-deploy-env-slack"]

    # Both agents wait for BOTH loads (fan-in barrier)
    for agent_sid in ("dag-deploy-agent-1", "dag-deploy-agent-2"):
        assert starts[agent_sid] >= ends["dag-load-email"], f"{agent_sid} started before load-email ended"
        assert starts[agent_sid] >= ends["dag-load-slack"], f"{agent_sid} started before load-slack ended"

    # agent1 || agent2 overlap
    assert min(ends["dag-deploy-agent-1"], ends["dag-deploy-agent-2"]) \
         > max(starts["dag-deploy-agent-1"], starts["dag-deploy-agent-2"]), \
         "Expected agent deploys to overlap"

    # prompts wait on their own agent only
    assert starts["dag-prompt-1"] >= ends["dag-deploy-agent-1"]
    assert starts["dag-prompt-2"] >= ends["dag-deploy-agent-2"]

    # prompt1 || prompt2 overlap
    assert min(ends["dag-prompt-1"], ends["dag-prompt-2"]) \
         > max(starts["dag-prompt-1"], starts["dag-prompt-2"]), \
         "Expected prompts to overlap"

    # verify depends only on prompt-1
    assert starts["dag-verify"] >= ends["dag-prompt-1"]

    # --- Tolerated-failure assertions ---
    failed = context.metadata.get("failed_steps") or []
    assert len(failed) == 1, f"Expected 1 failed step, got {failed}"
    fail_entry = failed[0]
    assert fail_entry["step_id"] == "synth-fail-root"
    assert fail_entry["is_fatal"] is False
    assert fail_entry["error_type"] == "RuntimeError"
    assert fail_entry["error"] == "synthetic-failure"
    assert isinstance(fail_entry["started_at_utc"], str) and fail_entry["started_at_utc"]
    assert fail_entry["duration_seconds"] >= 0
    assert context.metadata.get("tolerated_chain") == ["ran"], "Dependent of failed step should still have run"

    # --- Mongo instance assertions ---
    # Tolerant-mode failures heartbeat with status=FAILURE, so every step appears
    # in completed_steps and the run flips to status="completed" once all heartbeat.
    inst = Task.get_instance(context.instance_id)
    assert inst.status == "completed", f"Expected status=completed, got {inst.status}"
    assert inst.completed_at_utc is not None
    assert {r.step_id for r in inst.completed_steps} == {s.id for s in steps}
    fail_entries = {(r.step_id, r.status) for r in inst.completed_steps if r.status == TaskStepStatus.FAILURE}
    assert fail_entries == {("synth-fail-root", TaskStepStatus.FAILURE)}, f"Unexpected failure entries: {fail_entries}"
    assert {r.status for r in inst.completed_steps} == {TaskStepStatus.SUCCESS, TaskStepStatus.FAILURE}
    assert inst.rev == len(steps), f"Expected rev == {len(steps)}, got {inst.rev}"
    assert inst.current_step == len(steps)
    logger.info(f"DAG instance verified: {inst.instance_id} rev={inst.rev}")

    # --- Mongo-persisted context assertions (path-level write correctness) ---
    persisted = (inst.context or {}).get("metadata") or {}
    assert "to_delete" not in persisted, "synth-delete's $unset didn't reach Mongo"
    assert persisted.get("seed") == {"new": 2}, "synth-replace's subtree didn't reach Mongo"
    persisted_verifications = persisted.get("verifications") or {}
    assert set(persisted_verifications.keys()) == expected_verifiers, (
        "concurrent sibling-leaf writes under metadata.verifications clobbered each other: "
        f"persisted {sorted(persisted_verifications.keys())}, expected {sorted(expected_verifiers)}"
    )
    for i in range(6):
        assert persisted_verifications[f"v_{i}"] == {"score": i / 10}
    persisted_run_ids = {r["run_id"] for r in persisted.get("completed_runs", [])}
    assert persisted_run_ids == {"r1", "r2"}, "sequential $push chain lost an append"

    # --- the step journal is a faithful record of this DAG ---
    # `live_context` is the independent oracle: nothing writes `context` wholesale on the
    # success path, so a diff that under-reports a real write shows up here and nowhere else.
    journal.assert_journal_replays(
        context.instance_id, expect_steps={s.id for s in steps}, live_context=context,
    )

    # --- Verifier score assertion ---
    vdata = verifications["dag-verify-prompt-1"]
    assert vdata["score"] == 1.0, f"Expected verify score 1.0, got {vdata['score']}"

    # --- Teardown ---
    for deployed in context.deployed_envs:
        try:
            env = await type(Env.get(deployed.env_id, deployed.env_version)).from_deployed_env(deployed)
            await env.close()
        except Exception as e:
            logger.warning(f"Failed to clean up env sandbox {deployed.sandbox_id}: {e}")
    for agent in context.deployed_agents:
        if agent.sandbox_id:
            try:
                from agent_env.providers.sandbox_providers.sandbox_provider import get_sandbox_provider
                sandbox = await get_sandbox_provider().get_sandbox(agent.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                logger.warning(f"Failed to clean up agent sandbox {agent.sandbox_id}: {e}")

    logger.info("DAG parallel e2e test passed!")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_task_external_cancel_marks_failed():
    """Cancel-path: external cancel of Task.run() (a runner's timeout,
    worker SIGTERM, etc.) should:

      1. Cancel any in-flight steps.
      2. Mark the Mongo task instance as failed (not stuck `running`).
      3. Re-raise the original CancelledError so the runner sees a cancel,
         not a generic step failure.

    Exercises the try/except (asyncio.CancelledError, Exception) block we
    added around Task.run()'s scheduler. Without that block, in-flight tasks
    get GC'd silently and the Mongo doc stays `running` forever.
    """
    from agent_env.task_step.task_step import TaskStepDependency as Dep

    # Two parallel root steps that both block on a never-resolved event.
    started_a = asyncio.Event()
    started_b = asyncio.Event()

    def _seed(ctx):
        ctx.metadata["seed"] = "value"

    steps = [
        _MetadataWriterStep(id="cancel-seed", mutate_fn=_seed, depends_on=[]),
        _AsyncBlockingStep(id="cancel-blocker-a", started_event=started_a,
                           depends_on=[Dep(task_step_id="cancel-seed")]),
        _AsyncBlockingStep(id="cancel-blocker-b", started_event=started_b,
                           depends_on=[Dep(task_step_id="cancel-seed")]),
    ]
    task = Task.put(id=f"task-test-cancel-{uuid.uuid4().hex[:6]}", steps=steps)
    logger.info(f"Created cancel-path task: id={task.id} version={task.version}")

    context = TaskStepContext()
    run_task = asyncio.create_task(task.run(context=context))

    # Wait until BOTH blocking steps are inside execute().
    await asyncio.wait_for(started_a.wait(), timeout=10)
    await asyncio.wait_for(started_b.wait(), timeout=10)
    instance_id = context.instance_id
    assert instance_id is not None, "instance_id should be set by Task.run() before steps fire"
    logger.info(f"Both blocking steps in-flight; instance_id={instance_id}; cancelling Task.run()...")

    # External cancel — simulates a runner timeout / worker SIGTERM.
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    # Give the scheduler's `await asyncio.gather(*in_flight, return_exceptions=True)`
    # a moment to drain. The cancel handler also writes to Mongo, so allow time.
    await asyncio.sleep(0.5)

    # Mongo assertions: the instance should be marked failed, NOT still running.
    inst = Task.get_instance(instance_id)
    assert inst.status == "failed", (
        f"Expected status=failed after external cancel, got {inst.status!r}. "
        "The new try/except in Task.run() didn't fire — in-flight tasks were "
        "GC'd silently and the Mongo doc stayed `running`."
    )
    assert inst.completed_at_utc is not None, "completed_at_utc should be set"

    # The seed step's path-level write should still have landed (it completed
    # before the cancel).
    persisted = (inst.context or {}).get("metadata") or {}
    assert persisted.get("seed") == "value", (
        f"Pre-cancel completed step's data missing from Mongo: {persisted}"
    )

    logger.info(f"Cancel-path test passed: instance {instance_id} correctly marked failed")


# ---------- Multi-turn PromptAgentTaskStep integration tests ----------

_MULTI_TURN_AGENT_ID = "claude-code-cli"
_MULTI_TURN_SOLVER_NAME = "solver"
_MULTI_TURN_USERSIM_NAME = "hana_kim"
_MULTI_TURN_SOLVER_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Help the user with their request. "
    "If their question is vague, ask one clarifying question, then give a concrete recommendation."
)
_MULTI_TURN_INITIAL_PROMPT = "Hi! Can you help me figure out something to do this weekend?"


def _build_two_agent_multi_turn_task(task_id: str, user_sim_system_prompt: str, max_conversation_turns: int) -> Task:
    """Deploy solver + user-sim claude-code-cli agents on Modal and chain a multi-turn PromptAgentTaskStep."""
    deploy_solver_id = f"{task_id}-deploy-solver"
    deploy_usersim_id = f"{task_id}-deploy-usersim"
    prompt_id = f"{task_id}-prompt"
    deploy_solver = DeployAgentTaskStep(
        id=deploy_solver_id, version=None, env_ids=[],
        a2a_agent_id=_MULTI_TURN_AGENT_ID, agent_name=_MULTI_TURN_SOLVER_NAME,
        system_prompt=_MULTI_TURN_SOLVER_SYSTEM_PROMPT,
        sandbox_type="modal", ttl_seconds=1800, disk_size_gb=10, depends_on=[],
    )
    deploy_usersim = DeployAgentTaskStep(
        id=deploy_usersim_id, version=None, env_ids=[],
        a2a_agent_id=_MULTI_TURN_AGENT_ID, agent_name=_MULTI_TURN_USERSIM_NAME,
        system_prompt=user_sim_system_prompt,
        sandbox_type="modal", ttl_seconds=1800, disk_size_gb=10, depends_on=[],
    )
    prompt = PromptAgentTaskStep(
        id=prompt_id, version=None,
        agent_name=_MULTI_TURN_SOLVER_NAME, prompt_id=prompt_id,
        prompt=_MULTI_TURN_INITIAL_PROMPT,
        max_conversation_turns=max_conversation_turns,
        user_agent_name=_MULTI_TURN_USERSIM_NAME,
        timeout_seconds=600, user_agent_timeout_seconds=600,
        depends_on=[TaskStepDependency(deploy_solver_id), TaskStepDependency(deploy_usersim_id)],
    )
    return Task.put(id=task_id, steps=[deploy_solver, deploy_usersim, prompt])


@pytest.mark.asyncio
@pytest.mark.integration
@skip_without_remote_sandbox("modal")
async def test_multi_turn_done_signal_terminates_early():
    """User-sim with done=true ends the loop before hitting max_conversation_turns."""
    max_turns = 5
    user_sim_prompt = (
        "You are Hana Kim, planning a weekend getaway from San Francisco with your spouse and 6-year-old son. "
        "Preferences (share only when asked): nature/outdoor, animals, max 4-hour drive, $1500 budget. "
        "As soon as the assistant gives you a concrete destination recommendation that fits your needs, "
        "thank them briefly and set done=true. Keep replies to 1-2 sentences."
    )
    task_id = f"integ-multi-turn-done-{uuid.uuid4().hex[:8]}"
    task = _build_two_agent_multi_turn_task(task_id, user_sim_prompt, max_conversation_turns=max_turns)
    instance_id = uuid.uuid4().hex
    logger.info(f"Running {task.id} v{task.version}, instance_id={instance_id}")

    context = await task.run(instance_id=instance_id)

    assert context.prompt_responses, "expected at least one PromptResponse"
    pr = context.prompt_responses[-1]
    traj_uris = pr.target_agent_per_turn_trajectory_s3_uris
    prompts = pr.source_agent_per_turn_prompt_parts
    assert traj_uris is not None and len(traj_uris) >= 1
    assert prompts is not None and len(prompts) == len(traj_uris), \
        f"per-turn arrays diverged: trajectories={len(traj_uris)} prompts={len(prompts)}"
    assert len(traj_uris) < max_turns, \
        f"expected done=true to terminate early (turns < {max_turns}), got {len(traj_uris)}"
    for uri in traj_uris:
        if uri is not None:
            assert _scheme(uri) == _scheme(get_config().get_object_store().object_url("probe")), \
                f"trajectory URI not on the configured object store: {uri!r}"

    last_successful = next((u for u in reversed(traj_uris) if u is not None), None)
    assert pr.agent_trajectory_s3_uri == last_successful, \
        "agent_trajectory_s3_uri should equal the last successful per-turn URI"
    assert pr.prompt_text == _MULTI_TURN_INITIAL_PROMPT
    assert pr.response, "expected non-empty final response"

    conv = conversation_store.get_conversation(pr.a2a_context_id)
    assert conv is not None, f"conversation {pr.a2a_context_id} not found"
    assert conv["status"] == "closed"
    assert len(conv["messages"]) == 2 * len(traj_uris), \
        f"expected {2 * len(traj_uris)} messages (user+agent per turn), got {len(conv['messages'])}"


@pytest.mark.asyncio
@pytest.mark.integration
@skip_without_remote_sandbox("modal")
async def test_multi_turn_max_turns_cap_terminates_loop():
    """When the user-sim never signals done=true, the loop terminates at max_conversation_turns."""
    max_turns = 2
    user_sim_prompt = (
        "You are an extremely curious user planning a trip. After EVERY assistant response, "
        "ALWAYS ask another follow-up question to explore more details. NEVER set done=true under any circumstances. "
        "Keep replies to 1 sentence."
    )
    task_id = f"integ-multi-turn-cap-{uuid.uuid4().hex[:8]}"
    task = _build_two_agent_multi_turn_task(task_id, user_sim_prompt, max_conversation_turns=max_turns)
    instance_id = uuid.uuid4().hex
    logger.info(f"Running {task.id} v{task.version}, instance_id={instance_id}")

    context = await task.run(instance_id=instance_id)

    pr = context.prompt_responses[-1]
    traj_uris = pr.target_agent_per_turn_trajectory_s3_uris
    prompts = pr.source_agent_per_turn_prompt_parts
    assert traj_uris is not None and len(traj_uris) == max_turns, \
        f"expected exactly {max_turns} turns (cap), got {len(traj_uris) if traj_uris else None}"
    assert prompts is not None and len(prompts) == max_turns

    conv = conversation_store.get_conversation(pr.a2a_context_id)
    assert conv is not None
    assert conv["status"] == "closed", \
        f"expected post-loop mark_closed to set status=closed, got {conv['status']}"
