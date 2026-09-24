from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator import verify_a2a_agent_mcp
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_mcp import (
    VerifyA2AAgentMCPStep,
)


@pytest.mark.asyncio
async def test_mcp_verifier_recognizes_typed_usage_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.json.return_value = {"result": {"id": "task-1"}}

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    monkeypatch.setattr(verify_a2a_agent_mcp.httpx, "AsyncClient", lambda: client)

    agent = MagicMock()
    agent.metadata = {}
    monkeypatch.setattr(A2AAgent, "get", lambda *_args, **_kwargs: agent)

    step = VerifyA2AAgentMCPStep(
        id="verify-mcp",
        version=None,
        a2a_agent_id="agent-1",
    )
    monkeypatch.setattr(
        step,
        "_poll_task",
        AsyncMock(
            return_value={
                "status": {
                    "state": "completed",
                    "message": {
                        "parts": [
                            {"kind": "text", "text": "done"},
                            {
                                "kind": "data",
                                "data": {"usage": {"tool_call_count": 1}},
                            },
                        ]
                    },
                }
            }
        ),
    )
    monkeypatch.setattr(step, "_count_tool_calls", AsyncMock(return_value=1))
    context = TaskStepContext(
        deployed_agents=[
            SimpleNamespace(a2a_url="http://agent", api_url="", a2a_card={})
        ],
        deployed_envs=[SimpleNamespace(sandbox_id="sandbox-1")],
    )

    result = await step.execute(context)

    assert result.metadata["verifications"]["a2a_agent_mcp"] == {
        "passed": True,
        "tools_invoked": 1,
        "task_status": "completed",
        "task_id": "task-1",
        "agent_response": "done",
    }
    metadata = agent.update_metadata.call_args.args[0]
    assert metadata["validated_data_extensions"]["tool_call_count"] == {
        "supported": True
    }
