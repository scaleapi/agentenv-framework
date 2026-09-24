"""The A2A agent the integration suites deploy when they need an agent but not a model.

Deterministic: it echoes the prompt and records a native trajectory. Agent-config, mcp-config,
trajectory and triggers come from the agentenv-protocol framework. ``model_params`` is a write-only
config field so the agent-config negotiation and its redaction on read-back can be observed.
"""

from typing import Any

from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    WriteOnly,
    a2a_agent,
)


class EchoAgentConfig(AgentConfig):
    model: str | None = None
    system_prompt: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    model_params: WriteOnly[dict[str, Any] | None] = None


@a2a_agent(
    identity=AgentIdentity(
        name="agentenv-echo-agent",
        description="Deterministic agent for the agent-env integration suites: echoes its prompt.",
        version="1.0.0",
    ),
    config=EchoAgentConfig,
    extensions=(MCP_CONFIG_V1, TRAJECTORY_V1, TRIGGERS_V1),
)
class EchoAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[EchoAgentConfig]) -> TaskResult:
        prompt = "\n".join(part.text for part in request.parts if isinstance(part, TextPart))
        reply = f"Echo: {prompt}"
        return (
            TaskResult.builder()
            .succeeded()
            .add_text(reply)
            .usage(Usage(tool_call_count=0, input_tokens=len(prompt), output_tokens=len(reply)))
            .native_trajectory(
                format="agentenv-echo-agent/v1",
                payload=[{"type": "echo", "input": prompt, "output": reply}],
            )
            .build()
        )


if __name__ == "__main__":
    EchoAgent().serve()
