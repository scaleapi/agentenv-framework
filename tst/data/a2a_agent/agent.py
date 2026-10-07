"""The A2A agent the integration suites deploy when they need an agent but not a model.

Deterministic: it echoes the prompt and records a native trajectory. Agent-config, mcp-config,
trajectory and triggers come from the agentenv-protocol framework. ``model_params`` is a write-only
config field so the agent-config negotiation and its redaction on read-back can be observed.

It also moves files both ways. After the echo it adds a ``read <name> over <scheme>: <text>`` line for
each file part it is sent, fetching only inline bytes and HTTP(S) URLs (``could not read ...`` otherwise),
and it answers each ``send-file <uri>`` line of its prompt with a file part naming that URI.
"""

import base64
from typing import Any
from urllib.parse import urlsplit

import httpx
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    FilePart,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    WriteOnly,
    a2a_agent,
)

SEND_FILE = "send-file "
READ_TIMEOUT_SECONDS = 120


class EchoAgentConfig(AgentConfig):
    model: str | None = None
    system_prompt: str | None = None
    task_id: str | None = None
    model_params: WriteOnly[dict[str, Any] | None] = None


async def _read(part: FilePart) -> str:
    """What a file part holds, inline or at an HTTP(S) URL."""
    scheme = "bytes" if part.bytes is not None else urlsplit(part.uri).scheme
    try:
        if part.bytes is not None:
            data = base64.b64decode(part.bytes)
        else:
            async with httpx.AsyncClient(timeout=READ_TIMEOUT_SECONDS) as client:
                response = await client.get(part.uri)
                response.raise_for_status()
                data = response.content
    except Exception as exc:  # noqa: BLE001 -- said in the reply, so a test sees why
        return f"could not read {part.name} over {scheme}: {type(exc).__name__}"
    return f"read {part.name} over {scheme}: {data.decode(errors='replace').strip()}"


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
        reply = "\n".join([reply, *[await _read(part) for part in request.parts if isinstance(part, FilePart)]])
        sends = [line.removeprefix(SEND_FILE).strip() for line in prompt.splitlines() if line.startswith(SEND_FILE)]
        files = [FilePart(uri=uri, name=uri.rsplit("/", 1)[-1], mime_type="text/plain") for uri in sends]
        return (
            TaskResult.builder()
            .succeeded()
            .parts([TextPart(text=reply), *files])
            .usage(Usage(tool_call_count=0, input_tokens=len(prompt), output_tokens=len(reply)))
            .native_trajectory(
                format="agentenv-echo-agent/v1",
                payload=[{"type": "echo", "input": prompt, "output": reply}],
            )
            .build()
        )


if __name__ == "__main__":
    EchoAgent().serve()
