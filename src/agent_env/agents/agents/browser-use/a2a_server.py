"""Serves Browser Use Pi as an A2A agent: each turn runs ``agent.mjs`` once, and a context's turns share one
conversation through a saved history file. ``AGENT_CARD`` is what agent-env's install/v1 reads."""

import asyncio
import contextlib
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import Literal

from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    TRAJECTORY_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    a2a_agent,
    enable,
)

HERE = Path(__file__).parent
HOME = Path(os.environ.get("BROWSER_USE_A2A_HOME", Path.home() / ".browser-use-a2a"))
# Leaves the agent time to deliver what it has before the turn's deadline kills it.
DELIVERY_SECONDS = 30
STOP_SECONDS = 5
# Streamed token deltas; each message's end event carries the whole message.
_UNRECORDED_EVENTS = frozenset({"message_update", "tool_execution_update"})


class BrowserUseConfig(AgentConfig):
    model: str | None = None
    effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    system_prompt: str | None = None
    max_steps: int = 40
    # A Chrome DevTools endpoint to drive instead of a Browser Use Cloud browser.
    cdp_url: str | None = None
    timeout_seconds: int = 1800


@a2a_agent(
    identity=AgentIdentity(
        name="browser-use",
        description="Browser Use Pi: drives a real browser over CDP from a persistent JavaScript REPL.",
        version="1.0.0",
        input_modes=("text",),
    ),
    config=BrowserUseConfig,
    config_description="Configure Browser Use Pi: model, effort, system prompt, step budget and browser.",
    extensions=(
        MCP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=True, description="Browser Use Pi's agent events, live."),
    ),
)
class BrowserUseAgent(AgentEnvAgent):
    def __init__(self, command: tuple[str, ...] = ("node", str(HERE / "agent.mjs"))) -> None:
        self.command = command

    async def run(self, request: TaskRequest[BrowserUseConfig]) -> TaskResult:
        config = request.config
        model = config.model or os.environ.get("BROWSER_USE_A2A_MODEL")
        if not model:
            return TaskResult.failure("browser_use.no_model", "no model is configured; set 'model' through agent-config")
        session = request.session_ref or hashlib.sha256(request.context_id.encode()).hexdigest()[:32]
        workspace = request.workspace or HOME / "workspaces" / session
        workspace.mkdir(parents=True, exist_ok=True)
        turn = {
            "prompt": "\n\n".join(part.text for part in request.parts if isinstance(part, TextPart)),
            "model": model,
            "effort": config.effort,
            "systemPrompt": config.system_prompt,
            "maxSteps": config.max_steps,
            "timeoutMs": max(config.timeout_seconds - DELIVERY_SECONDS, 1) * 1000,
            "cdpUrl": config.cdp_url,
            "mcpServers": {name: {"url": s["url"], "headers": s.get("headers") or {}} for name, s in request.mcp_servers.items()},
            "workspace": str(workspace),
            # Browser Use Pi refuses a history saved under another model, so a model change starts a new conversation.
            "historyFile": str(HOME / "sessions" / f"{session}-{hashlib.sha256(model.encode()).hexdigest()[:8]}.json"),
        }
        (HOME / "sessions").mkdir(parents=True, exist_ok=True)
        request.trajectory.set_format("browser-use-pi-events/1")

        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, limit=64 * 1024 * 1024,
        )
        stderr = asyncio.create_task(process.stderr.read())
        result = None
        tool_calls = 0
        try:
            async with asyncio.timeout(config.timeout_seconds):
                process.stdin.write(json.dumps(turn).encode())
                process.stdin.close()
                async for line in process.stdout:
                    if not line.startswith(b"{"):
                        continue
                    message = json.loads(line)
                    if message["type"] == "result":
                        result = message["result"]
                    elif message["event"]["type"] not in _UNRECORDED_EVENTS:
                        request.trajectory.append(message["event"])
                        tool_calls += message["event"]["type"] == "tool_execution_end"
                await process.wait()
        except TimeoutError:
            return TaskResult.failure("browser_use.timeout", f"no answer within {config.timeout_seconds}s")
        finally:
            # On a timeout or a cancel, the agent stops its browser, then goes with everything it started.
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(process.wait()), STOP_SECONDS)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

        if result is None:
            error = (await stderr).decode(errors="replace")[-4000:] or f"agent.mjs exited {process.returncode}"
            return TaskResult.failure("browser_use.crashed", error, error_type="infra_error")
        usage = result["usage"]
        builder = TaskResult.builder()
        if result["status"] == "completed":
            builder.succeeded()
        else:
            builder.failed(f"browser_use.{result['status']}", result.get("error") or result["status"])
        return (
            builder.add_text(result["text"] or result["status"])
            .session_ref(session)
            .usage(Usage(
                tool_call_count=tool_calls,
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                total_tokens=usage["totalTokens"],
                provider_details={"steps": result["steps"], "model": result["model"]},
            ))
            .build()
        )


agent = BrowserUseAgent()
app = agent.create_app()
AGENT_CARD = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
