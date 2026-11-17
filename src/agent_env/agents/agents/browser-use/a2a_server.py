"""Serves browser-use 1.0 as an A2A agent: each turn runs ``agent.py`` once, in browser-use's own environment, on the
browser the task gives it through cdp-config/v1, and a context's turns share one conversation through a saved history
file. ``AGENT_CARD`` is what agent-env's install/v1 reads."""

import asyncio
import contextlib
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import Literal

from agentenv_protocol.a2a_agent import (
    CDP_CONFIG_V1,
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
#: The Python of browser-use's own environment, whose pins (mcp 2, among others) the protocol's can't share.
PYTHON = os.environ.get("BROWSER_USE_PYTHON", "/opt/browser-use/bin/python")
#: How much of the turn's deadline browser-use's own run leaves: its forced delivery when it runs out, and its stop.
DELIVERY_SECONDS = 100
STOP_SECONDS = 5
#: browser-use and its browser daemon report usage to their makers unless told not to.
_PRIVATE_ENV = {"ANONYMIZED_TELEMETRY": "false", "BH_TELEMETRY": "0"}


class BrowserUseConfig(AgentConfig):
    model: str | None = None
    #: browser-use's reasoning effort, on a Claude model its adaptive thinking's; unset, browser-use's own, medium.
    effort: Literal["low", "medium", "high"] | None = None
    #: browser-use's mode: ``ultrafast`` adds its accessibility-tree helpers; unset, its own, ``default`` here.
    mode: Literal["default", "ultrafast"] | None = None
    system_prompt: str | None = None
    #: Model requests a turn may make, its delivery included; unset, browser-use's own, 100.
    max_steps: int | None = None
    # The cdp-config endpoint to drive, the turn's only one when unnamed, and which of its addresses to connect by: the
    # browser's WebSocket, or its http_url to discover that from.
    browser: str | None = None
    connect_by: Literal["ws", "http"] = "ws"
    timeout_seconds: int = 1800


@a2a_agent(
    identity=AgentIdentity(
        name="browser-use",
        description="browser-use 1.0: drives a real browser over CDP from a persistent Python REPL.",
        version="2.0.0",
        input_modes=("text",),
    ),
    config=BrowserUseConfig,
    config_description="Configure browser-use: model, effort, mode, system prompt, step budget and browser.",
    extensions=(
        MCP_CONFIG_V1,
        CDP_CONFIG_V1,
        enable(TRAJECTORY_V1, live=False, description="browser-use's messages, each turn's when it ends."),
    ),
)
class BrowserUseAgent(AgentEnvAgent):
    def __init__(self, command: tuple[str, ...] = (PYTHON, str(HERE / "agent.py"))) -> None:
        self.command = command

    async def run(self, request: TaskRequest[BrowserUseConfig]) -> TaskResult:
        config = request.config
        model = config.model or os.environ.get("BROWSER_USE_A2A_MODEL")
        if not model:
            return TaskResult.failure("browser_use.no_model", "no model is configured; set 'model' through agent-config")
        # With no browser from the task the turn fails closed, so the agent never drives one of its own.
        endpoint = request.cdp_endpoint(config.browser)
        if endpoint["headers"]:
            # browser-use connects with the address alone, so a browser that needs headers isn't reached without them.
            return TaskResult.failure(
                "browser_use.cdp_headers",
                f"the browser needs headers ({', '.join(sorted(endpoint['headers']))}), which browser-use can't send",
                error_type="infra_error")
        session = request.session_ref or hashlib.sha256(request.context_id.encode()).hexdigest()[:32]
        workspace = request.workspace or HOME / "workspaces" / session
        workspace.mkdir(parents=True, exist_ok=True)
        turn = {
            "prompt": "\n\n".join(part.text for part in request.parts if isinstance(part, TextPart)),
            "model": model,
            "effort": config.effort,
            "mode": config.mode,
            "systemPrompt": config.system_prompt,
            "max_steps": config.max_steps,
            "timeout": max(config.timeout_seconds - DELIVERY_SECONDS, 1),
            "cdpUrl": endpoint["ws_url"] if config.connect_by == "ws" else endpoint["http_url"],
            "mcpServers": {name: {"url": s["url"], "headers": s.get("headers") or {}} for name, s in request.mcp_servers.items()},
            "workspace": str(workspace),
            # A history holds the model's signed thinking, which another model refuses, so a model change starts anew.
            "historyFile": str(HOME / "sessions" / f"{session}-{hashlib.sha256(model.encode()).hexdigest()[:8]}.json"),
        }
        (HOME / "sessions").mkdir(parents=True, exist_ok=True)
        request.trajectory.set_format("browser-use-messages/1")

        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **_PRIVATE_ENV}, start_new_session=True, limit=256 * 1024 * 1024,
        )
        stderr = asyncio.create_task(process.stderr.read())
        result = None
        try:
            async with asyncio.timeout(config.timeout_seconds):
                process.stdin.write(json.dumps(turn).encode())
                process.stdin.close()
                async for line in process.stdout:
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue  # output of browser-use's libraries, not one of agent.py's lines
                    if isinstance(message, dict) and message.get("type") == "result":
                        result = message["result"]
                await process.wait()
        except TimeoutError:
            return TaskResult.failure("browser_use.timeout", f"no answer within {config.timeout_seconds}s")
        finally:
            # On a timeout or a cancel, the agent stops its browser daemon, then goes with everything it started.
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(process.wait()), STOP_SECONDS)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

        if result is None:
            error = (await stderr).decode(errors="replace")[-4000:] or f"agent.py exited {process.returncode}"
            return TaskResult.failure("browser_use.crashed", error, error_type="infra_error")
        tool_calls = 0
        for message in result["messages"]:
            request.trajectory.append(message)
            tool_calls += sum(part.get("part_kind") == "tool-call" for part in message.get("parts") or [])
        usage = result["usage"]
        builder = TaskResult.builder()
        if result["status"] == "completed":
            builder.succeeded()
        else:
            builder.failed(f"browser_use.{result['status']}", result.get("error") or result["status"])
        return (
            builder.add_text(result["text"] or result.get("error") or result["status"])
            .session_ref(session)
            .usage(Usage(
                tool_call_count=tool_calls,
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                total_tokens=usage["input"] + usage["output"],
                provider_details={"steps": result["steps"], "model": result["model"], "requests": usage["requests"],
                                  "cache_read_tokens": usage["cache_read"], "cache_write_tokens": usage["cache_write"],
                                  "cost_usd": result.get("cost_usd"), "success": result.get("success")},
            ))
            .build()
        )


agent = BrowserUseAgent()
app = agent.create_app()
AGENT_CARD = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
