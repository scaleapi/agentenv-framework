"""The A2A agent the integration suites deploy when they need an agent but not a model.

Deterministic: it echoes the prompt and records the echo in the framework's trajectory log, which a
reader can follow while the turn runs; a ``live-steps <n>`` in its prompt has it first append ``n`` step
events (at most ``MAX_LIVE_STEPS``), a second apart. Agent-config, mcp-config, trajectory and triggers
come from the agentenv-protocol framework. ``model_params`` is a write-only config field so the
agent-config negotiation and its redaction on read-back can be observed.

It also moves files both ways. After the echo it adds a ``read <name> over <scheme>: <text>`` line for
each file part it is sent, fetching only inline bytes and HTTP(S) URLs (``could not read ...`` otherwise),
and it answers each ``send-file <uri>`` line of its prompt with a file part naming that URI.

It installs skills under ``~/skills/<name>/``, a bundle's files downloaded through their read grants, and a
``list-skills`` line in its prompt adds a ``skill <name>: <path>=<sha256 prefix>, ...`` line for each one.
A ``run <command>`` line has it run that command in its container and add a ``ran <command>: <output>`` line.
"""

import asyncio
import base64
import contextlib
import hashlib
import os
import re
import signal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    SKILL_CONFIG_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
    AgentConfig,
    AgentEnvAgent,
    AgentIdentity,
    BundleSkillRequest,
    FilePart,
    InlineSkillRequest,
    TaskRequest,
    TaskResult,
    TextPart,
    Usage,
    WriteOnly,
    a2a_agent,
    download,
    enable,
    extension,
)

SEND_FILE = "send-file "
LIST_SKILLS = "list-skills"
RUN = "run "
RUN_TIMEOUT_SECONDS = 60
SKILLS_DIR = Path.home() / "skills"
LIVE_STEPS = re.compile(r"live-steps (\d+)")
LIVE_STEP_SECONDS = 1
# Bounds how long a prompt can hold up a turn.
MAX_LIVE_STEPS = 10
TRAJECTORY_FORMAT = "agentenv-echo-agent/v1"
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


async def _ran(command: str) -> str:
    """``command``'s output, stdout and stderr together, run by the shell in a process group of its own, which is
    stopped with whatever it started if the command times out or the request is cancelled."""
    process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    output = None
    try:
        output, _ = await asyncio.wait_for(process.communicate(), RUN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        pass
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    if output is None:
        return f"ran {command}: timed out after {RUN_TIMEOUT_SECONDS}s"
    return f"ran {command}: {output.decode(errors='replace').strip()}"


def _skills() -> list[str]:
    """One line per installed skill: each file's path and the first 12 hex digits of its sha256."""
    if not SKILLS_DIR.is_dir():
        return []
    return [f"skill {folder.name}: " + ", ".join(
        f"{path.relative_to(folder).as_posix()}={hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        for path in sorted(folder.rglob("*")) if path.is_file()) for folder in sorted(SKILLS_DIR.iterdir())]


@a2a_agent(
    identity=AgentIdentity(
        name="agentenv-echo-agent",
        description="Deterministic agent for the agent-env integration suites: echoes its prompt.",
        version="1.0.0",
    ),
    config=EchoAgentConfig,
    extensions=(MCP_CONFIG_V1, enable(TRAJECTORY_V1, live=True), TRIGGERS_V1),
)
class EchoAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[EchoAgentConfig]) -> TaskResult:
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        prompt = "\n".join(part.text for part in request.parts if isinstance(part, TextPart))
        if steps := LIVE_STEPS.search(prompt):
            for step in range(min(int(steps.group(1)), MAX_LIVE_STEPS)):
                request.trajectory.append({"type": "step", "step": step})
                await asyncio.sleep(LIVE_STEP_SECONDS)
        reply = f"Echo: {prompt}"
        reply = "\n".join([reply, *[await _read(part) for part in request.parts if isinstance(part, FilePart)]])
        if LIST_SKILLS in prompt.splitlines():
            reply = "\n".join([reply, *_skills()])
        runs = [line.removeprefix(RUN).strip() for line in prompt.splitlines() if line.startswith(RUN)]
        reply = "\n".join([reply, *[await _ran(command) for command in runs]])
        sends = [line.removeprefix(SEND_FILE).strip() for line in prompt.splitlines() if line.startswith(SEND_FILE)]
        files = [FilePart(uri=uri, name=uri.rsplit("/", 1)[-1], mime_type="text/plain") for uri in sends]
        request.trajectory.append({"type": "echo", "input": prompt, "output": reply})
        return (
            TaskResult.builder()
            .succeeded()
            .parts([TextPart(text=reply), *files])
            .usage(Usage(tool_call_count=0, input_tokens=len(prompt), output_tokens=len(reply)))
            .build()
        )

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> dict[str, str]:
        (SKILLS_DIR / request.name).mkdir(parents=True)
        (SKILLS_DIR / request.name / "SKILL.md").write_text(request.skill_md)
        return {"name": request.name}

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> dict[str, str]:
        for item in request.skill_bundle.files:
            await download(item.object, SKILLS_DIR / request.name / item.path)
        return {"name": request.name}


if __name__ == "__main__":
    EchoAgent().serve()
