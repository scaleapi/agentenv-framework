"""Vercel's agent-desktop as a computer driver on macOS. agent-desktop reads apps' accessibility trees and acts on what
it finds; its own MCP server, ``agent-desktop mcp``, serves each of its commands as a tool, and agents get what that
server serves at its pinned version (``mcp.json``): its instructions and its tools, each call forwarded to a server of
its own and answered as the server answers it, with each screenshot the call saved to a file beside the answer, since
the agent can't read the computer's files; agent-desktop's skill leads the computer's. The computer/v1
operations are MacOSDriver's. This process's Python is granted Accessibility, Post Event and Screen Recording, and
agent-desktop, which it starts, has them as its own. A tool can't pass agent-desktop's global flags, so the server's
come from the driver's ``headed`` setting: off, agent-desktop's default, it acts through the accessibility API alone and
refuses what needs physical input (double clicks, hovers, drags, the mouse commands); on, it allows those and prefers
physical input, moving the pointer and focus. It needs ``agentenv-framework-protocol[computer,macos]`` and
agent-desktop's release where it runs."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent

from agentenv_protocol import EnvironmentTool, ForwardedTool, server_instructions
from agentenv_protocol.computer import ComputerSettings, MacOSDriver, ReadyCheck, computer_driver

HERE = Path(__file__).resolve().parent
#: agent-desktop's executables, as its release unpacks into bin/, and the sha256 of each at the pinned version, 1.2.0.
BIN = HERE / "bin"
EXECUTABLES = {"agent-desktop": "dec75e0bba0d1f9b589bdb88188770ad1a199cde9e02cf0a5b94175491f738e9",
               "agent-desktop-macos-helper": "4a795a40ba7fc93b726b0834342f0225779c30d83e28f1f219db479860c38a78"}
EXECUTABLE = BIN / "agent-desktop"
#: What ``agent-desktop mcp`` says at connect and lists at the pinned version: its instructions and its tools.
MANIFEST = HERE / "mcp.json"
#: What of the driver's environment agent-desktop gets: the session user's own, none of a deploy's values.
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")
#: The PATH it runs, and launches apps, with.
PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
#: How long one of its commands that reads its skills may take.
SKILL_TIMEOUT_S = 60
#: The most bytes of saved screenshots one answer carries.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
#: What the instructions add to agent-desktop's: the computer's files stay on it.
FILES = ("Files these tools write stay on this computer, which you don't share: a screenshot saved to an output_path, "
         "in a batch too, comes back with its answer, up to 8 MB of them an answer, and other files, such as a trace "
         "export, can't be read from where you run.")


@computer_driver(name="agent-desktop", supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}])
class AgentDesktop(MacOSDriver):
    class Settings(ComputerSettings):
        headed: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._pinned: list[ReadyCheck] | None = None
        self._skill: tuple[str, str, dict[str, str]] | None = None

    def forwarded_tools(self) -> list[ForwardedTool]:
        return [ForwardedTool(EnvironmentTool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]),
                              functools.partial(self._forward, t["name"]), t.get("annotations"), t.get("outputSchema"))
                for t in _manifest()["tools"]]

    @server_instructions
    def agent_desktop_instructions(self) -> str:
        return f"{_manifest()['instructions']}\n\n{FILES}"

    async def ready_checks(self) -> list[ReadyCheck]:
        """agent-desktop's executables and the tools its server lists, as pinned, checked until they pass once, and the
        TCC permissions, every time."""
        if self._pinned is None or not all(check.ok for check in self._pinned):
            self._pinned = await self._pinned_checks()
        return [*self._pinned, *await super().ready_checks()]

    async def bundled_skill(self) -> tuple[str, str, dict[str, str]]:
        """agent-desktop's core skill, as its executable embeds it, with its references, and its guide for this OS and
        that guide's references under ``platform/``, as its other files."""
        if self._skill is None:
            listed = {skill["name"]: skill for skill in (await _agent_desktop("skills", "list"))["skills"]}
            platform = next(skill for skill in listed.values() if skill.get("current"))
            files = {ref: (await _agent_desktop("skills", "get", "desktop", ref))["content"]
                     for ref in listed["agent-desktop"]["references"]}
            files["platform/SKILL.md"] = (await _agent_desktop("skills", "get", "platform"))["content"]
            files.update({f"platform/{ref}": (await _agent_desktop("skills", "get", "platform", ref))["content"]
                          for ref in platform["references"]})
            _, frontmatter, body = (await _agent_desktop("skills", "get", "desktop"))["content"].split("---\n", 2)
            self._skill = _description(frontmatter), body.lstrip("\n"), files
        return self._skill

    async def _forward(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """A call to one of agent-desktop's tools, on a server of its own, answered as the server answers it. A server
        that fails before answering may have run the command."""
        try:
            async with self._server() as session:
                result = await session.call_tool(tool, arguments)
        except Exception as e:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=(
                f"agent-desktop's server failed before it answered, so {tool} may have run: {type(e).__name__}: {e}"))])
        return _with_images(result)

    @contextlib.asynccontextmanager
    async def _server(self) -> AsyncIterator[ClientSession]:
        """``agent-desktop mcp`` with the global flags the settings give, connected and initialized; it ends with the
        block."""
        flags = ["--headed"] if self.settings.headed else []
        server = StdioServerParameters(command=str(EXECUTABLE), args=["mcp", *flags], env=_environment())
        async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            yield session

    async def _pinned_checks(self) -> list[ReadyCheck]:
        digests = await asyncio.to_thread(_digests)
        differ = sorted(name for name, digest in digests.items() if digest != EXECUTABLES[name])
        checks = [ReadyCheck("agent-desktop", not differ, "the pinned executables" if not differ else
                             f"these differ from the pinned agent-desktop 1.2.0: {', '.join(differ)}")]
        if differ:
            return checks
        try:
            async with self._server() as session:
                listed = (await session.list_tools()).tools
        except Exception as e:
            return [*checks, ReadyCheck("agent-desktop tools", False, f"its server didn't list them: {e}")]
        running = {t.name: t.inputSchema for t in listed}
        pinned = {t["name"]: t["inputSchema"] for t in _manifest()["tools"]}
        changed = sorted(name for name in running.keys() | pinned.keys() if running.get(name) != pinned.get(name))
        return [*checks, ReadyCheck("agent-desktop tools", not changed, f"{len(running)} tools, as pinned" if not changed
                                    else f"these tools differ from {MANIFEST.name}: {', '.join(changed)}")]


def _with_images(result: CallToolResult) -> CallToolResult:
    """``result`` with each screenshot its answer saved to a file, up to ``MAX_IMAGE_BYTES`` in all, and a line naming
    the saved ones it doesn't carry."""
    images, total, past, unread = [], 0, [], []
    for path in dict.fromkeys(_saved(result)):
        try:
            if total + path.stat().st_size > MAX_IMAGE_BYTES:
                past.append(path)
                continue
            data = path.read_bytes()
        except OSError:
            unread.append(path)
            continue
        total += len(data)
        images.append(ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType="image/png"))
    notes = [f"{label}: {', '.join(map(str, paths))}" for label, paths in (
        ("Saved but not attached, past the most one answer carries", past), ("Saved but not readable", unread)) if paths]
    content = [*images, *(TextContent(type="text", text=note) for note in notes)]
    return result.model_copy(update={"content": [*result.content, *content]}) if content else result


def _saved(result: CallToolResult) -> list[Path]:
    """The files ``result``'s screenshots were saved to, a batch's included: each one's ``data.path``, a relative one
    from the server's working directory, which is this process's."""
    found: list[Path] = []

    def walk(answer: Any) -> None:
        if not isinstance(answer, dict) or not isinstance(data := answer.get("data"), dict):
            return
        if answer.get("command") == "screenshot" and data.get("path"):
            found.append(Path(data["path"]))
        for each in data.get("results") or []:
            walk(each)

    for block in result.content:
        if isinstance(block, TextContent):
            with contextlib.suppress(ValueError):
                walk(json.loads(block.text))
    return found


@functools.cache
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _environment() -> dict[str, str]:
    return {**{name: os.environ[name] for name in KEPT if name in os.environ}, "PATH": PATH}


def _digests() -> dict[str, str | None]:
    return {name: hashlib.sha256((BIN / name).read_bytes()).hexdigest() if (BIN / name).is_file() else None
            for name in EXECUTABLES}


async def _agent_desktop(*words: str) -> dict:
    """What agent-desktop answers for ``words``: its envelope's data."""
    process = await asyncio.create_subprocess_exec(str(EXECUTABLE), *words, env=_environment(),
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(process.communicate(), SKILL_TIMEOUT_S)
    envelope = json.loads(out or b"{}")
    if not envelope.get("ok"):
        raise RuntimeError(f"agent-desktop {' '.join(words)}: {envelope.get('error') or err.decode()}")
    return envelope["data"]


def _description(frontmatter: str) -> str:
    """A skill's description, as YAML reads it from its frontmatter: on its line, or folded over the indented lines
    under it."""
    lines = frontmatter.splitlines()
    at = next(n for n, line in enumerate(lines) if line.startswith("description:"))
    if (first := lines[at].partition(":")[2].strip()) not in ("", ">", ">-", "|", "|-"):
        return first
    folded = []
    for line in lines[at + 1:]:
        if line.strip() and not line.startswith(" "):
            break
        folded.append(line.strip())
    return " ".join(part for part in folded if part)
