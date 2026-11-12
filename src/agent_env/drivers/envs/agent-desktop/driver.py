"""Vercel's agent-desktop as a computer driver on macOS, on Windows and on an X11 desktop on Linux. agent-desktop reads
apps' accessibility trees, UI Automation's on Windows and AT-SPI's on Linux, and acts on what it finds; its own MCP
server, ``agent-desktop mcp``, serves each of its commands as a tool, and agents get what that server serves at its
pinned version (``mcp.json``): its instructions and its tools, each call forwarded to a server of its own and answered
as the server answers it, with each screenshot the call saved to a file beside the answer, since the agent can't read
the computer's files; agent-desktop's skill leads the computer's. The computer/v1 operations are MacOSDriver's on
macOS, WindowsDriver's on Windows and X11Driver's on Linux. On macOS this process's Python is granted Accessibility,
Post Event and Screen Recording, and agent-desktop, which it starts, has them as its own. A tool can't pass
agent-desktop's global flags, so the server's come from the driver's ``headed`` setting: off, agent-desktop's default,
it acts through the accessibility API alone and refuses what needs physical input (double clicks, hovers, drags, the
mouse commands); on, it allows those and prefers physical input, moving the pointer and focus. It needs
``agentenv-framework-protocol[computer,macos]`` on macOS, ``[computer,windows]`` on Windows or ``[computer,x11]`` on
Linux, and agent-desktop's release where it runs."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, NamedTuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ImageContent, TextContent

from agentenv_protocol import ForwardedTool, computer, server_instructions
from agentenv_protocol.computer import ComputerSettings, ReadyCheck, computer_driver


class _OS(NamedTuple):
    """What differs by OS: the protocol driver the computer/v1 operations run on, agent-desktop's executable in bin/,
    what of the driver's environment it gets beside the session user's own (on Windows the system's folders, which COM
    and the apps it launches need; on Linux its display and the session bus AT-SPI is found on), and the PATH it runs
    and launches apps with, ``{SystemRoot}`` filled in from the environment."""
    driver: str
    executable: str
    kept: tuple[str, ...]
    path: str


#: The release env.toml downloads for every OS, whose executables' sha256s EXECUTABLES pins.
VERSION = "1.2.0"
OSES = {
    "macos": _OS("MacOSDriver", "agent-desktop", (), "/usr/bin:/bin:/usr/sbin:/sbin"),
    "windows": _OS("WindowsDriver", "agent-desktop.exe", (
        "USERNAME", "USERDOMAIN", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
        "SystemRoot", "SystemDrive", "windir", "ComSpec", "PATHEXT", "ProgramData", "ProgramFiles", "ProgramFiles(x86)",
        "ProgramW6432", "CommonProgramFiles", "PUBLIC", "ALLUSERSPROFILE", "COMPUTERNAME"),
        r"{SystemRoot}\System32;{SystemRoot};{SystemRoot}\System32\Wbem;{SystemRoot}\System32\WindowsPowerShell\v1.0"),
    "linux": _OS("X11Driver", "agent-desktop", ("DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"),
                 "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"),
}
OS_NAME = {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux")
OS = OSES[OS_NAME]
OSDriver = getattr(computer, OS.driver)

HERE = Path(__file__).resolve().parent
#: agent-desktop's executables, as its release for each OS and arch unpacks into bin/, and the sha256 of each.
BIN = HERE / "bin"
EXECUTABLES = {
    "macos/arm64": {
        "agent-desktop": "dec75e0bba0d1f9b589bdb88188770ad1a199cde9e02cf0a5b94175491f738e9",
        "agent-desktop-macos-helper": "4a795a40ba7fc93b726b0834342f0225779c30d83e28f1f219db479860c38a78",
    },
    "linux/amd64": {
        "agent-desktop": "da52a1eb2152cdc21b756ab20e6208d242da8062c73cd20bc8a8ae2a0384392c",
        "agent-desktop-linux-clipboard-owner": "3e96f92b805c0d1f9de309935d62f4189b8e11e1385ad022bb878487bb3942ec",
    },
    "linux/arm64": {
        "agent-desktop": "915de59a8b3d22c9ad33023740bbcc5fe0b4fa830de9587bbf761237eafeb76f",
        "agent-desktop-linux-clipboard-owner": "a6bfde6778dadcf9fd04ba94de17cdf205eb0b23b643c3f34073d264d2117cc3",
    },
    "windows/amd64": {
        "agent-desktop.exe": "d274667031322e50b9c2b1ab81c2ca46835d52cc932769a5c34bd4d481fd94c3",
    },
}
EXECUTABLE = BIN / OS.executable
#: What ``agent-desktop mcp`` says at connect and lists at the pinned version, the same on macOS, Windows and Linux:
#: its instructions and its tools.
MANIFEST = HERE / "mcp.json"
#: What of the driver's environment agent-desktop gets on every OS: the session user's own; none of a deploy's values.
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")
#: How long one of its commands that reads its skills may take.
SKILL_TIMEOUT_S = 60
#: The most bytes of saved screenshots one answer carries.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
#: What the instructions add to agent-desktop's: the computer's files stay on it.
FILES = ("Files these tools write stay on this computer, which you don't share: a screenshot saved to an output_path, "
         "in a batch too, comes back with its answer, up to 8 MB of them an answer, and other files, such as a trace "
         "export, can't be read from where you run.")


@computer_driver(name="agent-desktop", supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]},
                                                 {"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]},
                                                 {"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class AgentDesktop(OSDriver):
    class Settings(ComputerSettings):
        headed: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._pinned: list[ReadyCheck] | None = None
        self._skill: tuple[str, str, dict[str, str]] | None = None

    def forwarded_tools(self) -> list[ForwardedTool]:
        return [ForwardedTool.listed(t, functools.partial(self._forward, t["name"])) for t in _manifest()["tools"]]

    @server_instructions
    def agent_desktop_instructions(self) -> str:
        return f"{_manifest()['instructions']}\n\n{FILES}"

    async def ready_checks(self) -> list[ReadyCheck]:
        """agent-desktop's executables and the tools its server lists, as pinned, checked until they pass once, and the
        OS's own checks, the TCC permissions on macOS, the input desktop on Windows and XTEST on Linux, every time."""
        if self._pinned is None or not all(check.ok for check in self._pinned):
            self._pinned = await self._pinned_checks()
        return [*self._pinned, *await super().ready_checks()]

    async def bundled_skill(self) -> tuple[str, str, dict[str, str]]:
        """agent-desktop's core skill, as its executable embeds it, with its references, and its guide for this OS and
        that guide's references under ``platform/``, as its other files."""
        if self._skill is None:
            listed = {skill["name"]: skill for skill in (await _agent_desktop("skills", "list"))["skills"]}
            platform = next(skill for skill in listed.values() if skill.get("current"))
            files = {ref: await _skill_file("desktop", ref) for ref in listed["agent-desktop"]["references"]}
            files["platform/SKILL.md"] = await _skill_file("platform")
            files.update({f"platform/{ref}": await _skill_file("platform", ref) for ref in platform["references"]})
            _, frontmatter, body = (await _skill_file("desktop")).split("---\n", 2)
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
                f"agent-desktop's server failed before it answered, so {tool} may have run: {_reason(e)}"))])
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
        pinned = EXECUTABLES[f"{self.platform.os}/{self.platform.arch}"]
        digests = await asyncio.to_thread(_digests, pinned)
        differ = sorted(name for name, digest in digests.items() if digest != pinned[name])
        checks = [ReadyCheck("agent-desktop", not differ, "the pinned executables" if not differ else
                             f"these differ from the pinned agent-desktop {VERSION}: {', '.join(differ)}")]
        if differ:
            return checks
        try:
            async with self._server() as session:
                listed = (await session.list_tools()).tools
        except Exception as e:
            return [*checks, ReadyCheck("agent-desktop tools", False, f"its server didn't list them: {_reason(e)}")]
        running = {t.name: t.inputSchema for t in listed}
        pinned = {t["name"]: t["inputSchema"] for t in _manifest()["tools"]}
        changed = sorted(name for name in running.keys() | pinned.keys() if running.get(name) != pinned.get(name))
        return [*checks, ReadyCheck("agent-desktop tools", not changed, f"{len(running)} tools, as pinned" if not changed
                                    else f"these tools differ from {MANIFEST.name}: {', '.join(changed)}")]


def _reason(e: BaseException) -> str:
    """What failed, under the task groups the MCP client wraps it in."""
    while isinstance(e, BaseExceptionGroup) and len(e.exceptions) == 1:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}"


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
    return {**{name: os.environ[name] for name in (*KEPT, *OS.kept) if name in os.environ},
            "PATH": OS.path.format_map(os.environ)}


def _digests(names) -> dict[str, str | None]:
    return {name: hashlib.sha256((BIN / name).read_bytes()).hexdigest() if (BIN / name).is_file() else None
            for name in names}


async def _agent_desktop(*words: str) -> dict:
    """What agent-desktop answers for ``words``: its envelope's data."""
    process = await asyncio.create_subprocess_exec(str(EXECUTABLE), *words, env=_environment(),
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(process.communicate(), SKILL_TIMEOUT_S)
    envelope = json.loads(out or b"{}")
    if not envelope.get("ok"):
        raise RuntimeError(f"agent-desktop {' '.join(words)}: "
                           f"{envelope.get('error') or err.decode() or f'it exited {process.returncode}'}")
    return envelope["data"]


async def _skill_file(*words: str) -> str:
    """One of agent-desktop's skill files, with LF line endings, which its Windows build embeds as CRLF."""
    return (await _agent_desktop("skills", "get", *words))["content"].replace("\r\n", "\n")


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
