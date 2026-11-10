"""browser-use's macOS harness as a computer driver on macOS. macos-harness reads an app as numbered text and acts on a
number or a pixel, in the background, through eleven commands its skill teaches an agent to run in a shell, chaining
the actions it's sure of and the read after them with ``&&``. Agents get one tool, ``macos_harness``, which runs the
pinned harness on a list of command lines in order, the first that fails stopping the rest, with no shell between;
instructions that map its shell to the tool and send the agent to its skill, which it then reads whole, as the vendor's
own setup has it load the skill, since Claude Code cuts a server's instructions at 4,096 characters; and a read's
screenshot as an image beside its text, since the file stays on the computer. The computer/v1 operations are
MacOSDriver's. This process's Python is granted Accessibility, Post Event and Screen Recording, and the harness, which
it starts, has them as its own. Telemetry is off. It needs ``agentenv-framework-protocol[computer,macos]`` and ``macos-harness`` where it runs."""
from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
from pathlib import Path
from typing import Annotated

from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import Field

from agentenv_protocol import server_instructions, tool
from agentenv_protocol.computer import MacOSDriver, ReadyCheck, computer_driver

VERSION = "0.2.1"
HERE = Path(__file__).resolve().parent
#: The skill the pinned harness prints, as it prints it.
SKILL = HERE / "skill.md"
#: The harness's console script, in the driver's venv.
EXECUTABLE = HERE / "venv" / "bin" / "macos-harness"
#: The commands a line may start with: the eleven its skill teaches, the skill, and its help and version.
COMMANDS = ("list_apps", "get_app_state", "click", "perform_secondary_action", "set_value", "select_text", "scroll",
            "drag", "press_key", "type_text", "paste", "skill", "help", "--help", "--version")
#: The line a read names the screenshot it kept on, and the image type of each file it keeps.
SCREENSHOT = re.compile(r"^Screenshot \(\d+x\d+ px\): (.+)$", re.MULTILINE)
MIME = {".jpeg": "image/jpeg", ".jpg": "image/jpeg", ".png": "image/png"}
#: How many command lines one call runs at most, and how long each may take.
MAX_COMMANDS, COMMAND_TIMEOUT_S = 20, 120
#: What of the driver's environment the harness gets, the session user's own and none of a deploy's values, and what's
#: added: no telemetry.
KEPT = ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")
ADDED = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "DO_NOT_TRACK": "1"}
PREFACE = (f"macos-harness {VERSION} runs on this computer through the macos_harness tool, with no shell. Its whole "
           "guide is its skill: before your first command, run [[\"skill\"]] and follow it. Where the skill shows "
           "`macos-harness ...` in a shell, call macos_harness with the words after `macos-harness`; a chain joined "
           "with `&&` is a list of command lines in one call. Text needs no quoting or heredoc: pass it as one word, "
           "such as [\"paste\", \"--app\", \"TextEdit\", \"--text\", \"line one\\nline two\"]. A read's screenshot, "
           "a --keep copy too, comes back as an image; the files the harness names stay on this computer, which you don't "
           "share, so open none from where you run.")


@computer_driver(name="macos-harness", supports=[{"os": {"type": "macos", "display": ["aqua"]}, "arch": ["arm64"]}])
class MacosHarness(MacOSDriver):
    def __init__(self) -> None:
        super().__init__()
        self._pinned: list[ReadyCheck] | None = None

    @tool()
    async def macos_harness(self, commands: Annotated[list[list[str]], Field(
            min_length=1, max_length=MAX_COMMANDS,
            description="Command lines, each the list of words after `macos-harness`, such as "
                        "[[\"click\", \"--app\", \"BenchLab\", \"--element_index\", \"5\"], [\"get_app_state\", "
                        "\"--app\", \"BenchLab\"]]")]) -> CallToolResult:
        """Run macos-harness on this computer: each command line in order, the first that fails stopping the rest. No
        shell: no pipes, globs, variables or redirection. A read's screenshot comes back as an image."""
        if refused := [words for words in commands if not words or words[0] not in COMMANDS]:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=(
                f"a command line starts with one of {', '.join(COMMANDS)}; not {refused[0][:1]}. Nothing ran."))])
        content: list[TextContent | ImageContent] = []
        for words in commands:
            code, out = await _run(words)
            content.append(TextContent(type="text", text=f"$ macos-harness {shlex.join(words)}\n{out.rstrip()}"
                                       + (f"\n(exit {code})" if code else "")))
            content += [image for path in SCREENSHOT.findall(out) if (image := _image(Path(path.strip())))]
            if code:
                return CallToolResult(isError=True, content=content)
        return CallToolResult(content=content)

    @server_instructions
    def harness_instructions(self) -> str:
        return PREFACE

    async def ready_checks(self) -> list[ReadyCheck]:
        """The pinned harness, printing the pinned skill, checked until they pass once, and the TCC permissions, every
        time."""
        if self._pinned is None or not all(check.ok for check in self._pinned):
            _, version = await _run(["--version"])
            _, skill = await _run(["skill"])
            same = skill == SKILL.read_text()
            ok = version.strip() == VERSION and same
            self._pinned = [ReadyCheck("macos-harness", ok, f"macos-harness {VERSION} and its skill, as pinned" if ok else
                                       f"version {version.strip()!r}, not the pinned {VERSION}" if same else
                                       f"its skill isn't the one {SKILL.name} pins")]
        return [*self._pinned, *await super().ready_checks()]

    async def bundled_skill(self) -> tuple[str, str, dict[str, str]]:
        _, frontmatter, body = SKILL.read_text().split("---\n", 2)
        description = next(line.partition(":")[2].strip() for line in frontmatter.splitlines()
                           if line.startswith("description:"))
        return description, body.lstrip("\n"), {}


async def _run(words: list[str]) -> tuple[int, str]:
    """``macos-harness`` run with ``words``: its exit code and its output, stderr after stdout. One that outlives
    ``COMMAND_TIMEOUT_S`` is killed and answers 124."""
    try:
        process = await asyncio.create_subprocess_exec(
            str(EXECUTABLE), *words, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env={**{name: os.environ[name] for name in KEPT if name in os.environ}, **ADDED})
    except OSError as e:
        return 127, f"macos-harness can't start: {e}"
    try:
        out, err = await asyncio.wait_for(process.communicate(), COMMAND_TIMEOUT_S)
    except TimeoutError:
        await _stopped(process)
        return 124, f"macos-harness {words[0]} didn't finish within {COMMAND_TIMEOUT_S} s"
    except BaseException:
        await _stopped(process)
        raise
    return process.returncode or 0, (out + err).decode(errors="replace")


async def _stopped(process: asyncio.subprocess.Process) -> None:
    """``process`` killed and reaped, so a call cancelled or out of time acts no more."""
    if process.returncode is None:
        process.kill()
    await asyncio.shield(process.wait())


def _image(path: Path) -> ImageContent | None:
    """The screenshot at ``path`` as an image, when it's one the harness keeps and still there."""
    mime = MIME.get(path.suffix.lower())
    if mime is None or not path.is_file():
        return None
    return ImageContent(type="image", data=base64.b64encode(path.read_bytes()).decode(), mimeType=mime)
