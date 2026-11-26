"""The Claude Code CLI the agent's Dockerfile pins tells the model the new date when it changes between turns: a resumed
session keeps its first turn's "Today's date is" line as history, and the next turn's message says "The date has
changed. Today's date is now ...". On an env clock that jumps between turns, that is how the model learns the jump.
The date here changes by time zone, 26 hours apart, so no clock is faked; bumping the CLI runs this against the new
version."""

import hashlib
import json
import os
import platform
import re
import subprocess
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tst.unit.agents.claude_code.agent import AGENT_DIR, claude_bridge
from tst.util.fake_anthropic import FakeAnthropic

pytestmark = [pytest.mark.integration]

VERSION = re.search(r"^ARG CLAUDE_CODE_VERSION=(\S+)$", (AGENT_DIR / "Dockerfile").read_text(), re.MULTILINE).group(1)
RELEASES = "https://downloads.claude.ai/claude-code-releases"
PLATFORMS = {("Linux", "x86_64"): "linux-x64", ("Linux", "aarch64"): "linux-arm64",
             ("Darwin", "arm64"): "darwin-arm64", ("Darwin", "x86_64"): "darwin-x64"}
# UTC+14 and UTC-12: whatever the hour, the two zones' dates differ.
FIRST_ZONE, SECOND_ZONE = "Etc/GMT-14", "Etc/GMT+12"
TURN_TIMEOUT_S = 180


@pytest.fixture(scope="module")
def claude(tmp_path_factory) -> Path:
    """The pinned CLI's build for this machine, checked against its release manifest, as the image installs it."""
    name = PLATFORMS[(platform.system(), platform.machine())]
    with urllib.request.urlopen(f"{RELEASES}/{VERSION}/manifest.json", timeout=60) as answer:
        checksum = json.load(answer)["platforms"][name]["checksum"]
    binary = tmp_path_factory.mktemp("claude-code") / "claude"
    with urllib.request.urlopen(f"{RELEASES}/{VERSION}/{name}/claude", timeout=600) as answer:
        binary.write_bytes(answer.read())
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == checksum, f"Claude Code {VERSION} for {name}"
    binary.chmod(0o755)
    return binary


def _texts(message: dict) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    return [block["text"] for block in content or [] if isinstance(block, dict) and isinstance(block.get("text"), str)]


def _date(zone: str) -> str:
    return datetime.now(ZoneInfo(zone)).date().isoformat()


def _turn(claude: Path, home: Path, endpoint: FakeAnthropic, session_id: str, resume: bool, prompt: str,
          zone: str) -> set[str]:
    """One turn with the bridge's own argv, in ``zone``; the dates the zone had while it ran."""
    argv = claude_bridge._command(session_id, resume, model="claude-sonnet-4-6", allowed_tools=None, mcp_servers=None,
                                  system_prompt=None, output_format=None, role=None, effort=None, max_turns=None,
                                  web_tools=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"), "TZ": zone,
           "ANTHROPIC_BASE_URL": endpoint.url, "ANTHROPIC_API_KEY": "sk-fake", "DISABLE_AUTOUPDATER": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    before = _date(zone)
    done = subprocess.run([str(claude), *argv[1:]], input=prompt, capture_output=True, text=True, env=env, cwd=home,
                          timeout=TURN_TIMEOUT_S)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return {before, _date(zone)}


def test_a_resumed_turn_tells_the_model_the_date_changed(claude, tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    session_id = str(uuid.uuid4())

    with FakeAnthropic(host="127.0.0.1") as endpoint:
        first = _turn(claude, home, endpoint, session_id, False, "FIRST TURN", FIRST_ZONE)
        second = _turn(claude, home, endpoint, session_id, True, "SECOND TURN", SECOND_ZONE)
        calls = [seen.body for seen in endpoint.messages()]

    turns = [body for body in calls if any("SECOND TURN" in text for message in body.get("messages") or []
                                           for text in _texts(message))]
    assert turns, f"Claude Code {VERSION} sent no request for the resumed turn"
    for body in turns:
        messages = body["messages"]
        history = "\n".join(text for message in messages for text in _texts(message))
        assert "FIRST TURN" in history, f"Claude Code {VERSION} resumed without the first turn"
        assert any(f"Today's date is {date}." in history for date in first), (
            f"Claude Code {VERSION}'s first turn named no date of {first}")
        assert any(f"The date has changed. Today's date is now {date}." in history for date in second), (
            f"Claude Code {VERSION} didn't tell the model the date changed to {second}: a clock that jumps between "
            "turns would leave it on the first turn's date")
