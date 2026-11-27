"""The Codex CLI the agent's package-lock.json pins, against what its bridge relies on to tell the model the env's date
while the CLI runs on real time: every place the CLI writes its date or time into a model call is one the turn's relay
rewrites, so the model is sent the env's date and no copy of the machine's, whether the turn starts a thread or resumes
one. Bumping the CLI runs this against the new version, so a version that moves or adds a date fails here."""

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tst.unit.agents.codex.agent import AGENT_DIR, codex_bridge, codex_config, codex_date
from tst.util import fake_responses

pytestmark = [pytest.mark.integration]

VERSION = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["@openai/codex"]
ENV_TODAY, ENV_NOW = "1950-01-01", "1950-01-01 12:00:00"
_DAY_S = 86400


@pytest.fixture(scope="module")
def codex_bin(tmp_path_factory) -> Path:
    """The CLI installed as the image installs it: npm ci from the lockfile."""
    npm = shutil.which("npm")
    assert npm and shutil.which("node"), "node and npm install the pinned Codex CLI"
    folder = tmp_path_factory.mktemp("codex")
    for name in ("package.json", "package-lock.json"):
        shutil.copy(AGENT_DIR / name, folder / name)
    subprocess.run([npm, "ci", "--ignore-scripts", "--omit=dev", "--no-audit", "--no-fund", "--loglevel=error"],
                   cwd=folder, check=True, timeout=600)
    return folder / "node_modules" / ".bin"


def _machine_dates() -> set[str]:
    """This machine's date for yesterday, today and tomorrow, as the CLI writes one (``%Y-%m-%d``, in this zone and in
    UTC), so a turn across midnight is still caught."""
    return {time.strftime("%Y-%m-%d", read(time.time() + offset))
            for offset in (-_DAY_S, 0, _DAY_S) for read in (time.localtime, time.gmtime)}


def _texts(body: dict) -> list[str]:
    texts = [body["instructions"]] if isinstance(body.get("instructions"), str) else []
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type", "message") == "message":
            content = item.get("content")
            parts = content if isinstance(content, list) else [{"text": content}]
            texts += [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    return texts


def test_a_turn_on_an_armed_clock_tells_the_model_the_env_date_and_no_other(codex_bin, tmp_path, monkeypatch):
    async def env_now() -> tuple[str, str]:
        return ENV_TODAY, ENV_NOW

    for name in ("HOME", "CODEX_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("PATH", f"{codex_bin}:{subprocess.os.environ['PATH']}")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-fake")
    monkeypatch.setattr(codex_bridge, "SESSIONS_DIR", tmp_path / "codex_home" / "sessions")
    monkeypatch.setattr(codex_bridge, "WORKSPACE_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(codex_date, "clock_armed", lambda: True)
    monkeypatch.setattr(codex_date, "_env_now", env_now)
    (tmp_path / "work").mkdir()

    respond = fake_responses.tool_then_answer("exec_command", {"cmd": "echo hello"}, prefix="the shell said: ")
    with fake_responses.FakeResponses(respond=respond) as endpoint:
        monkeypatch.setenv("LITELLM_BASE_URL", endpoint.url)
        codex_config.write_base_config(subprocess.os.environ)
        first = asyncio.run(codex_bridge.run_codex("Run echo hello.", None, model="gpt-x", timeout_s=300))
        assert not first.error and first.session_id, (first.error, first.trajectory[-3:])
        second = asyncio.run(codex_bridge.run_codex("Again.", first.session_id, model="gpt-x", timeout_s=300))

    assert not second.error, (second.error, second.trajectory[-3:])
    sent = [seen.body for seen in endpoint.calls()]
    assert len(sent) >= 4 and any("hello" in result for result in endpoint.tool_results()), endpoint.tool_results()
    dated = [text for body in sent for text in _texts(body) if "<current_date" in text]
    assert dated and all(f"<current_date>{ENV_TODAY}</current_date>" in text for text in dated), (
        f"Codex {VERSION} no longer writes its date where codex_date's _CURRENT_DATE finds it: {dated}")
    assert all(text.strip().startswith("<environment_context>") and text.strip().endswith("</environment_context>")
               for text in dated), f"Codex {VERSION} no longer writes its environment context as a text of its own"
    machine = _machine_dates()
    for body in sent:
        found = [date for date in machine if date in json.dumps(body)]
        assert not found, f"Codex {VERSION} wrote this machine's date {found} where the relay doesn't rewrite it"
