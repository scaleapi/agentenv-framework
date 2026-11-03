"""A selection of the CLI's own tools, as `-c` overrides on the turn's CLI: ``"*"`` turns off every setting that brings
one, the `multi_agent_v2` collaboration tools included, ``only`` every setting but those it keeps, and a list to turn off
those it names; config.toml and the MCP servers agent-env gives the agent stay as they are. A selection without
`apply_patch`, which only the model catalog turns on, also points the CLI at a copy of its catalog with the tool off. With
no selection, the CLI gets the overrides it got before. The card's agent-config takes the field, so deploy_agent accepts a selection for
codex, and a tool the CLI doesn't have, or part of a setting's tools, is refused. The selection's type has the
protocol's schema, and its tool names are those of the CLI version package.json pins."""

import asyncio
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import BuiltinTools

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.task_steps.deploy_agent import _check_builtin_tools
from tst.unit.agents.codex.agent import AGENT_DIR, a2a_server, codex_bridge, codex_config

_ALL_OFF = [
    "features.shell_tool=false",
    "tools.update_plan.enabled=false",
    "tools.experimental_request_user_input.enabled=false",
    "features.view_image=false",
    "features.multi_agent_v2=false",
    "features.multi_agent=false",
    "features.goals=false",
    'web_search="disabled"',
]


_CATALOG = {"models": [
    {"slug": "gpt-5.4", "apply_patch_tool_type": "freeform", "context_window": 272000},
    {"slug": "gpt-5.6-terra", "apply_patch_tool_type": "freeform", "shell_type": "shell_command"},
]}


def _overrides(selection: dict | None) -> list[str]:
    return codex_config.builtin_tools_overrides(a2a_server.CodexConfig(builtin_tools=selection).builtin_tools)


def _catalog_overrides(selection: dict | None) -> list[str]:
    return asyncio.run(codex_config.catalog_overrides(a2a_server.CodexConfig(builtin_tools=selection).builtin_tools))


@pytest.fixture
def cli(monkeypatch, tmp_path) -> Path:
    """A `codex` on PATH whose `debug models` prints `_CATALOG` and records its arguments in the returned file, under a
    CODEX_HOME of the test's own."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    bin_dir, calls = tmp_path / "bin", tmp_path / "codex-calls"
    bin_dir.mkdir()
    (tmp_path / "catalog.json").write_text(json.dumps(_CATALOG))
    script = bin_dir / "codex"
    script.write_text(f'#!/bin/sh\necho "$*" >> {calls}\ncat {tmp_path / "catalog.json"}\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return calls


def _catalog_file() -> Path:
    return Path(os.environ["CODEX_HOME"]) / codex_config.CATALOG_WITHOUT_APPLY_PATCH


def _catalog_override() -> str:
    return f"model_catalog_json={json.dumps(str(_catalog_file()))}"


@pytest.mark.parametrize("selection, overrides", [
    (None, []),
    ({"disable": "*"}, _ALL_OFF),
    ({"only": []}, _ALL_OFF),
    ({"disable": []}, []),
    ({"disable": ["web_search", "view_image"]}, ["features.view_image=false", 'web_search="disabled"']),
    ({"disable": ["spawn_agent", "send_message", "wait_agent", "followup_task", "interrupt_agent", "list_agents"]},
     ["features.multi_agent_v2=false", "features.multi_agent=false"]),
    ({"only": ["exec_command", "write_stdin", "update_plan"]},
     [o for o in _ALL_OFF if o not in ("features.shell_tool=false", "tools.update_plan.enabled=false")]),
])
def test_a_selection_is_the_overrides_that_turn_tools_off(selection, overrides):
    assert _overrides(selection) == overrides


def test_turns_get_the_selections_overrides_after_their_role_and_config_toml_stays(codex, cli, monkeypatch):
    """No selection, with no role, is no override at all, as before, and no catalog copy; a selection adds its
    overrides after the role's, and config.toml, whose MCP servers give the agent agent-env's tools, is the same either
    way."""
    codex_home = Path(os.environ["CODEX_HOME"])
    monkeypatch.setenv("A2A_PORT", "8000")
    turns = []

    async def fake_run_codex(**kwargs):
        turns.append((tomllib.loads((codex_home / "config.toml").read_text()), list(kwargs["config_overrides"]),
                      _catalog_file().exists()))
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            await client.post("/ext/mcp-config", json={"url": "https://gw/mcp", "name": "env4821"})
            await codex.send(client, "one")
            assert (await client.post("/ext/agent-config", json={"builtin_tools": {"disable": "*"}})).status_code == 200
            await codex.send(client, "two")
            await client.post("/ext/agent-config", json={"role": "solver", "builtin_tools": {"disable": ["web_search"]}})
            await codex.send(client, "three")

    asyncio.run(exercise())

    (before, none, catalog_before), (after, all_off, _), (_, solver, _) = turns
    assert none == [] and not catalog_before
    assert all_off == [*_ALL_OFF, _catalog_override()]
    assert solver == [*codex_config.role_overrides({"env4821": {}, "peer_agents": {}}, "solver"),
                      'web_search="disabled"']
    assert before == after and before["features"] == {"multi_agent_v2": True}
    assert set(before["mcp_servers"]) == {"env4821", "peer_agents"}


def test_the_overrides_reach_the_cli_argv(monkeypatch, tmp_path):
    """run_codex appends each override as its own `-c`, after the turn's own settings."""
    seen = []

    class _Stop(Exception):
        pass

    async def _exec(*cmd, **kwargs):
        seen.append(list(cmd))
        raise _Stop

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    with pytest.raises(_Stop):
        asyncio.run(codex_bridge._run_codex_once(prompt="go", thread_id=None, config_overrides=_overrides(
            {"disable": "*"})))

    (argv,) = seen
    assert argv[argv.index("--dangerously-bypass-approvals-and-sandbox") + 1:] == [
        arg for override in _ALL_OFF for arg in ("-c", override)]


@pytest.mark.parametrize("selection, removes", [
    (None, False),
    ({"disable": []}, False),
    ({"disable": ["web_search"]}, False),
    ({"only": ["apply_patch"]}, False),
    ({"only": ["apply_patch", "update_plan"]}, False),
    ({"disable": "*"}, True),
    ({"disable": ["apply_patch"]}, True),
    ({"only": []}, True),
    ({"only": ["update_plan"]}, True),
])
def test_a_selection_without_apply_patch_points_the_cli_at_a_catalog_without_it(cli, selection, removes):
    """The copy is the installed CLI's bundled catalog with `apply_patch_tool_type` null for every model and nothing
    else changed; a selection that keeps the tool, or none, runs nothing and writes nothing."""
    overrides = _catalog_overrides(selection)

    if not removes:
        assert overrides == [] and not _catalog_file().exists() and not cli.exists()
        return
    assert overrides == [_catalog_override()]
    assert cli.read_text().split() == ["debug", "models", "--bundled"]
    assert json.loads(_catalog_file().read_text()) == {"models": [
        {**model, "apply_patch_tool_type": None} for model in _CATALOG["models"]]}


def test_a_catalog_config_toml_names_is_the_one_copied(cli):
    """A `model_catalog_json` already in config.toml, relative to CODEX_HOME as the CLI reads it, is the catalog the CLI
    would use, so its copy is the one without `apply_patch`; the bundled one isn't read. config.toml stays."""
    codex_home = Path(os.environ["CODEX_HOME"])
    codex_home.mkdir()
    mine = {"models": [{"slug": "house-model", "apply_patch_tool_type": "function", "priority": 1}]}
    (codex_home / "house.json").write_text(json.dumps(mine))
    (codex_home / "config.toml").write_text('model_catalog_json = "house.json"\n')

    assert _catalog_overrides({"disable": "*"}) == [_catalog_override()]
    assert json.loads(_catalog_file().read_text()) == {"models": [
        {"slug": "house-model", "apply_patch_tool_type": None, "priority": 1}]}
    assert not cli.exists()
    assert (codex_home / "config.toml").read_text() == 'model_catalog_json = "house.json"\n'


def test_a_catalog_the_cli_cant_give_fails_the_selection(cli, tmp_path):
    """Without a catalog, the turn would get the `apply_patch` its selection turned off: it raises instead."""
    (tmp_path / "bin" / "codex").write_text("#!/bin/sh\necho 'no catalog' >&2\nexit 3\n")

    with pytest.raises(RuntimeError, match="exited 3: no catalog"):
        _catalog_overrides({"disable": "*"})


def _hanging_cli(tmp_path) -> Path:
    """Turn the fixture's `codex` into one that records its pid, then never prints a catalog."""
    pid_file = tmp_path / "codex-pid"
    (tmp_path / "bin" / "codex").write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 60\n")
    return pid_file


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_catalog_read_past_its_time_is_killed_and_fails_the_selection(cli, tmp_path):
    """The read gets only the time it's given, and its child doesn't outlive it."""
    pid_file = _hanging_cli(tmp_path)
    tools = a2a_server.CodexConfig(builtin_tools={"disable": "*"}).builtin_tools

    with pytest.raises(RuntimeError, match="no catalog within 1s"):
        asyncio.run(codex_config.catalog_overrides(tools, timeout_s=1))
    assert not _alive(int(pid_file.read_text()))


def test_a_cancelled_catalog_read_kills_its_child(cli, tmp_path):
    pid_file = _hanging_cli(tmp_path)
    tools = a2a_server.CodexConfig(builtin_tools={"disable": "*"}).builtin_tools

    async def cancel_mid_read():
        task = asyncio.create_task(codex_config.catalog_overrides(tools))
        while not pid_file.exists():
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_mid_read())
    assert not _alive(int(pid_file.read_text()))


def test_the_catalog_read_counts_against_the_turns_time_limit(codex, cli, monkeypatch):
    """The CLI gets what is left of `timeout_seconds` after the catalog read; without a read, all of it, as before."""
    monkeypatch.setenv("A2A_PORT", "8000")
    timeouts = []
    real_catalog_overrides = codex_config.catalog_overrides

    async def slow_catalog_overrides(builtin_tools, timeout_s):
        overrides = await real_catalog_overrides(builtin_tools, timeout_s)
        if overrides:
            await asyncio.sleep(0.5)
        return overrides

    async def fake_run_codex(**kwargs):
        timeouts.append(kwargs["timeout_s"])
        return codex_bridge.CodexResult(response="ok", tool_call_count=0, session_id="t")

    monkeypatch.setattr(codex_config, "catalog_overrides", slow_catalog_overrides)
    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.app.router.lifespan_context(codex.app), codex.client() as client:
            await client.post("/ext/agent-config", json={"timeout_seconds": 30})
            await codex.send(client, "one")
            await client.post("/ext/agent-config", json={"builtin_tools": {"disable": "*"}})
            await codex.send(client, "two")

    asyncio.run(exercise())

    without_read, with_read = timeouts
    assert without_read == 30
    assert 29 < with_read <= 29.5


@pytest.mark.parametrize("selection, match", [
    ({"only": ["shell"]}, "shell"),
    ({"disable": ["tool_search"]}, "tool_search"),
    ({"disable": ["exec_command"]}, "on and off together"),
    ({"only": ["get_goal"]}, "on and off together"),
    ({"disable": "*", "only": ["update_plan"]}, "exactly one"),
    ({}, "exactly one"),
])
def test_an_invalid_selection_is_refused(selection, match):
    with pytest.raises(ValueError, match=match):
        a2a_server.CodexConfig(builtin_tools=selection)


def test_the_card_takes_builtin_tools_so_deploy_agent_accepts_a_selection(codex):
    extension = A2AAgent.find_extension(codex.card, A2AAgent.EXT_AGENT_CONFIG)

    assert "builtin_tools" in extension["params"]["methods"]["set"]["request"]["supported"]
    _check_builtin_tools(extension, "codex", {"disable": "*"})
    _check_builtin_tools(extension, "codex", {"only": ["update_plan"]})
    _check_builtin_tools(extension, "codex", {"disable": ["apply_patch"]})
    with pytest.raises(RuntimeError, match="can't take builtin_tools"):
        _check_builtin_tools(extension, "codex", {"disable": ["shell"]})


def test_the_selection_has_the_protocols_schema():
    def untitled(schema: Any) -> Any:
        if isinstance(schema, dict):
            return {k: untitled(v) for k, v in schema.items() if k not in ("title", "description")}
        return [untitled(v) for v in schema] if isinstance(schema, list) else schema

    assert untitled(codex_config.CodexTools.model_json_schema()) == untitled(
        BuiltinTools[codex_config.CodexTool].model_json_schema())


def test_every_tool_is_turned_off_by_one_setting():
    """`apply_patch` by the catalog copy, every other tool by one `-c` setting."""
    switched = [tool for tools, _ in codex_config._TOOL_SWITCHES for tool in tools]

    assert sorted([*switched, "apply_patch"]) == sorted(codex_config.CodexTool.__args__)


def test_the_tool_names_are_those_of_the_cli_version_package_json_pins():
    pinned = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["@openai/codex"]

    assert codex_config.TOOLS_CLI_VERSION == pinned, (
        "the CLI version changed: record the tools the new CLI offers in CodexTool, then TOOLS_CLI_VERSION")
