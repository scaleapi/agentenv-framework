"""The plugin report's format: `agent-env plugin list / show / check --json`, format version 1.

The goldens pin every key, status and code for scenarios that between them produce almost every
code. Reason text is replaced by a placeholder: it may change in any release, and a reword should
not touch the goldens. ``_SCHEMA`` is the format's keys and types, so a key added to the report
fails here until it is added there too; that edit is where to decide whether the format version
moves (PLUGINS.md "Plugin report format"). After an intended change, regenerate and review the diff:

    AGENT_ENV_REGENERATE_GOLDENS=1 python -m pytest tst/unit/cli/plugin_report_test.py
"""

import ast
import json
import os
import re
import sys
import typing
from importlib.metadata import EntryPoint
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import agent_env.plugins
from agent_env.cli.plugin import plugin
from agent_env.plugins import Status, inventory
from agent_env.plugins import _cli as cli_plugins_mod
from agent_env.plugins._cli import load_cli_plugins, load_cli_root_options
from agent_env.plugins._report import CODES, FORMAT_VERSION
from tst.unit.cli.plugin_command_test import _broken, _CliEP, _root_with_cli_plugins
from tst.unit.plugins_test import _EP, _HERE, _Dist, _fresh, _install, _use_config  # noqa: F401  (_fresh: autouse)

_THIS = "tst.unit.cli.plugin_report_test"
_GOLDEN = Path(__file__).parent / "golden" / "plugin_report"
_REGENERATE = os.environ.get("AGENT_ENV_REGENERATE_GOLDENS") == "1"
_PLUGINS_MD = Path(__file__).parents[3] / "PLUGINS.md"
_COMMANDS = cli_plugins_mod.CLI_PLUGINS_GROUP
_OPTIONS = cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP


def _fragile_callback(ctx, param, value):
    raise RuntimeError("fragile")


tools = click.Group(name="tools")
shadow = click.Command(name="plugin")
tenant = click.Option(["--tenant"], expose_value=False)
exposed = click.Option(["--exposed"])
fragile = click.Option(["--fragile"], expose_value=False, callback=_fragile_callback)
region = click.Option(["--region"], expose_value=False)
region_elsewhere = click.Option(["--region"], expose_value=False)


def _cli(name: str, attr: str, group: str, dist: str = "agentenv-tools") -> _CliEP:
    ep = _CliEP(name, attr, group, dist=dist)
    ep.value = f"{_THIS}:{attr}"
    return ep


# ---------------------------------------------------------------- the format's keys and types

_DIAGNOSTIC = {"code": str, "reason": str}
_CONTRIBUTION = {
    "group": str, "name": str, "value": str, "status": str, "code": (str, None), "reason": (str, None),
    "replaced_by": ({"file": (str, None), "table": str, "impl": str}, None),
    "conflicts_with": [{"package": str, "version": str, "value": str}],
}
_LIST = {
    "format_version": int,
    "agent_env": {"version": str, "environment": str, "location": str},
    "config": {"path": (str, None), "error": (_DIAGNOSTIC, None)},
    "loaded": bool,
    "group_errors": {str: _DIAGNOSTIC},
    "discovery_errors": {str: _DIAGNOSTIC},
    "plugins": [{"name": str, "version": str, "contributions": [_CONTRIBUTION]}],
}
_SCHEMA = {
    "list": _LIST,
    "show": {**_LIST, "config_effect": (str, None)},
    "check": {**_LIST, "ok": bool, "problems": [{"package": str, "version": str, **_CONTRIBUTION}]},
}


def _mismatches(value, schema, at: str = "$") -> list[str]:
    """Where ``value`` departs from ``schema``: a type, ``None``, a tuple of alternatives, ``[item]``,
    a record ``{key: schema}`` with exactly those keys, or a map ``{str: schema}``."""
    if isinstance(schema, tuple):
        found = [_mismatches(value, option, at) for option in schema]
        return [] if any(not f for f in found) else [f"{at}: {value!r} matches none of its alternatives"]
    if schema is None:
        return [] if value is None else [f"{at}: expected null"]
    if isinstance(schema, type):
        ok = isinstance(value, schema) and not (schema is int and isinstance(value, bool))
        return [] if ok else [f"{at}: expected {schema.__name__}, got {value!r}"]
    if isinstance(schema, list):
        if not isinstance(value, list):
            return [f"{at}: expected a list"]
        return [m for i, item in enumerate(value) for m in _mismatches(item, schema[0], f"{at}[{i}]")]
    if not isinstance(value, dict):
        return [f"{at}: expected an object"]
    if list(schema) == [str]:
        return [m for key, item in value.items() for m in _mismatches(item, schema[str], f"{at}.{key}")]
    if set(value) != set(schema):
        return [f"{at}: keys {sorted(set(value) ^ set(schema))} differ from the schema"]
    return [m for key, item in value.items() for m in _mismatches(item, schema[key], f"{at}.{key}")]


# ---------------------------------------------------------------- scenarios


_STEP = {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"}


def _bundles(monkeypatch, tmp_path, package: str, names: list[str], **step) -> None:
    """An importable package on sys.path holding a one-task bundle folder per name, its step ``_STEP`` with
    ``step``'s fields."""
    for name in names:
        tasks = tmp_path / "site" / package / name / "tasks"
        tasks.mkdir(parents=True)
        (tasks / "t.json").write_text(json.dumps([{**_STEP, **step}]))
    monkeypatch.syspath_prepend(str(tmp_path / "site"))


def _bundle(name: str, package: str, dist: str, requires: list[str] | None = None) -> EntryPoint:
    return EntryPoint(name, package, "agent_env.bundles")._for(_Dist(dist, "1.0", requires))


def _mixed(monkeypatch, tmp_path) -> click.Group:
    """A contribution for nearly every status and code."""
    _install(
        monkeypatch,
        envs=[_EP("browser", "_BrowserEnv", dist="agentenv-a"), _EP("website", "_BrowserEnv", dist="agentenv-a")],
        task_steps=[
            _EP("plugin_step", "_PluginStep", dist="agentenv-a"),
            _broken("broken_step", dist="agentenv-b"),
            _EP("not_a_step", "_BrowserEnv", dist="agentenv-b"),
            _EP("future_step", "_PluginStep", dist="agentenv-d", requires=["agentenv-framework>=999"]),
        ],
        sandbox_providers=[
            _EP("box", "_OtherRecordingProvider", dist="agentenv-a"),
            _EP("box", "_GuardedProvider", dist="agentenv-b"),
            _EP("plugin_box", "_RecordingProvider", dist="agentenv-b"),
        ],
        state_providers=[_EP("plugin_state", "_PluginState", dist="agentenv-a")],
        explorer_plugins=[_EP("broken_routes", "_BrokenCtorRoutes", dist="agentenv-b")],
        bundles=[
            _bundle("hello", "report_bundles_a", "agentenv-a"), _bundle("hello", "report_bundles_b", "agentenv-b"),
            _bundle("missing", "report_bundles_b", "agentenv-b"),
            _bundle("twice", "report_bundles_b", "agentenv-b"), _bundle("twice", "report_bundles_a", "agentenv-b"),
            _bundle("future", "report_bundles_a", "agentenv-d", requires=["agentenv-framework>=999"]),
            _bundle("tools", "report_bundles_only", "agentenv-bundles"),
            _bundle("empty", "report_bundles_only", "agentenv-bundles"),
            _bundle("unknown_step", "report_bundles_steps", "agentenv-bundles"),
            _bundle("bad_field", "report_bundles_steps", "agentenv-bundles"),
        ],
    )
    _bundles(monkeypatch, tmp_path, "report_bundles_a", ["hello", "twice", "future"])
    _bundles(monkeypatch, tmp_path, "report_bundles_b", ["hello", "twice"])
    _bundles(monkeypatch, tmp_path, "report_bundles_only", ["tools"])
    (tmp_path / "site" / "report_bundles_only" / "empty").mkdir()
    _bundles(monkeypatch, tmp_path, "report_bundles_steps", ["unknown_step"], type="no_such_step")
    _bundles(monkeypatch, tmp_path, "report_bundles_steps", ["bad_field"], sandbox_mode=["vm"])
    _use_config(monkeypatch, tmp_path, f'[envs]\nimpls = ["{_HERE}:_OtherBrowserEnv"]\n')
    monkeypatch.setattr(sys.modules["agent_env.cli.plugin"], "config_effect",
                        lambda name: "importing it leaves AGENT_ENV_CONFIG unset")
    return _root_with_cli_plugins(
        monkeypatch,
        commands=[_cli("tools", "tools", _COMMANDS), _cli("plugin", "shadow", _COMMANDS),
                  _cli("tools", "tools", _COMMANDS, dist="agentenv-c")],
        options=[_cli("tenant", "tenant", _OPTIONS), _cli("tenant_again", "tenant", _OPTIONS),
                 _cli("exposed", "exposed", _OPTIONS), _cli("fragile", "fragile", _OPTIONS),
                 _cli("region", "region", _OPTIONS), _cli("region", "region_elsewhere", _OPTIONS, dist="agentenv-c")],
    )


def _broken_install(monkeypatch, tmp_path) -> click.Group:
    """No config file where AGENT_ENV_CONFIG points, and CLI entry points that cannot be read."""
    def unreadable(*, group):
        raise TypeError("Pair.__new__() missing 1 required positional argument: 'value'")

    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(cli_plugins_mod, "entry_points", unreadable)
    root = click.Group()
    root.add_command(plugin)
    load_cli_plugins(root)
    load_cli_root_options(root)
    return root


_CASES = {
    "list": (_mixed, ["list", "--json"], 0),
    "list_no_load": (_mixed, ["list", "--json", "--no-load"], 0),
    "show": (_mixed, ["show", "agentenv-a", "--json"], 0),
    "show_bundles": (_mixed, ["show", "agentenv-bundles", "--json"], 0),
    "check": (_mixed, ["check", "--json"], 1),
    "check_broken": (_broken_install, ["check", "--json"], 1),
}


def _report(case: str, monkeypatch, tmp_path) -> dict:
    scenario, args, exit_code = _CASES[case]
    result = CliRunner().invoke(scenario(monkeypatch, tmp_path), ["plugin", *args])
    assert result.exit_code == exit_code, result.output
    return json.loads(result.stdout)


def _normalized(node, tmp_path: Path):
    """``node`` with machine-specific values and free text replaced by placeholders."""
    if isinstance(node, dict):
        return {
            key: "<text>" if key in ("reason", "config_effect") and value is not None
            else {k: f"<{k}>" for k in value} if key == "agent_env"
            else _normalized(value, tmp_path)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_normalized(item, tmp_path) for item in node]
    if isinstance(node, str):
        return node.replace(str(tmp_path.resolve()), "<tmp>").replace(str(tmp_path), "<tmp>")
    return node


def _contributions(report: dict) -> list[dict]:
    return [c for d in report["plugins"] for c in d["contributions"]] + report.get("problems", [])


# ---------------------------------------------------------------- the tests


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_report_matches_its_golden(case, monkeypatch, tmp_path):
    report = _normalized(_report(case, monkeypatch, tmp_path), tmp_path)
    golden = _GOLDEN / f"{case}.json"
    text = json.dumps(report, indent=2) + "\n"
    if _REGENERATE:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert golden.is_file(), f"no golden for {case}; regenerate: AGENT_ENV_REGENERATE_GOLDENS=1 python -m pytest {__file__}"
    assert text == golden.read_text(), f"the {case} report changed; if intended, regenerate and review the diff"


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_report_has_the_formats_keys_and_types(case, monkeypatch, tmp_path):
    report = _report(case, monkeypatch, tmp_path)

    assert list(report)[0] == "format_version" and report["format_version"] == FORMAT_VERSION
    assert _mismatches(report, _SCHEMA[case.split("_")[0]]) == []


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_report_keeps_its_invariants(case, monkeypatch, tmp_path):
    report = _report(case, monkeypatch, tmp_path)
    statuses = typing.get_args(Status)

    errors = [*report["group_errors"].values(), *report["discovery_errors"].values()]
    errors += [report["config"]["error"]] if report["config"]["error"] else []
    assert all("error" in CODES[e["code"]] for e in errors)
    for c in _contributions(report):
        assert c["status"] in statuses
        assert (c["code"] is None) == (c["reason"] is None), c
        assert c["status"] == "active" or c["code"] is not None, c
        assert c["code"] is None or c["status"] in CODES[c["code"]], c
        assert (c["status"] == "replaced") == (c["replaced_by"] is not None), c
        if c["status"] == "blocked":
            assert report["group_errors"][c["group"]]["code"] == c["code"], c
        if c["status"] == "conflict" or (c["code"] == "name-conflict" and c["status"] != "blocked"):
            assert c["conflicts_with"], c
        if c["code"] == "builtin-name":
            assert not c["conflicts_with"], c
    if case.startswith("check"):
        failing = {"conflict", "blocked", "failed", "skipped", "unloaded"}
        expected = [c for d in report["plugins"] for c in d["contributions"] if c["status"] in failing]
        assert [{k: v for k, v in p.items() if k not in ("package", "version")} for p in report["problems"]] == expected
        assert report["ok"] is False


def test_no_contribution_field_takes_a_name_check_problems_add():
    """`check` flattens a contribution next to its package and version."""
    assert not {"package", "version"} & set(_CONTRIBUTION)


def test_codes_are_kebab_case():
    assert all(re.fullmatch(r"[a-z]+(-[a-z]+)*", code) for code in CODES)


def _readme_section() -> str:
    return _PLUGINS_MD.read_text().split("### Plugin report format", 1)[1].split("\n### ", 1)[0]


def test_the_readme_lists_every_code():
    listed = re.findall(r"^\| `([a-z-]+)` \|", _readme_section(), flags=re.MULTILINE)

    assert sorted(listed) == sorted(CODES)


def test_the_readme_example_has_the_formats_keys_and_types():
    example = json.loads(_readme_section().split("```json", 1)[1].split("```", 1)[0])

    assert _mismatches(example, _SCHEMA["list"]) == []


@pytest.mark.parametrize(("body", "code"), [
    ("[envs\n", "config-unreadable"),
    ('[envs]\nimpls = "x"\n', "config-invalid"),
    ('[envs]\nimpls = ["agentenv_exits_on_import:X"]\n', "group-build-failed"),
])
def test_a_config_problem_is_a_group_error_with_its_code(body, code, monkeypatch, tmp_path):
    (tmp_path / "agentenv_exits_on_import.py").write_text("raise SystemExit(1)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, body)

    assert inventory().group_errors["agent_env.envs"].code == code


def test_every_code_is_produced_by_some_scenario(monkeypatch, tmp_path):
    produced = set()
    for case in _CASES:
        report = json.loads((_GOLDEN / f"{case}.json").read_text())
        produced |= {c["code"] for c in _contributions(report)}
        produced |= {e["code"] for e in (*report["group_errors"].values(), *report["discovery_errors"].values())}
        produced |= {report["config"]["error"]["code"]} if report["config"]["error"] else set()
    # From test_a_config_problem_is_a_group_error_with_its_code, and one the inventory cannot reach
    # while every registry records what it did (plugin_inventory_test covers it by forcing it).
    produced |= {"config-unreadable", "config-invalid", "group-build-failed", "status-unknown"}

    assert produced - {None} == set(CODES)


def _subscripted(node: ast.AST) -> str | None:
    """``failures`` for ``failures[name] = ...`` or ``self.failures[name] = ...``."""
    if isinstance(node, ast.Subscript):
        target = node.value
        return target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
    return None


def test_a_contribution_that_does_not_take_effect_is_always_made_with_a_code():
    """Where the code builds contributions and records failures, so a new path cannot leave
    ``code`` null where no scenario happens to reach it."""
    built, unpaired = [], []
    for path in sorted(Path(agent_env.plugins.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            name = getattr(getattr(node, "func", None), "id", None)
            builders = ("Contribution", "_record") if path.name == "_cli.py" else ("Contribution",)
            if isinstance(node, ast.Call) and name in builders:
                status = node.args[3]
                keywords = {k.arg for k in node.keywords}
                if isinstance(status, ast.Name) and status.id == "status":  # _record passing its own on
                    continue
                assert isinstance(status, ast.Constant), f"{path.name}:{node.lineno}: a status this test cannot read"
                built.append((f"{path.name}:{node.lineno}", status.value, "code" in keywords))
            for body in (getattr(node, field, None) for field in ("body", "orelse", "finalbody")):
                if not isinstance(body, list):
                    continue
                assigned = {_subscripted(t) for stmt in body if isinstance(stmt, ast.Assign) for t in stmt.targets}
                if "failures" in assigned and "codes" not in assigned:
                    unpaired.append(f"{path.name}:{body[0].lineno}")

    assert len(built) > 10, "the scan no longer finds where contributions are built"
    assert [site for site, status, coded in built if status != "active" and not coded] == []
    assert unpaired == [], "a registration failure recorded without its code"

