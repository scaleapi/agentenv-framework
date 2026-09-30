"""CLI import contracts: retired re-export paths and no import-time config reads.

The sandbox-url extraction behavior itself lives, with its tests, in a
platform plugin; nothing in this repo imports it.
"""

import importlib
import subprocess
import sys

from click.testing import CliRunner
from unittest.mock import MagicMock, patch

from agent_env.utils.docker_build import DockerBuildError


def test_old_cli_path_no_longer_re_exports_the_helper():
    # The compatibility re-export is retired: its consumer, a sandbox proxy
    # service, imports `extract_sandbox_id` from the plugin that owns it now, so
    # nothing reads it from the CLI module any more.
    # `import agent_env.cli.env.mcp_server as X` cannot be used here: the parent
    # package binds the name `mcp_server` to the click group, shadowing the
    # submodule for the `import ... as` binding form.
    mcp_server_cli = importlib.import_module("agent_env.cli.env.mcp_server")

    assert not hasattr(mcp_server_cli, "_extract_sandbox_id")



_IMPORT_PURITY_PROBE = """
import sys

import agent_env.config as cfg
import agent_env.config.runtime as runtime

calls = []


def _tripwire(*a, **kw):
    calls.append(True)
    raise AssertionError("get_config() was called during `import agent_env.cli`")


cfg.get_config = _tripwire
runtime.get_config = _tripwire

import agent_env.cli  # noqa: F401

print("OK" if not calls else "CALLED")
"""


def test_importing_the_cli_does_not_read_config():
    # Subprocess: agent_env.cli is already imported in-session; the AST test
    # below is the static backstop.
    res = subprocess.run(
        [sys.executable, "-c", _IMPORT_PURITY_PROBE],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert res.stdout.strip().endswith("OK"), f"stdout={res.stdout}\nstderr={res.stderr}"


def test_no_decorator_argument_calls_get_config():
    import ast
    from pathlib import Path

    import agent_env.cli as cli_pkg

    offenders = []
    for path in sorted(Path(cli_pkg.__file__).parent.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for deco in node.decorator_list:
                for sub in ast.walk(deco):
                    if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                            and sub.func.id == "get_config"):
                        offenders.append(f"{path.name}:{sub.lineno} ({node.name})")
    assert not offenders, (
        "get_config() in a decorator argument is evaluated at import time — "
        f"resolve it inside the command body instead: {offenders}"
    )


def _config(**attrs):
    cfg = MagicMock()
    for k, v in attrs.items():
        setattr(cfg, k, v)
    return cfg


def test_service_db_put_falls_back_to_the_config_default():
    from agent_env.cli.env.service_db import service_db

    cfg = _config(default_service_db_env_id="svc-db-from-config")
    with patch("agent_env.cli.env.service_db.get_config", return_value=cfg), \
         patch("agent_env.cli.env.service_db.DockerImageArtifact") as artifact, \
         patch("agent_env.cli.env.service_db.build_image") as run:
        artifact.put.side_effect = RuntimeError("stop once the id is recorded")
        res = CliRunner().invoke(service_db, ["put"])

    assert artifact.put.call_args.kwargs["id"] == "service-db-svc-db-from-config"
    assert res.exit_code != 0


def test_service_db_put_explicit_id_wins_and_skips_the_config_read():
    from agent_env.cli.env.service_db import service_db

    with patch("agent_env.cli.env.service_db.get_config") as get_config, \
         patch("agent_env.cli.env.service_db.build_image") as run:
        run.side_effect = DockerBuildError("stop here")
        CliRunner().invoke(service_db, ["put", "--id", "explicit-env"])

    get_config.assert_not_called()


def test_website_browser_put_falls_back_to_the_config_default():
    from agent_env.cli.env.website_browser import website_browser

    cfg = _config(default_website_browser_env_id="wb-from-config")
    with patch("agent_env.config.get_config", return_value=cfg), \
         patch("agent_env.cli.env.website_browser.DockerImageArtifact") as artifact, \
         patch("agent_env.cli.env.website_browser.build_image") as run:
        artifact.put.side_effect = RuntimeError("stop once the id is recorded")
        res = CliRunner().invoke(website_browser, ["put"])

    assert artifact.put.call_args.kwargs["id"] == "website-browser-wb-from-config"
    assert res.exit_code != 0


def test_help_text_names_the_config_key_for_each_lazy_default():
    from agent_env.cli.env.service_db import service_db
    from agent_env.cli.env.website_browser import website_browser

    runner = CliRunner()
    assert "default_service_db_env_id" in runner.invoke(service_db, ["put", "--help"]).output
    assert "default_website_browser_env_id" in runner.invoke(
        website_browser, ["put", "--help"]).output
