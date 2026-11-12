"""``agent-env run`` on the hello bundle agent-env ships, with no infra: the local stores, a local VM-mode
sandbox, and a verifier scoring the files the task loaded into it. Needs `bash` and the usual shell tools on PATH."""

import asyncio
import importlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import pytest
from click.testing import CliRunner

import agent_env
from agent_env.artifact import FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle.hub import HubSource
from agent_env.bundle.installed import find_bundle
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.config.paths import state_root
from agent_env.providers.sandbox_providers.local_docker.provider import (
    LocalSandboxProvider,
    local_sandbox_ids,
    remove_local_work_dir,
)
from agent_env.providers.sandbox_providers.sandbox_provider import reset_sandbox_provider
from agent_env.store import Filter, LocalSqliteDocumentStore
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from tst.util.capabilities import skip_without_gnu_stat

pytestmark = pytest.mark.integration

SHIPPED = Path(agent_env.__file__).parent / "examples" / "hello"


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    """A copy of the shipped hello, with an eval. It isn't called hello, since a folder named like an installed
    bundle is refused. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    root = tmp_path / "my-hello"
    shutil.copytree(SHIPPED, root)
    (root / "evals").mkdir()
    (root / "evals/smoke.toml").write_text('tasks = ["hello"]\n')
    try:
        yield root
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        reset_artifact_store()
        reset_config()


@pytest.fixture
def quiet_logs():
    """pytest's live logging swaps its own stdout back in to print a record, so CliRunner loses whatever is
    echoed after the first one."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _run(*args):
    result = CliRunner().invoke(cli, ["run", *map(str, args)])
    assert result.exit_code == 0, result.output
    return result.output


def test_the_shipped_hello_runs_by_name_and_a_rerun_reuses_every_write(bundle, quiet_logs):
    listing = _run().splitlines()
    first = _run("hello")
    second = _run("hello")

    (row,) = [i for i, line in enumerate(listing) if line.startswith("hello ")]
    assert f"agentenv-framework {version('agentenv-framework')}" in listing[row]
    assert listing[row + 1].strip() == str(find_bundle("hello").root) == str(SHIPPED)
    assert "artifacts/greeting: v1 (new)" in first
    assert "artifacts/greeting: v1, unchanged" in second
    assert "tasks/hello.json: v1, unchanged" in second
    for output in (first, second):
        assert "tasks/hello.json v1: passed (hello: 1)" in output
        (instance_id,) = re.findall(r"instance (@local/agentenv-framework/hello/hello-\S+)", output)
        with namespace_routing():
            instance = Task.get_instance(instance_id)
        assert instance.status == "completed"
        assert instance.context["metadata"]["verifications"]["hello"]["score"] == 1.0


def test_the_shipped_hello_stays_local_when_config_defaults_to_another_provider(bundle, monkeypatch, tmp_path,
                                                                               quiet_logs):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    (tmp_path / ".agentenv").mkdir()
    (tmp_path / ".agentenv/config.toml").write_text('[sandbox]\ndefault = "no-such-provider"\n')
    reset_config()
    reset_sandbox_provider()

    assert "tasks/hello.json v1: passed (hello: 1)" in _run("hello")


def test_a_hub_bundle_runs_by_its_hf_reference_at_a_pinned_commit_and_a_rerun_reuses_its_download(bundle, monkeypatch,
                                                                                                   tmp_path, quiet_logs):
    """The fixture is a public dataset holding the hello example under bundles/hello, another bundle that needs a
    plugin under bundles/v3, and its recorded runs as tables at its root."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    repo, sha = "earakely-scale/agentenv-hf-smoke", "5fbd709207a73e2aeb73a77e272a61003d1be768"

    several = CliRunner().invoke(cli, ["run", f"hf://{repo}@{sha}"])
    first = _run(f"hf://{repo}@{sha}/bundles/hello")
    second = _run(f"hf://{repo}@{sha}/bundles/hello")

    assert several.exit_code == 1
    assert (f"run one of hf://{repo}@{sha}/bundles/hello, hf://{repo}@{sha}/bundles/v3") in several.output
    for output in (first, second):
        assert output.startswith(f"{repo} at {sha[:12]}, bundle bundles/hello\n")
        assert f"From {repo} at {sha[:12]}, bundle bundles/hello\nTasks:\n  tasks/hello.json v1: passed" in output
    assert "tasks/hello.json: v1, unchanged" in second
    folder = tmp_path / "cache" / "agentenv" / "hf" / repo / sha
    assert sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*")
                  if path.is_file() and path.relative_to(folder).parts[0] != ".cache") == [
        "README.md", "bundles/hello/README.md", "bundles/hello/artifacts/greeting/check.sh",
        "bundles/hello/artifacts/greeting/hello.txt", "bundles/hello/tasks/hello.json"]
    (instance_id,) = re.findall(rf"instance (@hf/{repo}/hello/hello-\S+)", second)
    with namespace_routing():
        instance = Task.get_instance(instance_id)
    assert instance.context["metadata"]["bundle_source"] == HubSource(repo, sha, sha, "bundles/hello").to_metadata()


def test_a_bundle_runs_from_its_folder_and_a_rerun_writes_only_what_changed(bundle, quiet_logs):
    first = _run(bundle)
    assert "tasks/hello.json v1: passed (hello: 1)" in first
    assert "evals/smoke.toml v1: 1 of 1 passed" in first

    second = _run(bundle)
    assert "artifacts/greeting: v1, unchanged" in second
    assert "tasks/hello.json: v1, unchanged" in second

    (bundle / "artifacts/greeting/hello.txt").write_text("goodbye\n")
    third = _run(bundle)
    assert "artifacts/greeting: v2 (files changed: hello.txt)" in third
    assert "tasks/hello.json v1: scored below 1 (hello: 0)" in third
    (instance_id,) = re.findall(r"instance (\S+)", third)
    with namespace_routing():
        results = Task.get_instance(instance_id).context["metadata"]["verifications"]["hello"]["results"]
    assert [(result["criterion"], result["result"]) for result in results] == [("greets", False), ("check passes", False)]


@skip_without_gnu_stat()
def test_a_bundle_task_collects_from_its_sandbox_into_a_universe_named_after_the_run(bundle, quiet_logs):
    steps = json.loads((bundle / "tasks/hello.json").read_text())
    steps.append({"id": "collect", "type": "collect_artifacts", "sandbox_name": "box", "base_path": "/app/greeting",
                  "artifact_paths": ["hello.txt"]})
    (bundle / "tasks/hello.json").write_text(json.dumps(steps))

    output = _run(bundle)

    assert "tasks/hello.json v1: passed (hello: 1)" in output
    (instance_id,) = re.findall(r"instance (@local/~/my-hello/hello-[a-z0-9]{8})\b", output)
    with namespace_routing():
        universe = FileArtifactUniverse.get(instance_id)
        assert {name: fa.load() for name, fa in universe.get_file_artifacts().items()} == {"hello.txt": b"hello\n"}
    documents = LocalSqliteDocumentStore(str(state_root() / "document_store" / "documents.db"))
    assert not documents.path.exists() or documents.count("artifacts", Filter()) == 0


def test_an_installed_bundle_runs_by_name_with_ids_rooted_at_its_package(bundle, monkeypatch, tmp_path, quiet_logs):
    """A real .dist-info registers the folder through agent_env.bundles; nothing is imported to find it. agent-env
    ships a hello too, so this one runs by its qualified name."""
    site = tmp_path / "site"
    shutil.copytree(bundle, site / "demo_installed_bundles" / "hello")
    (site / "demo_installed_bundles" / "__init__.py").write_text("raise RuntimeError('imported')\n")
    dist_info = site / "demo_installed_bundles-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: demo-installed-bundles\nVersion: 1.0\n")
    (dist_info / "entry_points.txt").write_text("[agent_env.bundles]\nhello = demo_installed_bundles\n")
    (dist_info / "RECORD").write_text("")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()

    listing = _run().splitlines()
    result = _run("demo-installed-bundles/hello")

    assert any(line.startswith("demo-installed-bundles/hello ") and "demo-installed-bundles 1.0" in line
               for line in listing)
    assert any(line.startswith("hello ") and "agentenv-framework" in line for line in listing)
    assert "tasks/hello.json v1: passed (hello: 1)" in result
    assert "instance @local/demo-installed-bundles/hello/hello-" in result
    assert "demo_installed_bundles" not in sys.modules


def _signalled(args, when, signum=signal.SIGINT, group=True, ready=lambda: True):
    """Run ``agent-env run ARGS`` in its own process group, as a terminal would, and send ``signum`` to the group
    (or to the process alone) once a line containing ``when`` is printed and ``ready()`` holds. Returns the exit code,
    the output, and what the sandbox root held when the signal went."""
    process = subprocess.Popen([sys.executable, "-m", "agent_env.cli", "run", *map(str, args)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    watchdog = threading.Timer(120, process.kill)
    watchdog.start()
    lines, held = [], None
    try:
        for line in process.stdout:
            lines.append(line)
            if held is None and when in line:
                deadline = time.monotonic() + 30
                while not ready() and time.monotonic() < deadline:
                    time.sleep(0.05)
                root = os.environ["AGENT_ENV_LOCAL_SANDBOX_DIR"]
                held = sorted(os.listdir(root)) if os.path.isdir(root) else []
                (os.killpg if group else os.kill)(process.pid, signum)
        return process.wait(), "".join(lines), held
    finally:
        watchdog.cancel()


@pytest.mark.parametrize("signum, group, code", [(signal.SIGINT, True, 130), (signal.SIGTERM, False, 143)],
                         ids=["ctrl-c-to-the-group", "sigterm-to-the-process"])
def test_a_signal_mid_run_marks_the_run_cancelled_tears_it_down_and_stops_its_commands(bundle, tmp_path, signum,
                                                                                       group, code, sigint_handled):
    napping = tmp_path / "napping"
    (napping / "tasks").mkdir(parents=True)
    (napping / "tasks/nap.json").write_text(json.dumps([
        {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local"},
        {"id": "nap", "type": "verify_sandbox", "sandbox_name": "box", "base_dir": "/app", "verifier_id": "nap",
         "criteria": [{"type": "bash_cmd_succeeds", "criterion": "naps",
                       "bash_cmd": 'sleep 30 & echo $! > "$HOME/nap.pid"; wait'}]},
    ]))

    nap = tmp_path / "nap.pid"
    exit_code, output, held = _signalled([napping], "step 2/2 nap (verify_sandbox)", signum, group, ready=nap.exists)

    assert exit_code == code, output
    assert len(held) == 1
    assert re.search(r"tasks/nap.json v1: cancelled, [\d.]+s, instance (\S+)", output), output
    assert output.endswith("\nTore down 1 sandbox.\n")
    (instance_id,) = re.findall(r"instance (\S+)", output)
    with namespace_routing():
        assert Task.get_instance(instance_id).status == "cancelled"
    assert not _alive(int(nap.read_text()))


def test_a_run_killed_outright_leaves_its_sandboxes_under_its_root_for_a_supervisor_to_terminate(bundle, tmp_path):
    napping = tmp_path / "napping"
    (napping / "tasks").mkdir(parents=True)
    (napping / "tasks/nap.json").write_text(json.dumps([
        {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local"},
        {"id": "web", "type": "deploy_sandbox", "sandbox_name": "web", "sandbox_mode": "container",
         "sandbox_type": "local", "image": "public.ecr.aws/nginx/nginx:stable-alpine", "port": 80},
        {"id": "nap", "type": "verify_sandbox", "sandbox_name": "box", "base_dir": "/app", "verifier_id": "nap",
         "depends_on": ["box", "web"], "criteria": [{"type": "bash_cmd_succeeds", "criterion": "naps",
                                                    "bash_cmd": 'sleep 300 & echo $! > "$HOME/nap.pid"; wait'}]},
    ]))

    nap = tmp_path / "nap.pid"
    exit_code, output, held = _signalled([napping], "step 3/3 nap (verify_sandbox)", signal.SIGKILL, group=False,
                                         ready=nap.exists)
    found = local_sandbox_ids()
    for sandbox_id in found:
        asyncio.run(LocalSandboxProvider().terminate_sandbox(sandbox_id))
        remove_local_work_dir(sandbox_id)

    assert exit_code == -signal.SIGKILL, output
    assert len(held) == len(found) == 2
    assert local_sandbox_ids() == []
    assert not _alive(int(nap.read_text()))
    for sandbox_id in found:
        for selector in (f"label=agentenv.sandbox={sandbox_id}", f"name=^agent-{sandbox_id}$"):
            listed = subprocess.run(["docker", "ps", "-aq", "--filter", selector], capture_output=True, text=True,
                                    check=True)
            assert listed.stdout == "", selector


def _alive(pid):
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        time.sleep(0.02)
    return True


def test_keep_holds_the_sandbox_up_until_ctrl_c_then_tears_it_down(bundle, sigint_handled):
    code, output, held = _signalled([bundle, "--keep"], "Holding 1 sandbox up")

    assert code == 0, output
    assert len(held) == 1
    assert f"folder  {os.environ['AGENT_ENV_LOCAL_SANDBOX_DIR']}/{held[0]}" in output
    assert output.endswith("\nTore down 1 sandbox.\n")
