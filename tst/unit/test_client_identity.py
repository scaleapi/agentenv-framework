import asyncio
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IDENTITY_MODULE_PATH = REPO_ROOT / "src/agent_env/cli/identity/client_identity.py"


def _load_identity_module():
    spec = importlib.util.spec_from_file_location(
        "_agent_env_cli_identity_under_test",
        IDENTITY_MODULE_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


client_identity_module = _load_identity_module()
get_agent_env_client_id = client_identity_module.get_agent_env_client_id


def _completed(args, returncode=0, stdout=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")


def _load_cli_run_module(monkeypatch, relpath: str, module_name: str):
    cli_package = types.ModuleType("agent_env.cli")
    cli_package.__path__ = []
    banner_module = types.ModuleType("agent_env.cli.banner")
    banner_module.print_banner = lambda: None
    identity_module = types.ModuleType("agent_env.cli.identity")
    identity_module.get_agent_env_client_id = get_agent_env_client_id
    monkeypatch.setitem(sys.modules, "agent_env.cli", cli_package)
    monkeypatch.setitem(sys.modules, "agent_env.cli.banner", banner_module)
    monkeypatch.setitem(sys.modules, "agent_env.cli.identity", identity_module)

    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relpath)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolves_git_user_name_as_github_username(monkeypatch):
    def fake_run(args, **kwargs):
        assert args == ["git", "config", "--get", "user.name"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 2
        return _completed(args, stdout="Alexander Schwartzman\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_agent_env_client_id() == "agent-env-cli/alexander-schwartzman"


def test_resolves_git_user_name_with_punctuation(monkeypatch):
    def fake_run(args, **kwargs):
        assert args == ["git", "config", "--get", "user.name"]
        return _completed(args, stdout="Alex+Dev@Example.com\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_agent_env_client_id() == "agent-env-cli/alex-dev-example-com"


def test_client_id_empty_without_git_username(monkeypatch):
    def fake_run(args, **kwargs):
        return _completed(args, returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_agent_env_client_id() is None


def test_task_cli_stamps_client_metadata_without_overwriting(monkeypatch):
    task_run_module = _load_cli_run_module(
        monkeypatch,
        "src/agent_env/cli/task/run.py",
        "_agent_env_task_run_under_test",
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: _completed(args, stdout="Alex Dev\n"),
    )

    metadata = {"agent_env_hub": {"caller": "existing", "request_id": "old-req"}}
    client_id = get_agent_env_client_id()

    task_run_module._stamp_agent_env_client_metadata(metadata, client_id)

    assert metadata == {"agent_env_hub": {"caller": "existing", "request_id": "old-req"}}

    metadata = {"agent_env_hub": {"request_id": "old-req"}}
    task_run_module._stamp_agent_env_client_metadata(metadata, client_id)

    assert metadata == {
        "agent_env_hub": {
            "caller": "agent-env-cli/alex-dev",
            "request_id": "old-req",
        }
    }

    metadata = {}
    task_run_module._stamp_agent_env_client_metadata(metadata, client_id)

    assert metadata["agent_env_hub"]["caller"] == "agent-env-cli/alex-dev"


def test_task_cli_stamp_client_metadata_noops_without_client_id(monkeypatch):
    task_run_module = _load_cli_run_module(
        monkeypatch,
        "src/agent_env/cli/task/run.py",
        "_agent_env_task_run_missing_metadata_under_test",
    )
    metadata = {}

    task_run_module._stamp_agent_env_client_metadata(metadata, None)

    assert metadata == {}


def test_eval_run_single_passes_client_metadata_to_task(monkeypatch):
    task_step_package = types.ModuleType("agent_env.task_step")
    task_step_package.__path__ = []
    context_module = types.ModuleType("agent_env.task_step.context")

    class FakeTaskStepContext:
        def __init__(self, metadata=None):
            self.metadata = metadata or {}

    context_module.TaskStepContext = FakeTaskStepContext
    monkeypatch.setitem(sys.modules, "agent_env.task_step", task_step_package)
    monkeypatch.setitem(sys.modules, "agent_env.task_step.context", context_module)
    eval_run_module = _load_cli_run_module(
        monkeypatch,
        "src/agent_env/cli/eval/run.py",
        "_agent_env_eval_run_under_test",
    )

    class FakeTask:
        id = "fake-task"

        def __init__(self):
            self.context = None

        async def run(self, **kwargs):
            self.context = kwargs["context"]
            return self.context

    task = FakeTask()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: _completed(args, stdout="Alex Dev\n"),
    )

    task_id, tag, context = asyncio.run(
        eval_run_module._run_single(
            task,
            "[fake-task]",
            None,
            base_metadata={"agent_env_hub": {"caller": get_agent_env_client_id()}},
        )
    )

    assert task_id == "fake-task"
    assert tag == "[fake-task]"
    assert context is task.context
    assert context.metadata == {
        "agent_env_hub": {"caller": "agent-env-cli/alex-dev"}
    }
