"""A local sandbox holds the containers it starts to the cpu and memory it was created with."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import agent_env.a2a_agent.a2a_agent as a2a_module
import agent_env.config as cfg
import agent_env.providers.sandbox_providers.local_sandbox as ls
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import ContainerLimits

_real_engine_cpus = ls._engine_cpus  # before the unit conftest answers it


@pytest.fixture(autouse=True)
def sandbox_root(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    monkeypatch.setattr(ls, "_host_ips", lambda: ("127.0.0.1",))
    return tmp_path


@pytest.fixture
def no_local_ca(monkeypatch):
    """Containers start with ``docker run``, not created, given the local CA's files, then started."""
    monkeypatch.setattr(ls, "local_grant_trust", lambda: None)
    monkeypatch.setattr(a2a_module, "local_grant_trust", lambda: None)


@pytest.fixture
def engine_answers(monkeypatch):
    """``docker info`` answers each item in turn: a string is its output, None a failure."""
    asked = []

    def answer(*outputs):
        replies = iter(outputs)

        def run(args, **kwargs):
            asked.append(args)
            out = next(replies)
            return SimpleNamespace(returncode=1 if out is None else 0, stdout=out or "", stderr="daemon down")

        monkeypatch.setattr(ls.subprocess, "run", run)
        return asked

    return answer


class _RecordingLocalSandbox(LocalSandbox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scripts: list[str] = []

    async def exec_script(self, script, *, max_retries=0):
        self.scripts.append(script)
        return ""


def _agent() -> A2AAgent:
    return A2AAgent(id="a", version=None, docker_image_artifact=SimpleNamespace(image_name="img:1"))


def test_limits_render_as_docker_flags_and_compose_keys():
    limits = ContainerLimits(cpus=1.0, memory_mib=8192)
    assert limits.docker_args == ("--cpus", "1", "--memory", "8192m", "--memory-swap", "8192m")
    assert limits.compose_keys == ("cpus: 1", "mem_limit: 8192m", "memswap_limit: 8192m")
    assert ContainerLimits(cpus=0.5, memory_mib=512).docker_args[:2] == ("--cpus", "0.5")


@pytest.mark.parametrize("cpus, memory_mib", [(0, 1024), (1, 0), (-1, 1024)])
def test_limits_must_be_positive(cpus, memory_mib):
    with pytest.raises(ValueError, match="must be positive"):
        ContainerLimits(cpus=cpus, memory_mib=memory_mib)


@pytest.mark.asyncio
async def test_a_local_vm_keeps_its_limits_for_a_handle_rebuilt_later():
    sandbox = await LocalSandboxProvider().create_vm(cpu=2, memory=4096)

    assert sandbox.container_limits == ContainerLimits(cpus=2, memory_mib=4096)
    reopened = await LocalSandboxProvider().get_sandbox(sandbox.sandbox_id)
    assert reopened.container_limits == ContainerLimits(cpus=2, memory_mib=4096)


@pytest.mark.asyncio
async def test_cpu_beyond_the_engines_is_held_to_it(monkeypatch, caplog):
    """Docker refuses a container more CPUs than its engine has, so a bigger ask is held to the engine's."""
    monkeypatch.setattr(ls, "_engine_cpus", lambda: 4)

    with caplog.at_level(logging.WARNING, logger=ls.logger.name):
        sandbox = await LocalSandboxProvider().create_vm(cpu=8, memory=8192)

    assert sandbox.container_limits == ContainerLimits(cpus=4, memory_mib=8192)
    assert "Asked for 8 CPUs, but the Docker engine has 4" in caplog.text


@pytest.mark.asyncio
async def test_cpu_within_the_engines_is_kept(monkeypatch):
    monkeypatch.setattr(ls, "_engine_cpus", lambda: 4)
    sandbox = await LocalSandboxProvider().create_vm(cpu=1.5, memory=2048)
    assert sandbox.container_limits == ContainerLimits(cpus=1.5, memory_mib=2048)


def test_the_engine_is_asked_each_time_so_a_resized_one_is_seen(engine_answers):
    asked = engine_answers("6\n", None, "2\n")

    assert [_real_engine_cpus() for _ in range(3)] == [6, None, 2]
    assert asked == [["docker", "info", "--format", "{{.NCPU}}"]] * 3


@pytest.mark.asyncio
async def test_one_cpu_or_less_is_not_checked_against_the_engine(monkeypatch):
    def asked():
        raise AssertionError("every engine has at least one CPU")

    monkeypatch.setattr(ls, "_engine_cpus", asked)
    sandbox = await LocalSandboxProvider().create_vm(cpu=1, memory=1024)
    assert sandbox.container_limits == ContainerLimits(cpus=1, memory_mib=1024)


def test_no_docker_means_no_cpu_count(monkeypatch, engine_answers):
    def missing(args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(ls.subprocess, "run", missing)
    assert _real_engine_cpus() is None


def test_a_sandbox_whose_limits_cant_be_recorded_is_not_made(sandbox_root):
    """A handle a later step rebuilds would start its containers unheld."""
    with pytest.raises(FileNotFoundError):
        LocalSandbox(sandbox_id="local-gone", work_dir=sandbox_root / "missing", container_limits=ContainerLimits(1, 512))


def test_a_new_sandbox_whose_limits_cant_be_recorded_leaves_no_folder(monkeypatch, sandbox_root):
    def full(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ls.Path, "write_text", full)
    with pytest.raises(OSError, match="No space left"):
        LocalSandbox(container_limits=ContainerLimits(1, 512))
    assert list(sandbox_root.iterdir()) == []


def test_a_sandbox_from_before_limits_were_recorded_holds_nothing(sandbox_root):
    work_dir = sandbox_root / "agent-env-local-old-x"
    work_dir.mkdir()
    assert LocalSandbox(sandbox_id="local-old", work_dir=work_dir).container_limits is None


def test_an_unreadable_limits_file_is_ignored(sandbox_root, caplog):
    work_dir = sandbox_root / "agent-env-local-bad-x"
    work_dir.mkdir()
    (work_dir / ls._LIMITS_FILE).write_text("{not json")

    with caplog.at_level(logging.WARNING, logger=ls.logger.name):
        assert LocalSandbox(sandbox_id="local-bad", work_dir=work_dir).container_limits is None
    assert "Ignoring unreadable container limits" in caplog.text


@pytest.mark.asyncio
async def test_a_container_mode_sandbox_starts_its_container_held_to_the_limits(monkeypatch, no_local_ca):
    """An agent's own sandbox is one container, given the cpu and memory the sandbox was asked for."""
    monkeypatch.setattr(cfg, "get_config", lambda: SimpleNamespace(get_image_store=lambda: SimpleNamespace(auth=lambda _: None)))
    monkeypatch.setattr(ls, "_free_host_port", lambda: 41337)
    made: list[_RecordingLocalSandbox] = []

    class _Provider(LocalSandboxProvider):
        async def create_vm(self, **kwargs):
            sandbox = await LocalSandboxProvider.create_vm(self, **kwargs)
            reopened = _RecordingLocalSandbox(sandbox_id=sandbox.sandbox_id, work_dir=sandbox.work_dir)
            made.append(reopened)
            return reopened

    await _Provider().create_sandbox(image_name="img:v1", port=8000, env={}, cpu=1.5, memory=2048)

    (run,) = [s for s in made[0].scripts if s.startswith("docker run -d")]
    assert "--cpus 1.5 --memory 2048m --memory-swap 2048m " in run


@pytest.mark.asyncio
async def test_an_agent_placed_on_a_local_vm_is_held_to_its_limits(no_local_ca):
    """An agent placed on a sandbox shares the sandbox's budget, as it would a VM's."""
    agent = _agent()
    agent._sandbox = _RecordingLocalSandbox(container_limits=ContainerLimits(cpus=2, memory_mib=4096))

    await agent._run_container("img:1", 8000, {})

    assert "--cpus 2 --memory 4096m --memory-swap 4096m " in agent._sandbox.scripts[0]


@pytest.mark.asyncio
async def test_the_docker_sidecar_is_held_to_the_same_limits(no_local_ca):
    agent = _agent()
    agent._sandbox = _RecordingLocalSandbox(container_limits=ContainerLimits(cpus=2, memory_mib=4096))

    await agent._run_container("img:1", 8000, {}, enable_docker=True)

    script = agent._sandbox.scripts[0]
    assert "DIND_LIMITS='--cpus 2 --memory 4096m --memory-swap 4096m'\n" in script
    assert '--name "$DIND_CONTAINER" \\\n    $DIND_LIMITS \\\n' in script


def test_a_sandbox_without_limits_gives_the_docker_sidecar_none():
    assert "DIND_LIMITS=\n" in _agent()._dind_setup_script()


@pytest.mark.asyncio
async def test_a_sandbox_without_limits_starts_containers_unlimited(no_local_ca):
    """A VM's own size bounds its containers, so a sandbox that records no limits passes no flags."""
    agent = _agent()
    agent._sandbox = _RecordingLocalSandbox()

    await agent._run_container("img:1", 8000, {})

    assert "--cpus" not in agent._sandbox.scripts[0] and "--memory" not in agent._sandbox.scripts[0]
