"""LocalRegistryImageStore's lazy registry bring-up, with docker and the /v2/ ping faked."""

import subprocess

import pytest

from agent_env.store.image_store import local_registry_image_store
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore


class _Docker:
    """Answers the docker commands bring-up issues and records each one."""

    def __init__(self, *, run_rc=0, existing_port=None, start_rc=0):
        self.run_rc, self.existing_port, self.start_rc = run_rc, existing_port, start_rc
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        command = argv[1]
        if command == "run":
            return subprocess.CompletedProcess(argv, self.run_rc, "", "Conflict" if self.run_rc else "")
        if command == "inspect":
            found = self.existing_port is not None
            return subprocess.CompletedProcess(argv, 0 if found else 1, str(self.existing_port or ""), "")
        if command == "start":
            return subprocess.CompletedProcess(argv, self.start_rc, "", "boom" if self.start_rc else "")
        raise AssertionError(f"unexpected docker call: {argv}")


@pytest.fixture
def docker(monkeypatch):
    def install(**kwargs):
        fake = _Docker(**kwargs)
        monkeypatch.setattr(local_registry_image_store.subprocess, "run", fake)
        return fake
    return install


@pytest.fixture
def listening(monkeypatch):
    """The registry answers only after bring-up has run a docker command."""
    def install(fake):
        monkeypatch.setattr(LocalRegistryImageStore, "_registry_listening", staticmethod(lambda port: bool(fake.calls)))
        monkeypatch.setattr(local_registry_image_store.time, "sleep", lambda seconds: None)
    return install


def test_the_default_port_runs_agentenv_registry(docker, listening):
    fake = docker()
    listening(fake)
    LocalRegistryImageStore("localhost:5000").ensure_repository("r")
    assert fake.calls == [["docker", "run", "-d", "--restart", "unless-stopped", "--name", "agentenv-registry",
                           "-p", "127.0.0.1:5000:5000", "registry:2"]]


def test_another_port_runs_the_same_container_on_that_port(docker, listening):
    fake = docker()
    listening(fake)
    LocalRegistryImageStore("localhost:5001").ensure_repository("r")
    assert fake.calls[0][fake.calls[0].index("--name") + 1] == "agentenv-registry"
    assert "127.0.0.1:5001:5000" in fake.calls[0]


def test_a_serving_registry_is_left_alone(docker, monkeypatch):
    fake = docker()
    monkeypatch.setattr(LocalRegistryImageStore, "_registry_listening", staticmethod(lambda port: True))
    LocalRegistryImageStore("localhost:5000").ensure_repository("r")
    assert fake.calls == []


@pytest.mark.parametrize("port", [5000, 5001])
def test_a_stopped_container_on_this_port_is_started(docker, listening, port):
    fake = docker(run_rc=125, existing_port=port)
    listening(fake)
    LocalRegistryImageStore(f"localhost:{port}").ensure_repository("r")
    assert [call[1] for call in fake.calls] == ["run", "inspect", "start"]
    assert fake.calls[-1] == ["docker", "start", "agentenv-registry"]


@pytest.mark.parametrize("existing, wanted", [(5000, 5001), (5001, 5000)])
def test_a_container_on_another_port_is_refused_not_removed(docker, monkeypatch, existing, wanted):
    fake = docker(run_rc=125, existing_port=existing)
    monkeypatch.setattr(LocalRegistryImageStore, "_registry_listening", staticmethod(lambda port: False))
    with pytest.raises(RuntimeError, match=f"agentenv-registry is using port {existing}, not {wanted}"):
        LocalRegistryImageStore(f"localhost:{wanted}").ensure_repository("r")
    assert [call[1] for call in fake.calls] == ["run", "inspect"]


def test_a_failed_run_with_no_container_surfaces_dockers_error(docker, monkeypatch):
    fake = docker(run_rc=125)
    monkeypatch.setattr(LocalRegistryImageStore, "_registry_listening", staticmethod(lambda port: False))
    with pytest.raises(RuntimeError, match="could not start the local registry: Conflict"):
        LocalRegistryImageStore("localhost:5000").ensure_repository("r")
    assert all(call[1] != "rm" for call in fake.calls)
