"""Offline unit tests for the HF Jobs sandbox provider, over a fake Jobs API and sandbox server."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from types import SimpleNamespace

import httpx
import pytest

from agent_env.config.errors import ConfigError
from agent_env.env.gateway.constants import PRIVATE_HOST_ENV
from agent_env.providers.sandbox_providers import hf_jobs
from agent_env.providers.sandbox_providers.hf_jobs import HfJobsSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError

JOB = "6ace9550c66288145223121d"
PREFIX = "nwa-6ace955091389f104546662a-"
SERVER = ["python", "-m", "server"]


class _Api:
    endpoint = "https://huggingface.co"

    def __init__(self, stages=("RUNNING",), labels=None, ports=(18765,)):
        self.stages = list(stages)
        self.labels = labels or {}
        self.ports = ports
        self.cancelled: list[str] = []

    def whoami(self):
        return {"name": "someone"}

    def inspect_job(self, job_id, namespace=None):
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        urls = [f"https://{job_id}--49983.hf.jobs", *(f"https://{job_id}--{p}.hf.jobs/" for p in self.ports)]
        return SimpleNamespace(status=SimpleNamespace(stage=stage, message=None, expose_urls=urls), labels=self.labels)

    def cancel_job(self, job_id, namespace=None):
        self.cancelled.append(job_id)


class _Box:
    def __init__(self):
        self.runs: list[tuple] = []
        self.files = SimpleNamespace(write=lambda path, data: self.runs.append(("write", path, data)))

    def run(self, cmd, check=True, background=False, env=None, timeout=None):
        self.runs.append((cmd, background, env))
        if "HF_NETWORK_GROUP_PREFIX" in " ".join(cmd):
            return SimpleNamespace(exit_code=0, stdout=PREFIX, stderr="")
        return SimpleNamespace(exit_code=3, stdout="out", stderr="err")

    def close(self):
        pass


@pytest.fixture
def jobs(monkeypatch):
    """The provider over a fake Jobs API and sandbox server; ``posted`` collects each job spec it sends."""
    api, box, posted = _Api(), _Box(), []

    def post(url, json, headers, timeout):
        posted.append(json)
        assert url == "https://huggingface.co/api/jobs/someone" and headers == {"Authorization": "Bearer hf_test"}
        return httpx.Response(200, json={"id": JOB})

    monkeypatch.setattr(hf_jobs.httpx, "post", post)
    monkeypatch.setattr(hf_jobs.HubSandbox, "connect", staticmethod(lambda job_id, namespace, token: box))
    monkeypatch.setattr(hf_jobs, "_image_command", lambda image_name, args: SERVER)
    monkeypatch.setattr(hf_jobs, "_POLL_SECONDS", 0)
    provider = HfJobsSandboxProvider(token="hf_test")
    provider._api = api
    return SimpleNamespace(provider=provider, api=api, box=box, posted=posted)


def _started_command(box: _Box) -> tuple:
    return next(run for run in box.runs if run[1])


def test_a_private_container_is_a_job_in_the_providers_network_group_its_port_public(jobs):
    sandbox = asyncio.run(jobs.provider.create_container(
        image_name="ghcr.io/org/server:v1", port=18765, env={"DATABASE_URL": "postgres://db"}, timeout=900,
        attribution={"run_id": "run/1"}, private_network=True))

    [spec] = jobs.posted
    alias = spec["network"]["aliases"][0]
    assert re.fullmatch(r"c[0-9a-f]{12}", alias)
    assert spec["network"]["group"] == jobs.provider._network_group
    assert spec["dockerImage"] == "ghcr.io/org/server:v1" and spec["timeoutSeconds"] == 900
    assert spec["flavor"] == "cpu-basic"
    assert spec["expose"] == {"ports": [49983, 18765], "portsPublic": [18765]}
    assert spec["secrets"]["DATABASE_URL"] == "postgres://db" and "DATABASE_URL" not in spec["environment"]
    assert {key: spec["labels"][key] for key in ("agentenv", "agentenv-alias", "hf-sandbox", "run_id")} == {
        "agentenv": "1", "agentenv-alias": alias, "hf-sandbox": "1", "run_id": "run-1"}
    assert (sandbox.sandbox_id, sandbox.type, sandbox.mode) == (JOB, "hf_jobs", "container")
    assert sandbox.private_host == f"{PREFIX}{alias}"
    assert sandbox.tunnel_urls == {18765: f"https://{JOB}--18765.hf.jobs"}
    assert _started_command(jobs.box) == (["/bin/sh", "-c", 'exec "$@" >>/tmp/agentenv-main.log 2>&1', "sh", *SERVER],
                                          True, {PRIVATE_HOST_ENV: sandbox.private_host})


def test_a_container_off_the_internet_exposes_only_the_sandbox_server_and_runs_its_image_with_the_command_given(
        jobs, monkeypatch):
    monkeypatch.setattr(hf_jobs, "_image_command", lambda image_name, args: ["docker-entrypoint.sh", *args])

    sandbox = asyncio.run(jobs.provider.create_container(
        image_name="postgres:16-alpine", port=5432, env={}, command=["postgres", "-c", "fsync=off"],
        expose_externally=False, private_network=True))

    assert jobs.posted[0]["expose"] == {"ports": [49983]}
    assert _started_command(jobs.box)[0][4:] == ["docker-entrypoint.sh", "postgres", "-c", "fsync=off"]
    assert sandbox.private_host.startswith(PREFIX)


def test_an_agents_sandbox_is_exposed_on_no_network_group(jobs):
    jobs.api.ports = (8000,)
    sandbox = asyncio.run(jobs.provider.create_sandbox(image_name="ghcr.io/org/agent:v1", port=8000,
                                                       env={"LITELLM_API_KEY": "key"}))

    assert "network" not in jobs.posted[0] and "agentenv-alias" not in jobs.posted[0]["labels"]
    assert sandbox.private_host is None and _started_command(jobs.box)[2] is None


def test_a_job_that_ends_before_running_is_reported_and_cancelled(jobs, monkeypatch):
    jobs.api.stages = ["ERROR"]
    monkeypatch.setattr(hf_jobs.hub_sandbox, "_tail_job_logs", lambda api, job_id, namespace: " Last logs: pull failed")

    with pytest.raises(RuntimeError, match="ended ERROR before running.*pull failed"):
        asyncio.run(jobs.provider.create_container(image_name="ghcr.io/org/server:v1", port=18765, env={}))

    assert jobs.api.cancelled == [JOB]


def test_a_cancelled_create_cancels_its_job(jobs):
    jobs.api.stages = ["STARTING"]

    async def cancel_while_starting():
        create = asyncio.ensure_future(jobs.provider.create_container(image_name="ghcr.io/org/server:v1", port=18765,
                                                                      env={}))
        while not jobs.posted:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        create.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create

    asyncio.run(cancel_while_starting())

    assert jobs.api.cancelled == [JOB]


def test_a_run_torn_down_while_hf_creates_a_job_cancels_the_job(jobs, monkeypatch):
    posting, created = threading.Event(), threading.Event()

    def post(url, json, headers, timeout):
        posting.set()
        created.wait(5)
        return httpx.Response(200, json={"id": JOB})

    monkeypatch.setattr(hf_jobs.httpx, "post", post)

    async def end_while_posting():
        asyncio.ensure_future(jobs.provider.create_container(image_name="ghcr.io/org/server:v1", port=18765, env={}))
        await asyncio.to_thread(posting.wait, 5)
        threading.Timer(0.1, created.set).start()

    asyncio.run(end_while_posting())

    assert jobs.api.cancelled == [JOB]


def test_an_allowlist_it_cant_enforce_is_refused_before_any_job_is_created(jobs):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("workload.example",))

    with pytest.raises(NetworkPolicyUnsupportedError):
        asyncio.run(jobs.provider.create_container(image_name="ghcr.io/org/server:v1", port=18765, env={},
                                                   network_policy=policy))

    assert jobs.posted == []


def test_a_job_is_found_again_by_its_id_with_its_alias_and_one_that_ended_can_still_be_cancelled(jobs):
    jobs.api.labels = {"agentenv-alias": "c0123456789ab"}
    found = asyncio.run(jobs.provider.get_sandbox(JOB))
    assert (found.private_host, found.tunnel_urls) == (f"{PREFIX}c0123456789ab", {18765: f"https://{JOB}--18765.hf.jobs"})

    jobs.api.stages = ["COMPLETED"]
    ended = asyncio.run(jobs.provider.get_sandbox(JOB))
    asyncio.run(ended.terminate())
    assert jobs.api.cancelled == [JOB]
    with pytest.raises(RuntimeError, match="isn't running"):
        asyncio.run(ended.exec("true"))


def test_exec_gives_back_the_commands_exit_code_and_output(jobs):
    jobs.api.ports = (8000,)
    sandbox = asyncio.run(jobs.provider.create_sandbox(image_name="ghcr.io/org/agent:v1", port=8000, env={}))

    assert asyncio.run(sandbox.exec_with_output("pg_isready")) == (3, "out", "err")


@pytest.mark.parametrize("config, message", [
    ({"token": "hf_test", "region": "us"}, "unknown key"),
    ({"token": "hf_test", "flavor": " "}, "'flavor' must be a non-empty string"),
    ({}, "needs a 'token'"),
])
def test_a_config_it_cant_run_on_is_refused(monkeypatch, config, message):
    monkeypatch.setattr(hf_jobs, "get_token", lambda: None)

    with pytest.raises(ConfigError, match=message):
        HfJobsSandboxProvider.from_config(**config)


def test_without_a_configured_token_the_hugging_face_login_is_used(monkeypatch):
    monkeypatch.setattr(hf_jobs, "get_token", lambda: "hf_login")

    provider = HfJobsSandboxProvider.from_config(flavor="cpu-upgrade")

    assert (provider._token, provider._flavor) == ("hf_login", "cpu-upgrade")
    assert provider._network_group.startswith("agentenv-") and len(provider._network_group) <= 46


def test_a_huggingface_hub_whose_sandbox_helpers_changed_is_refused_as_the_provider_is_built(monkeypatch):
    monkeypatch.setattr(hf_jobs.hub_sandbox, "_bootstrap_job_spec", lambda api, hf_token, *, env, secrets: None)
    monkeypatch.delattr(hf_jobs.hub_sandbox, "_derive_sandbox_token")

    with pytest.raises(ConfigError, match="_bootstrap_job_spec takes no forward_hf_token, idle_timeout, sandbox_token, "
                                          "volumes; no _derive_sandbox_token"):
        HfJobsSandboxProvider.from_config(token="hf_test")


@pytest.mark.parametrize("args, command", [
    (None, ["docker-entrypoint.sh", "postgres"]),
    (["postgres", "-c", "fsync=off"], ["docker-entrypoint.sh", "postgres", "-c", "fsync=off"]),
])
def test_the_image_command_is_its_entrypoint_then_the_args_given_or_its_cmd(monkeypatch, args, command):
    monkeypatch.setattr(hf_jobs, "image_config",
                        lambda ref, auth: {"Entrypoint": ["docker-entrypoint.sh"], "Cmd": ["postgres"]})
    monkeypatch.setattr(hf_jobs, "get_config", lambda: SimpleNamespace(
        get_image_store_at=lambda name: SimpleNamespace(auth=lambda ref: None)))

    assert hf_jobs._image_command("postgres:16-alpine", args) == command


def test_the_job_spec_is_json(jobs):
    asyncio.run(jobs.provider.create_container(image_name="ghcr.io/org/server:v1", port=18765, env={},
                                               private_network=True))
    assert json.loads(json.dumps(jobs.posted[0])) == jobs.posted[0]
