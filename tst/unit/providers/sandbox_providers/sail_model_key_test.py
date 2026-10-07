"""Model-key injection on Sailboxes: the key stays at Sail, the Sailbox only ever holds a placeholder."""

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.sandbox_providers.sail import _sdk
from agent_env.providers.sandbox_providers.sail import model_key
from agent_env.providers.sandbox_providers.sail.model_key import (
    DOCKER_SHIM_PATH,
    PLACEHOLDER,
    ModelKeyInjection,
    docker_shim,
    secret_name,
)
from agent_env.providers.sandbox_providers.sail.provider import SailSandboxProvider
from agent_env.providers.sandbox_providers.sail.sandbox import SailSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy

_KEY = "sk-live-model-key-0123456789"
_ENV = {"LITELLM_API_KEY": _KEY, "LITELLM_BASE_URL": "https://llm.example.com/v1", "A2A_PORT": "8000"}


class _SecretInUse(Exception):
    pass


class _NotFound(Exception):
    pass


class _HostLost(Exception):
    pass


def _fake_sdk(sailbox):
    saved = []

    async def create_policy(name, document):
        policy = MagicMock(id=f"ep_{len(saved) + 1}", document=document)
        policy.name = name
        policy.delete.aio = AsyncMock()
        saved.append(policy)
        return policy

    secret = MagicMock()
    secret.delete.aio = AsyncMock()
    sdk = SimpleNamespace(
        App=SimpleNamespace(find=MagicMock(return_value=SimpleNamespace(id="app_1"))),
        Sailbox=SimpleNamespace(
            create=SimpleNamespace(aio=AsyncMock(return_value=sailbox)),
            get=SimpleNamespace(aio=AsyncMock(return_value=sailbox)),
        ),
        Image=SimpleNamespace(devbox=MagicMock(return_value="devbox-amd64")),
        AutoSleep=SimpleNamespace(never=lambda: "never", default=lambda: "default", not_before=lambda s: s),
        Secret=SimpleNamespace(set=SimpleNamespace(aio=AsyncMock()), get=SimpleNamespace(aio=AsyncMock(return_value=secret))),
        EgressPolicy=SimpleNamespace(
            create=SimpleNamespace(aio=AsyncMock(side_effect=create_policy)),
            get=SimpleNamespace(aio=AsyncMock(side_effect=lambda policy_id: next(p for p in saved if p.id == policy_id))),
        ),
        reset_transports=MagicMock(),
        NotFoundError=_NotFound,
        SecretInUseError=_SecretInUse,
        SailboxHostLostError=_HostLost,
        TransportError=_HostLost,
    )
    return sdk, saved, secret


def _sailbox(ports=(8000,)):
    sailbox = MagicMock(sailbox_id="sb_1", status="running")
    sailbox.listeners.aio = AsyncMock(return_value=[
        SimpleNamespace(guest_port=p, endpoint=SimpleNamespace(url=f"https://sb-1-{p}.sail.box")) for p in ports
    ])
    sailbox.terminate.aio = AsyncMock()
    sailbox.fs.write.aio = AsyncMock()
    sailbox.set_egress_policy.aio = AsyncMock()
    return sailbox


@pytest.fixture(autouse=True)
def fresh_key_state(monkeypatch):
    monkeypatch.setattr(_sdk, "_installed_key", None)
    monkeypatch.setattr(_sdk, "_apps", {})


@pytest.fixture(autouse=True)
def no_vm_setup(monkeypatch):
    monkeypatch.setattr(SailSandbox, "setup_vm_for_gateway", AsyncMock())


def _process(stdout=b""):
    async def chunks(data):
        yield data

    process = MagicMock()
    process.stdout_bytes = chunks(stdout)
    process.stderr_bytes = chunks(b"")
    process.wait = AsyncMock(return_value=SimpleNamespace(exit_code=0))
    return process


def test_secrets_are_named_from_the_key_so_keys_never_share_one():
    assert secret_name(_KEY) == secret_name(_KEY)
    assert secret_name(_KEY) != secret_name(_KEY + "x")
    assert secret_name(_KEY).startswith("AGENTENV_LITELLM_") and _KEY not in secret_name(_KEY)


def test_an_env_without_a_model_key_injects_nothing():
    assert ModelKeyInjection.for_env({"LITELLM_BASE_URL": "https://llm.example.com"}) is None
    assert ModelKeyInjection.for_env({"LITELLM_API_KEY": PLACEHOLDER}) is None


@pytest.mark.parametrize("base_url", ["http://llm.internal:4000", "", "llm.example.com"])
def test_injection_needs_an_https_model_endpoint(base_url):
    with pytest.raises(ValueError, match="only into HTTPS requests"):
        ModelKeyInjection.for_env({"LITELLM_API_KEY": _KEY, "LITELLM_BASE_URL": base_url})


def test_rules_set_both_auth_headers_from_the_secret_and_never_hold_the_key():
    injection = ModelKeyInjection.for_env(_ENV)
    ref = f"${{secrets.{secret_name(_KEY)}}}"
    assert injection.rules() == {
        "llm.example.com": [{"request": {"set": {"headers": {"authorization": f"Bearer {ref}", "x-api-key": ref}}}}]
    }
    assert _KEY not in repr(injection) and _KEY not in str(injection.rules())


def test_an_injection_is_recovered_from_its_saved_policy_only():
    injection = ModelKeyInjection.for_env(_ENV)
    recovered = ModelKeyInjection.from_document({"allowlist": ["x.example"], "rules": injection.rules()}, "ep_1")
    assert injection.matches(recovered) and recovered.policy_id == "ep_1" and recovered.key is None
    foreign = {"rules": {"api.github.com": [{"request": {"set": {"headers": {"authorization": "Bearer ${secrets.GITHUB_TOKEN}"}}}}]}}
    assert ModelKeyInjection.from_document(foreign, "ep_2") is None
    assert ModelKeyInjection.from_document({"allowlist": []}, "ep_3") is None


def test_the_key_is_recovered_only_when_it_named_the_secret():
    recovered = ModelKeyInjection.from_document({"rules": ModelKeyInjection.for_env(_ENV).rules()}, "ep_1")
    recovered.recover_key("some-other-key")
    assert recovered.key is None
    recovered.recover_key(_KEY)
    assert recovered.scrub(f"-e LITELLM_API_KEY='{_KEY}'") == f"-e LITELLM_API_KEY='{PLACEHOLDER}'"


@pytest.mark.parametrize(
    ("args", "expected_prefix"),
    [
        (["run", "-d", "--name", "agent-api", "img"], ["run", "-v"]),
        (["create", "--name", "agent-api", "img"], ["create", "-v"]),
        (["container", "run", "img"], ["container", "run", "-v"]),
        (["exec", "agent-api", "ls"], ["exec", "agent-api", "ls"]),
        (["load"], ["load"]),
    ],
)
def test_the_docker_shim_adds_the_ca_bundle_only_to_containers_it_starts(tmp_path, args, expected_prefix):
    shim = tmp_path / "docker"
    shim.write_text(docker_shim().replace("/usr/bin/docker", "printf '%s\\n'"))
    shim.chmod(0o755)
    argv = subprocess.run([str(shim), *args], capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
    assert argv[: len(expected_prefix)] == expected_prefix
    trusts = f"{model_key.VM_CA_BUNDLE}:{model_key.CONTAINER_CA_BUNDLE}:ro" in argv
    assert trusts is (expected_prefix[-1] == "-v")
    if trusts:
        assert "NODE_EXTRA_CA_CERTS=/etc/ssl/certs/sailbox-ca-bundle.crt" in argv
        assert argv[-len(args) + (2 if args[0] == "container" else 1):] == args[(2 if args[0] == "container" else 1):]


@pytest.mark.asyncio
async def test_an_agent_sailbox_gets_its_key_through_a_secret_and_a_saved_policy(caplog):
    caplog.set_level(logging.DEBUG)
    sailbox = _sailbox()
    sdk, saved, _ = _fake_sdk(sailbox)

    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)

    sdk.Secret.set.aio.assert_awaited_once_with(secret_name(_KEY), _KEY)
    (policy,) = saved
    assert policy.document == {"rules": ModelKeyInjection.for_env(_ENV).rules()}
    kwargs = sdk.Sailbox.create.aio.await_args.kwargs
    assert kwargs["egress_policy"] is policy
    assert _KEY not in repr(kwargs)
    sailbox.fs.write.aio.assert_awaited_once_with(DOCKER_SHIM_PATH, docker_shim(), mode=0o755)
    assert sandbox._injection.policy_id == policy.id
    assert _KEY not in caplog.text


@pytest.mark.asyncio
async def test_a_restricted_agent_sailbox_allows_the_model_endpoint_it_injects_into():
    sdk, saved, _ = _fake_sdk(_sailbox())
    restricted = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))

    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(
        image_name="agent:1", port=8000, env=_ENV, network_policy=restricted,
    )

    assert "llm.example.com" in saved[0].document["allowlist"]
    assert "llm.example.com" in sandbox.network_policy.allow_hosts


@pytest.mark.asyncio
async def test_injection_can_be_turned_off_to_pass_the_key_in():
    sdk, saved, _ = _fake_sdk(_sailbox())
    sandbox = await SailSandboxProvider(api_key="sail-key", inject_model_key=False, sdk=sdk).create_sandbox(
        image_name="agent:1", port=8000, env={**_ENV, "LITELLM_BASE_URL": "http://llm.internal:4000"},
    )
    sdk.Secret.set.aio.assert_not_awaited()
    assert saved == [] and sandbox._injection is None
    assert sdk.Sailbox.create.aio.await_args.kwargs["egress_policy"] == {}


@pytest.mark.asyncio
async def test_a_plain_endpoint_is_refused_before_anything_is_created():
    sdk, saved, _ = _fake_sdk(_sailbox())
    with pytest.raises(ValueError, match="only into HTTPS requests"):
        await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(
            image_name="agent:1", port=8000, env={**_ENV, "LITELLM_BASE_URL": "http://llm.internal:4000"},
        )
    sdk.Secret.set.aio.assert_not_awaited()
    sdk.Sailbox.create.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_create_deletes_the_policy_it_made():
    sdk, saved, _ = _fake_sdk(_sailbox())
    sdk.Sailbox.create.aio = AsyncMock(side_effect=RuntimeError("no capacity"))
    with pytest.raises(RuntimeError, match="no capacity"):
        await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)
    saved[0].delete.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_gateway_and_plain_vms_never_inject():
    sdk, saved, _ = _fake_sdk(_sailbox(ports=()))
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_vm(exposed_ports=[])
    assert saved == [] and sandbox._injection is None


@pytest.mark.asyncio
async def test_commands_and_files_carry_the_placeholder_never_the_key():
    sailbox = _sailbox()
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)

    await sandbox.exec_script(f"docker run -d -e LITELLM_API_KEY='{_KEY}' -e ANTHROPIC_API_KEY={_KEY} agent:1")
    await sandbox.write_host_file(f"key: {_KEY}\n".encode(), "/opt/agent/config.yaml")

    sent = [arg for call in sailbox.exec.aio.await_args_list for arg in call.args[0]]
    (script,) = [arg for arg in sent if "docker run" in arg]
    assert script.count(PLACEHOLDER) == 2
    assert not any(_KEY in arg for arg in sent)
    assert sailbox.fs.write.aio.await_args.args == ("/opt/agent/config.yaml", f"key: {PLACEHOLDER}\n".encode())


@pytest.mark.asyncio
async def test_terminate_deletes_the_policy_then_the_secret():
    sailbox = _sailbox()
    sdk, saved, secret = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)

    await sandbox.terminate()

    sailbox.terminate.aio.assert_awaited_once()
    saved[0].delete.aio.assert_awaited_once()
    sdk.Secret.get.aio.assert_awaited_once_with(secret_name(_KEY))
    secret.delete.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_secret_another_sailbox_still_uses_is_left_in_place(caplog):
    caplog.set_level(logging.INFO)
    sdk, _, secret = _fake_sdk(_sailbox())
    secret.delete.aio = AsyncMock(side_effect=_SecretInUse("in use"))
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)

    await sandbox.terminate()

    assert "still used by another Sailbox" in caplog.text


@pytest.mark.asyncio
async def test_widening_an_injected_sailbox_replaces_its_saved_policy():
    sailbox = _sailbox()
    sdk, saved, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(
        image_name="agent:1", port=8000, env=_ENV,
        network_policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",)),
    )

    await sandbox.apply_network_policy(sandbox.network_policy.with_hosts(["bucket.example"]))

    first, second = saved
    assert "bucket.example" in second.document["allowlist"] and second.document["rules"] == first.document["rules"]
    sailbox.set_egress_policy.aio.assert_awaited_once_with(second)
    first.delete.aio.assert_awaited_once()
    assert sandbox._injection.policy_id == second.id


@pytest.mark.asyncio
async def test_reconnect_restores_the_injection_and_scrubs_the_configured_key(monkeypatch):
    injection = ModelKeyInjection.for_env(_ENV)
    sailbox = _sailbox()
    sailbox.egress_policy = SimpleNamespace(policy_id="ep_9", document={"allowlist": ["llm.example.com"], "rules": injection.rules()})
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    monkeypatch.setattr(
        "agent_env.providers.sandbox_providers.sail.provider.get_config", lambda: MagicMock(get_litellm_api_key=lambda: _KEY)
    )

    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).get_sandbox("sb_1")

    assert sandbox.network_policy == NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("llm.example.com",))
    assert sandbox._injection.policy_id == "ep_9"
    await sandbox.exec_with_output("docker", "exec", "-e", f"LITELLM_API_KEY={_KEY}", "agent-api", "pytest")
    assert sailbox.exec.aio.await_args.args[0][3] == f"LITELLM_API_KEY={PLACEHOLDER}"


@pytest.mark.asyncio
async def test_reconnect_treats_someone_elses_saved_policy_as_unknown():
    sailbox = _sailbox()
    sailbox.egress_policy = SimpleNamespace(policy_id="ep_7", document={"allowlist": ["a.example"]})
    sdk, _, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).get_sandbox("sb_1")
    assert sandbox.network_policy is None and sandbox._injection is None
