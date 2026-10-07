"""Model-key injection on Sailboxes: the key stays at Sail, the Sailbox only ever holds a placeholder."""

import asyncio
import logging
import shlex
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
    carries_model_key,
    docker_shim,
    secret_name,
)
from agent_env.providers.sandbox_providers.sail.provider import SailSandboxProvider
from agent_env.providers.sandbox_providers.sail.sandbox import (
    ModelKeyRefusedError,
    SailSandbox,
    create_saved_policy,
    delete_policy_named,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy

_KEY = "sk-live-model-key-0123456789"
_ENV = {"LITELLM_API_KEY": _KEY, "LITELLM_BASE_URL": "https://llm.example.com/v1", "A2A_PORT": "8000"}


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
            list=SimpleNamespace(aio=AsyncMock(side_effect=lambda search: [p for p in saved if p.name == search])),
        ),
        reset_transports=MagicMock(),
        NotFoundError=_NotFound,
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
    monkeypatch.setattr(model_key, "_injected_keys", {})
    monkeypatch.setattr(model_key, "_holders", {})


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


def test_a_key_injected_by_this_process_is_recovered_even_when_it_is_not_the_configured_one():
    injected = ModelKeyInjection.for_env(_ENV)
    injected.hold("sb_1")
    recovered = ModelKeyInjection.from_document({"rules": injected.rules()}, "ep_1")
    recovered.recover_key("the-configured-model-key")
    assert recovered.key == _KEY


def test_in_another_process_only_the_configured_key_is_recovered(monkeypatch):
    rules = ModelKeyInjection.for_env(_ENV).rules()
    monkeypatch.setattr(model_key, "_injected_keys", {})
    recovered = ModelKeyInjection.from_document({"rules": rules}, "ep_1")
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
async def test_a_failed_create_deletes_its_policy_but_never_the_secret():
    sdk, saved, secret = _fake_sdk(_sailbox())
    sdk.Sailbox.create.aio = AsyncMock(side_effect=RuntimeError("no capacity"))
    with pytest.raises(RuntimeError, match="no capacity"):
        await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)
    saved[0].delete.aio.assert_awaited_once()
    secret.delete.aio.assert_not_awaited()
    assert secret_name(_KEY) not in model_key._injected_keys


@pytest.mark.asyncio
async def test_a_create_cancelled_while_saving_the_policy_forgets_the_key():
    sdk, _, secret = _fake_sdk(_sailbox())
    started = asyncio.Event()

    async def hang(name, document):
        started.set()
        await asyncio.Event().wait()

    sdk.EgressPolicy.create.aio = AsyncMock(side_effect=hang)
    task = asyncio.ensure_future(
        SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    sdk.Sailbox.create.aio.assert_not_awaited()
    secret.delete.aio.assert_not_awaited()
    assert secret_name(_KEY) not in model_key._injected_keys


@pytest.mark.asyncio
async def test_a_create_abandoned_in_flight_that_then_fails_deletes_its_policy():
    sdk, saved, secret = _fake_sdk(_sailbox())
    started, fail = asyncio.Event(), asyncio.Event()

    async def create(**_kwargs):
        started.set()
        await fail.wait()
        raise RuntimeError("no capacity")

    sdk.Sailbox.create.aio = create
    task = asyncio.ensure_future(
        SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved[0].delete.aio.assert_not_awaited()
    fail.set()
    for _ in range(10):
        await asyncio.sleep(0)
    saved[0].delete.aio.assert_awaited_once()
    secret.delete.aio.assert_not_awaited()


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
async def test_terminate_deletes_the_policy_and_keeps_the_secret():
    sailbox = _sailbox()
    sdk, saved, secret = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)

    await sandbox.terminate()

    sailbox.terminate.aio.assert_awaited_once()
    saved[0].delete.aio.assert_awaited_once()
    secret.delete.aio.assert_not_awaited()
    assert secret_name(_KEY) not in model_key._injected_keys


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
async def test_a_failed_policy_swap_deletes_the_replacement_and_keeps_the_old_one():
    sailbox = _sailbox()
    sdk, saved, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(
        image_name="agent:1", port=8000, env=_ENV,
        network_policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",)),
    )
    sailbox.set_egress_policy.aio = AsyncMock(side_effect=RuntimeError("api down"))

    with pytest.raises(RuntimeError, match="api down"):
        await sandbox.apply_network_policy(sandbox.network_policy.with_hosts(["bucket.example"]))

    first, second = saved
    second.delete.aio.assert_awaited_once()
    first.delete.aio.assert_not_awaited()
    assert sandbox._injection.policy_id == first.id


@pytest.mark.asyncio
async def test_reconnect_in_another_process_scrubs_the_configured_key(monkeypatch):
    injection = ModelKeyInjection.for_env(_ENV)
    sailbox = _sailbox()
    sailbox.egress_policy = SimpleNamespace(policy_id="ep_9", document={"allowlist": ["llm.example.com"], "rules": injection.rules()})
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    monkeypatch.setattr(model_key, "_injected_keys", {})
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


@pytest.mark.asyncio
async def test_reconnect_in_the_deploying_process_scrubs_the_agents_own_key(monkeypatch):
    agent_key = "sk-agent-override-key-42"
    sailbox = _sailbox()
    sdk, _, _ = _fake_sdk(sailbox)
    provider = SailSandboxProvider(api_key="sail-key", sdk=sdk)
    created = await provider.create_sandbox(image_name="agent:1", port=8000, env={**_ENV, "LITELLM_API_KEY": agent_key})
    (policy,) = [c.args[1] for c in sdk.EgressPolicy.create.aio.await_args_list]
    sailbox.egress_policy = SimpleNamespace(policy_id=created._injection.policy_id, document=policy)
    sailbox.exec.aio = AsyncMock(return_value=_process())
    monkeypatch.setattr(
        "agent_env.providers.sandbox_providers.sail.provider.get_config", lambda: MagicMock(get_litellm_api_key=lambda: _KEY)
    )

    reconnected = await provider.get_sandbox("sb_1")
    await reconnected.exec_with_output("docker", "exec", "-e", f"LITELLM_API_KEY={agent_key}", "agent-api", "pytest")

    assert sailbox.exec.aio.await_args.args[0][3] == f"LITELLM_API_KEY={PLACEHOLDER}"


@pytest.mark.parametrize(
    ("text", "shell", "carries"),
    [
        (f"docker run -e LITELLM_API_KEY='{_KEY}' img", True, True),
        (f"docker exec -e ANTHROPIC_API_KEY={_KEY} c pytest", True, True),
        (f"- LITELLM_API_KEY={_KEY}", False, True),
        ("docker run -e LITELLM_API_KEY='$test-secret' img", True, True),
        ("docker run -e LITELLM_API_KEY=''\\''sk-secret' img", True, True),
        ("docker run -e LITELLM_API_KEY=$test-secret img", True, True),
        ('docker run -e LITELLM_API_KEY="$LITELLM_API_KEY-x" img', True, True),
        ("LITELLM_API_KEY=$test-secret", False, True),
        ("LITELLM_API_KEY=$LITELLM_API_KEY", False, True),
        (f"-e LITELLM_API_KEY='{PLACEHOLDER}'", True, False),
        (f"-e LITELLM_API_KEY={PLACEHOLDER}", False, False),
        ('-e LITELLM_API_KEY="$LITELLM_API_KEY"', True, False),
        ("-e LITELLM_API_KEY=${LITELLM_API_KEY}", True, False),
        ("-e LITELLM_API_KEY=$LITELLM_API_KEY", True, False),
        ("echo LITELLM_API_KEY=", True, False),
        ("docker run -e OTHER=1 img", True, False),
    ],
)
def test_a_model_key_assignment_is_detected_as_the_shell_reads_it(text, shell, carries):
    assert carries_model_key(text, shell=shell) is carries


def test_the_agents_own_escaping_of_a_key_is_seen_through():
    key = "'sk-starts-with-a-quote"
    escaped = key.replace("'", "'\\''")
    assert carries_model_key(f"docker run -e LITELLM_API_KEY='{escaped}' img", shell=True)


@pytest.mark.asyncio
async def test_a_dollar_value_in_a_plain_argument_is_a_literal_key_and_refused():
    sailbox = _sailbox(ports=())
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_vm(exposed_ports=[])
    with pytest.raises(ModelKeyRefusedError):
        await sandbox.exec_with_output("docker", "exec", "-e", "LITELLM_API_KEY=$looks-like-a-var", "agent-api", "true")
    await sandbox.exec_with_output("bash", "-c", 'docker exec -e LITELLM_API_KEY="$LITELLM_API_KEY" agent-api true')


@pytest.mark.asyncio
async def test_a_sailbox_without_injection_refuses_a_model_key_rather_than_take_it():
    sailbox = _sailbox(ports=())
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_vm(exposed_ports=[])
    sailbox.exec.aio.reset_mock()

    with pytest.raises(ModelKeyRefusedError, match="Deploy the agent on its own sandbox"):
        await sandbox.exec_script(f"docker run -d --name agent-api -e LITELLM_API_KEY='{_KEY}' agent:1")
    with pytest.raises(ModelKeyRefusedError):
        await sandbox.write_host_file(f"LITELLM_API_KEY={_KEY}\n".encode(), "/opt/agent/.env")
    assert not any(_KEY in arg for call in sailbox.exec.aio.await_args_list for arg in call.args[0])
    sailbox.fs.write.aio.assert_not_awaited()

    await sandbox.exec_script('docker run -e LITELLM_API_KEY="$LITELLM_API_KEY" agent:1')


@pytest.mark.asyncio
async def test_with_injection_off_keys_pass_through_as_before():
    sailbox = _sailbox(ports=())
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", inject_model_key=False, sdk=sdk).create_vm(exposed_ports=[])
    await sandbox.exec_script(f"docker run -e LITELLM_API_KEY='{_KEY}' agent:1")
    assert _KEY in sailbox.exec.aio.await_args.args[0][2]


@pytest.mark.asyncio
async def test_a_reconnected_handle_that_cannot_recover_the_key_refuses_it(monkeypatch):
    sailbox = _sailbox()
    sailbox.egress_policy = SimpleNamespace(policy_id="ep_9", document={"rules": ModelKeyInjection.for_env(_ENV).rules()})
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    monkeypatch.setattr(model_key, "_injected_keys", {})
    monkeypatch.setattr(
        "agent_env.providers.sandbox_providers.sail.provider.get_config", lambda: MagicMock(get_litellm_api_key=lambda: "other")
    )
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).get_sandbox("sb_1")
    with pytest.raises(ModelKeyRefusedError):
        await sandbox.exec_with_output("docker", "exec", "-e", f"LITELLM_API_KEY={_KEY}", "agent-api", "pytest")
    sailbox.exec.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminate_deletes_the_policy_actually_applied_even_if_another_handle_replaced_it():
    sailbox = _sailbox()
    sdk, saved, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(image_name="agent:1", port=8000, env=_ENV)
    replacement = await sdk.EgressPolicy.create.aio("agentenv-replacement", saved[0].document)
    sailbox.egress_policy = SimpleNamespace(policy_id=replacement.id, document=replacement.document)

    await sandbox.terminate()

    replacement.delete.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_policy_create_whose_response_is_lost_is_found_by_name_and_deleted():
    sdk, saved, _ = _fake_sdk(_sailbox())
    created = sdk.EgressPolicy.create.aio.side_effect

    async def create_then_lose(name, document):
        await created(name, document)
        raise TimeoutError("response lost")

    sdk.EgressPolicy.create.aio = AsyncMock(side_effect=create_then_lose)
    with pytest.raises(TimeoutError):
        await create_saved_policy(sdk, "agentenv-lost", {"rules": {}})
    saved[0].delete.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_key_is_remembered_until_the_last_box_here_that_needs_it_is_gone():
    first, second = _sailbox(), _sailbox()
    second.sailbox_id = "sb_2"
    sdk, _, secret = _fake_sdk(first)
    sdk.Sailbox.create.aio = AsyncMock(side_effect=[first, second])
    provider = SailSandboxProvider(api_key="sail-key", sdk=sdk)
    one = await provider.create_sandbox(image_name="agent:1", port=8000, env=_ENV)
    two = await provider.create_sandbox(image_name="agent:1", port=8000, env=_ENV)

    await one.terminate()
    assert model_key._injected_keys[secret_name(_KEY)] == _KEY
    await two.terminate()
    assert secret_name(_KEY) not in model_key._injected_keys
    secret.delete.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_launch_never_touches_the_secret_a_concurrent_launch_with_the_same_key_needs():
    sailbox = _sailbox()
    sdk, _, secret = _fake_sdk(sailbox)
    first_waiting, go = asyncio.Event(), asyncio.Event()
    calls = 0

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_waiting.set()
            await go.wait()
            return sailbox
        raise RuntimeError("no capacity")

    sdk.Sailbox.create.aio = create
    provider = SailSandboxProvider(api_key="sail-key", sdk=sdk)
    pending = asyncio.ensure_future(provider.create_sandbox(image_name="agent:1", port=8000, env=_ENV))
    await asyncio.wait_for(first_waiting.wait(), timeout=5)
    with pytest.raises(RuntimeError, match="no capacity"):
        await provider.create_sandbox(image_name="agent:1", port=8000, env=_ENV)
    assert model_key._injected_keys[secret_name(_KEY)] == _KEY
    go.set()
    sandbox = await asyncio.wait_for(pending, timeout=5)
    await sandbox.terminate()
    secret.delete.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminate_waits_for_a_policy_swap_in_flight_on_another_handle():
    sailbox = _sailbox()
    sdk, saved, _ = _fake_sdk(sailbox)
    provider = SailSandboxProvider(api_key="sail-key", sdk=sdk)
    sandbox = await provider.create_sandbox(
        image_name="agent:1", port=8000, env=_ENV,
        network_policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",)),
    )
    sailbox.egress_policy = SimpleNamespace(policy_id=saved[0].id, document=saved[0].document)
    other = SailSandbox(sailbox, sdk=sdk, tunnel_urls={}, network_policy=sandbox.network_policy,
                        injection=ModelKeyInjection.from_document(saved[0].document, saved[0].id))
    swapping, release = asyncio.Event(), asyncio.Event()

    async def slow_set(policy):
        swapping.set()
        await release.wait()
        sailbox.egress_policy = SimpleNamespace(policy_id=policy.id, document=policy.document)

    sailbox.set_egress_policy.aio = AsyncMock(side_effect=slow_set)
    sdk.Sailbox.get.aio = AsyncMock(side_effect=lambda _id: sailbox)
    swap = asyncio.ensure_future(other._allow_download_hosts(["https://bucket.example/a"], "download"))
    await asyncio.wait_for(swapping.wait(), timeout=5)
    teardown = asyncio.ensure_future(sandbox.terminate())
    await asyncio.sleep(0.01)
    sailbox.terminate.aio.assert_not_awaited()
    release.set()
    await asyncio.wait_for(asyncio.gather(swap, teardown), timeout=5)

    replacement = saved[-1]
    replacement.delete.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_lost_response_cleanup_retries_through_a_brief_outage(monkeypatch, caplog):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail.sandbox.asyncio.sleep", AsyncMock())
    sdk, saved, _ = _fake_sdk(_sailbox())
    await sdk.EgressPolicy.create.aio("agentenv-lost", {"rules": {}})
    listing = sdk.EgressPolicy.list.aio.side_effect
    sdk.EgressPolicy.list.aio = AsyncMock(side_effect=[RuntimeError("down"), listing(search="agentenv-lost")])

    await delete_policy_named(sdk, "agentenv-lost")

    saved[0].delete.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_lost_response_cleanup_that_never_reaches_sail_names_the_policy_to_sweep(monkeypatch, caplog):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail.sandbox.asyncio.sleep", AsyncMock())
    sdk, _, _ = _fake_sdk(_sailbox())
    sdk.EgressPolicy.list.aio = AsyncMock(side_effect=RuntimeError("down"))

    await delete_policy_named(sdk, "agentenv-lost")

    assert sdk.EgressPolicy.list.aio.await_count == 3
    assert "Egress policy agentenv-lost may be left in Sail" in caplog.text


@pytest.mark.asyncio
async def test_lost_response_cleanup_retries_a_failed_delete_too(monkeypatch):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail.sandbox.asyncio.sleep", AsyncMock())
    sdk, saved, _ = _fake_sdk(_sailbox())
    await sdk.EgressPolicy.create.aio("agentenv-lost", {"rules": {}})
    saved[0].delete.aio = AsyncMock(side_effect=[RuntimeError("down"), None])

    await delete_policy_named(sdk, "agentenv-lost")

    assert saved[0].delete.aio.await_count == 2


@pytest.mark.asyncio
async def test_a_direct_policy_update_waits_for_terminate_on_another_handle():
    sailbox = _sailbox()
    sdk, saved, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(
        image_name="agent:1", port=8000, env=_ENV,
        network_policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",)),
    )
    reading, release = asyncio.Event(), asyncio.Event()

    async def slow_get(_id):
        reading.set()
        await release.wait()
        return SimpleNamespace(egress_policy=SimpleNamespace(policy_id=saved[0].id, document=saved[0].document))

    sdk.Sailbox.get.aio = AsyncMock(side_effect=slow_get)
    other = SailSandbox(sailbox, sdk=sdk, tunnel_urls={}, network_policy=sandbox.network_policy,
                        injection=ModelKeyInjection.from_document(saved[0].document, saved[0].id))
    teardown = asyncio.ensure_future(sandbox.terminate())
    await asyncio.wait_for(reading.wait(), timeout=5)
    update = asyncio.ensure_future(other.apply_network_policy(sandbox.network_policy.with_hosts(["late.example"])))
    await asyncio.sleep(0.01)
    assert len(saved) == 1
    release.set()
    await asyncio.wait_for(asyncio.gather(teardown, update), timeout=5)
    sailbox.terminate.aio.assert_awaited_once()
    assert sailbox.set_egress_policy.aio.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("quoting", ["agent", "shlex"])
async def test_a_key_with_an_apostrophe_is_scrubbed_in_its_shell_encoded_form(quoting):


    key = "sk-a'b-quoted-key"
    sailbox = _sailbox()
    sailbox.exec.aio = AsyncMock(return_value=_process())
    sdk, _, _ = _fake_sdk(sailbox)
    sandbox = await SailSandboxProvider(api_key="sail-key", sdk=sdk).create_sandbox(
        image_name="agent:1", port=8000, env={**_ENV, "LITELLM_API_KEY": key},
    )
    encoded = "'" + key.replace("'", "'\\''") + "'" if quoting == "agent" else shlex.quote(key)

    await sandbox.exec_script(f"docker run -d --name agent-api -e LITELLM_API_KEY={encoded} agent:1")

    script = sailbox.exec.aio.await_args.args[0][2]
    assert "sk-a" not in script and PLACEHOLDER in script
    assert shlex.split(script.split("LITELLM_API_KEY=", 1)[1])[0] == PLACEHOLDER
