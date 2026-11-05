"""The lume provider against a fake ``lume``, which keeps its VMs in a file: a VM is cloned from its VM bundle, sized,
given the deploy's environment as the protocol's seed in the folder it shares, run, and relayed from this Mac's
loopback, and ended past its lifetime by the next VM admitted; a reset clones it afresh under its MAC address; a
registry image is pulled once by its tag, checked against its digest before and after, a pull its tag moved under
deleted; two VMs run at most, a reset holding its slot; and a builder shares its build folder and is kept as a local
VM bundle."""

import asyncio
import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest

from agentenv_protocol.computer import seed

from agent_env.providers.sandbox_providers.lume import provider as lume
from agent_env.providers.sandbox_providers.lume.provider import LumeSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, reachable_url

FAKE = textwrap.dedent('''\
    import contextlib, fcntl, json, os, signal, sys, time
    path = os.environ["FAKE_LUME_STATE"]
    @contextlib.contextmanager
    def held():
        with open(path + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
    def load():
        try:
            return json.load(open(path))
        except FileNotFoundError:
            return {"vms": {"bench": {"status": "stopped", "macAddress": "aa:00:00:00:00:00"}}, "calls": []}
    def save(state):
        part = f"{path}.{os.getpid()}.part"
        json.dump(state, open(part, "w"))
        os.replace(part, path)
    args = sys.argv[1:]
    command, name = args[0], args[1] if len(args) > 1 else None
    if command == "run":
        with held():
            state = load()
            state["calls"].append(args)
            if not os.environ.get("FAKE_LUME_NO_SESSION"):
                state["vms"][name].update(status="running")
            if not os.environ.get("FAKE_LUME_NO_SESSION") and not os.environ.get("FAKE_LUME_NO_ADDRESS"):
                state["vms"][name].update(ipAddress="192.168.64.%d" % len(state["vms"]))
            save(state)
        if os.environ.get("FAKE_LUME_NO_SESSION"):
            sys.exit("Unable to access security information. The virtual machine encountered a security error")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        while True:
            time.sleep(0.05)
    lock = held()
    lock.__enter__()
    state = load()
    state["calls"].append(args)
    vms = state["vms"]
    print("[2026-11-04T16:56:06Z] INFO: lume logs on stdout too command=" + command)
    if command == "get":
        if name not in vms:
            save(state)
            sys.exit(f"VM not found: {name}")
        print(json.dumps([{"name": name, **vms[name]}]))
    elif command == "ls":
        print(json.dumps([{"name": key, **vm} for key, vm in vms.items()]))
    elif command in ("clone", "pull"):
        vms[args[2]] = {"status": "stopped", "macAddress": "aa:00:00:00:00:%02x" % len(vms)}
    elif command == "set":
        for flag, value in zip(args[2::2], args[3::2]):
            vms[name][flag.lstrip("-")] = value
    elif command == "stop":
        vms[name].update(status="stopped", ipAddress=None)
    elif command == "delete":
        vms.pop(name, None)
        if os.environ.get("FAKE_LUME_SLOW_DELETE"):
            save(state)
            lock.__exit__(None, None, None)
            time.sleep(1)
            sys.exit(0)
    save(state)
''')
SSH = textwrap.dedent('''\
    import json, os, sys, time
    path = os.environ["FAKE_LUME_STATE"]
    args = sys.argv[1:]
    *_, login, command = args
    with open(path + ".ssh", "a") as calls:
        calls.write(json.dumps({"args": args, "askpass": os.environ.get("SSH_ASKPASS_REQUIRE") == "force"
                                and os.environ.get("AGENTENV_LUME_SSH_PASSWORD") == "s3cret-pw"}) + "\\n")
    print(command)
    if command == "true" and os.environ.get("FAKE_SSH_STALL"):
        time.sleep(60)
    if "shutdown -h now" in command and os.environ.get("FAKE_SSH_HANG"):
        time.sleep(60)
    if "shutdown -h now" in command:
        state = json.load(open(path))
        for vm in state["vms"].values():
            if vm.get("ipAddress") == login.split("@")[1]:
                vm.update(status="stopped", ipAddress=None)
        json.dump(state, open(path, "w"))
''')


@pytest.fixture
def mac(tmp_path, monkeypatch):
    """A fake lume and a relay that records its routes, with the provider's state under this test's state root."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lume").write_text(f"#!{sys.executable}\n{FAKE}")
    (bin_ / "ssh").write_text(f"#!{sys.executable}\n{SSH}")
    (bin_ / "relay").write_text(f"#!/bin/sh\necho \"$@\" > {tmp_path}/relay.part && mv {tmp_path}/relay.part "
                                f"{tmp_path}/relay.args\nexec sleep 60\n")
    for name in ("lume", "ssh", "relay"):
        (bin_ / name).chmod(0o755)
    monkeypatch.setenv("FAKE_LUME_STATE", str(tmp_path / "lume.json"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(lume, "_free_port", lambda: 48123)
    monkeypatch.setattr(lume, "_POLL_S", 0.05)
    monkeypatch.setattr(lume, "_local_grants", lambda: None)
    provider = LumeSandboxProvider(lume=str(bin_ / "lume"), ssh=str(bin_ / "ssh"), relay_python=str(bin_ / "relay"),
                                   ssh_password="s3cret-pw", boot_timeout_s=10)
    provider.fake = tmp_path
    yield provider
    for vm in _state(tmp_path)["vms"]:
        try:
            pid = json.loads((tmp_path / "state" / "agent-env" / "lume" / vm / "state.json").read_text())["run"]
            os.killpg(pid, signal.SIGTERM)
        except (FileNotFoundError, KeyError, ProcessLookupError):
            pass


def _state(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "lume.json").read_text()) if (tmp_path / "lume.json").exists() else {"vms": {}}


@pytest.mark.asyncio
async def test_a_vm_is_a_clone_of_its_bundle_given_its_seed_relayed_and_policed_and_terminating_it_deletes_it(mac):
    env = {"ENVIRONMENT_NAME": "desk", "AGENTENV_SETTING_RESOLUTION": "1280x800"}

    sandbox = await mac.create_vm(image="local:bench", env=env, exposed_ports=[18765], cpu=4, memory=8192,
                                  network_policy=NetworkPolicy.from_dict({"mode": "allow_all"}))

    calls, folder = _state(mac.fake)["calls"], mac.fake / "state" / "agent-env" / "lume" / sandbox.sandbox_id
    assert ["clone", "bench", sandbox.sandbox_id] in calls
    assert ["set", sandbox.sandbox_id, "--cpu", "4", "--memory", "8192MB", "--display", "1280x800"] in calls
    assert ["run", sandbox.sandbox_id, "--display", "none", "--vnc", "disabled",
            f"--shared-dir={folder / lume.SEED_FOLDER}:ro"] in calls
    assert (folder / lume.SEED_FOLDER / "user-data").read_text() == seed.user_data(env)
    assert (folder / lume.SEED_FOLDER / seed.OLDEST).read_text() == "1\n"
    assert sandbox.tunnel_urls == {18765: "http://127.0.0.1:48123"}
    assert reachable_url("http://127.0.0.1:48123/mcp", from_sandbox_type="lume",
                         to_sandbox_type="local") == "http://host.docker.internal:48123/mcp"
    assert sandbox.network_policy.mode is NetworkMode.ALLOW_ALL
    address = _state(mac.fake)["vms"][sandbox.sandbox_id]["ipAddress"]
    assert _waited(lambda: (mac.fake / "relay.args").read_text().split()) == ["-I", str(lume.RELAY),
                                                                             f"127.0.0.1:48123={address}:18765"]

    await sandbox.terminate()

    assert sandbox.sandbox_id not in _state(mac.fake)["vms"] and not folder.exists()
    with pytest.raises(NetworkPolicyUnsupportedError, match="cannot enforce network policy mode 'allowlist'"):
        await mac.create_vm(image="local:bench", env={}, network_policy=NetworkPolicy.from_dict(
            {"mode": "allowlist", "allow_hosts": ["example.com"]}))
    (mac.fake / "relay.args").unlink()
    mac.relay_host = "10.0.0.5"
    other = await mac.create_vm(image="local:bench", env={}, exposed_ports=[18765])
    address = _state(mac.fake)["vms"][other.sandbox_id]["ipAddress"]
    assert other.tunnel_urls == {18765: "http://10.0.0.5:48123"}
    assert _waited(lambda: (mac.fake / "relay.args").read_text().split())[-1] == f"10.0.0.5:48123={address}:18765"
    await other.terminate()


@pytest.mark.asyncio
async def test_a_local_stores_grants_reach_a_vm_at_its_loopback_through_ssh_forwarding_with_the_trust_in_its_seed(
        mac, monkeypatch, tmp_path):
    trust = tmp_path / "trust"
    trust.mkdir()
    (trust / "ca-bundle.pem").write_text("roots and the CA")
    (trust / "ca.pem").write_text("the CA")
    monkeypatch.setattr(lume, "_local_grants", lambda: (54321, trust))

    sandbox = await mac.create_vm(image="local:bench", env={"ENVIRONMENT_NAME": "desk"}, exposed_ports=[18765])

    folder = mac.fake / "state" / "agent-env" / "lume" / sandbox.sandbox_id / lume.SEED_FOLDER
    inside = f"{lume.SHARED_FILES}/{lume.SEED_FOLDER}/{lume.TRUST_FOLDER}"
    assert seed.read_user_data((folder / "user-data").read_text()) == {
        "ENVIRONMENT_NAME": "desk", "SSL_CERT_FILE": f"{inside}/ca-bundle.pem",
        "REQUESTS_CA_BUNDLE": f"{inside}/ca-bundle.pem", "NODE_EXTRA_CA_CERTS": f"{inside}/ca.pem"}
    assert (folder / lume.TRUST_FOLDER / "ca.pem").read_text() == "the CA"
    address = _state(mac.fake)["vms"][sandbox.sandbox_id]["ipAddress"]
    [forward] = _waited(lambda: _forwards(mac.fake))
    assert forward["args"][-3:] == ["-R", "127.0.0.1:54321:127.0.0.1:54321", f"lume@{address}"]
    assert "ExitOnForwardFailure=yes" in forward["args"] and forward["askpass"]
    assert LumeSandboxProvider.REACHES_HOST_AT == "127.0.0.1"

    ports = iter([54322, 54323])
    monkeypatch.setattr(lume, "_local_grants", lambda: (next(ports), trust))
    await asyncio.gather(mac.get_sandbox(sandbox.sandbox_id), mac.get_sandbox(sandbox.sandbox_id))

    assert {call["args"][-2] for call in _waited(lambda: _forwards(mac.fake, 3))[1:]} == {
        "127.0.0.1:54322:127.0.0.1:54322", "127.0.0.1:54323:127.0.0.1:54323"}
    forwards = json.loads((folder.parent / "state.json").read_text())["forwards"]
    assert set(forwards) == {"54321", "54322", "54323"}

    monkeypatch.setattr(lume, "_local_grants", lambda: (54324, trust))
    monkeypatch.setenv("FAKE_LUME_SLOW_DELETE", "1")
    _, reconnect = await asyncio.gather(sandbox.terminate(), mac.get_sandbox(sandbox.sandbox_id),
                                        return_exceptions=True)

    assert isinstance(reconnect, SandboxNotFoundError) and len(_forwards(mac.fake)) == 3
    for pid in forwards.values():
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.asyncio
async def test_a_reset_clones_the_vm_afresh_under_its_mac_address_with_the_new_seed_and_one_not_up_is_stopped(
        mac, monkeypatch):
    sandbox = await mac.create_vm(image="local:bench", env={"N": "1"}, exposed_ports=[18765])
    mac_address = _state(mac.fake)["vms"][sandbox.sandbox_id]["macAddress"]

    await mac.reset_vm(sandbox, image="local:bench", env={"N": "2"})

    calls = _state(mac.fake)["calls"]
    reset = calls[calls.index(["stop", sandbox.sandbox_id, "--force"]) + 1:]
    assert reset[:3] == [["delete", sandbox.sandbox_id, "--force"], ["clone", "bench", sandbox.sandbox_id],
                         ["set", sandbox.sandbox_id, "--mac-address", mac_address]]
    folder = mac.fake / "state" / "agent-env" / "lume" / sandbox.sandbox_id
    assert seed.read_user_data((folder / lume.SEED_FOLDER / "user-data").read_text()) == {"N": "2"}
    monkeypatch.setenv("FAKE_LUME_NO_ADDRESS", "1")
    mac.boot_timeout_s = 1
    with pytest.raises(TimeoutError, match="had no address within 1 s"):
        await mac.reset_vm(sandbox, image="local:bench", env={"N": "3"})
    assert _state(mac.fake)["vms"][sandbox.sandbox_id]["status"] == "stopped"
    assert not {"run", "relay"} & set(json.loads((folder / "state.json").read_text()))
    await sandbox.terminate()


@pytest.mark.asyncio
async def test_a_registry_bundle_is_pulled_once_by_its_tag_checked_before_and_after_and_one_whose_tag_moved_deleted(
        mac, monkeypatch):
    pinned = []
    monkeypatch.setattr(lume, "pin_digest", lambda ref, auth: pinned.append(ref) or ref)
    image = "ghcr.io/trycua/macos:26-x@sha256:" + "7c" * 32

    first = await mac.create_vm(image=image, env={})
    await first.terminate()
    second = await mac.create_vm(image=image, env={})
    await second.terminate()

    pulls = [call for call in _state(mac.fake)["calls"] if call[0] == "pull"]
    assert pulls == [["pull", "macos:26-x", "agentenv-base-7c7c7c7c7c7c", "--registry", "ghcr.io", "--organization",
                      "trycua"]]
    assert pinned == [image, image]
    with pytest.raises(ValueError, match="Lume pulls by tag"):
        await mac.create_vm(image="ghcr.io/trycua/macos@sha256:" + "ab" * 32, env={})
    checks = iter([None, ValueError("its tag names sha256:" + "cd" * 32)])

    def moved(ref, auth):
        if (raised := next(checks)) is not None:
            raise raised
        return ref

    monkeypatch.setattr(lume, "pin_digest", moved)
    with pytest.raises(ValueError, match="its tag moved while it was pulled, so what was pulled is deleted"):
        await mac.create_vm(image="ghcr.io/trycua/macos:26-y@sha256:" + "ab" * 32, env={})
    assert "agentenv-base-abababababab" not in _state(mac.fake)["vms"]


@pytest.mark.asyncio
async def test_a_local_bundle_this_mac_lacks_is_refused_a_third_vm_is_refused_even_mid_reset_and_an_old_one_ended(mac):
    with pytest.raises(ValueError, match="no VM bundle named 'gone'"):
        await mac.create_vm(image="local:gone", env={})
    sandboxes = [await mac.create_vm(image="local:bench", env={}) for _ in range(2)]

    with pytest.raises(RuntimeError, match="at most 2 .Apple's licence allows 2."):
        await mac.create_vm(image="local:bench", env={})
    reset, third = await asyncio.gather(mac.reset_vm(sandboxes[0], image="local:bench", env={}),
                                        mac.create_vm(image="local:bench", env={}), return_exceptions=True)
    assert reset is None and isinstance(third, RuntimeError) and "at most 2" in str(third)

    for sandbox in sandboxes:
        await sandbox.terminate()
    abandoned = await mac.create_vm(image="local:bench", env={}, timeout=0)
    sandboxes = [await mac.create_vm(image="local:bench", env={}) for _ in range(2)]
    assert abandoned.sandbox_id not in _state(mac.fake)["vms"]
    for sandbox in sandboxes:
        await sandbox.terminate()
    with pytest.raises(ValueError, match="max_running is 1 to 2"):
        LumeSandboxProvider(max_running=3)


@pytest.mark.asyncio
async def test_a_vm_lume_cant_run_outside_a_login_session_fails_saying_so_and_leaves_nothing(mac, monkeypatch):
    monkeypatch.setenv("FAKE_LUME_NO_SESSION", "1")

    with pytest.raises(RuntimeError, match="runs a VM only in a logged-in session"):
        await mac.create_vm(image="local:bench", env={})

    assert list(_state(mac.fake)["vms"]) == ["bench"]


@pytest.mark.asyncio
async def test_a_builder_shares_its_build_folder_and_is_kept_as_a_local_vm_bundle_and_a_hung_shutdown_times_out(
        mac, tmp_path, monkeypatch):
    build = tmp_path / lume.BUILD_FOLDER
    build.mkdir()

    builder = await mac.open_builder("local:bench", "agentenv-computer-0123", build)
    _, said, _ = await builder.exec_with_output("bash", "-c", "echo hi")
    image = await mac.keep_builder(builder)

    calls, ssh = _state(mac.fake)["calls"], [json.loads(line) for line in (mac.fake / "lume.json.ssh").read_text().splitlines()]
    assert ["run", "agentenv-computer-0123", "--display", "none", "--vnc", "disabled", f"--shared-dir={build}:rw"] in calls
    assert "bash -c 'echo hi'" in said and all(call["askpass"] for call in ssh)
    assert any(call["args"][-1].startswith("sh -c 'sync; sudo -S") for call in ssh)
    assert not any("s3cret-pw" in arg for call in ssh for arg in call["args"])
    assert image == "local:agentenv-computer-0123" and await mac.has_vm_bundle(image)
    with pytest.raises(ValueError, match="a builder's folder is named agentenv-build"):
        await mac.open_builder("local:bench", "other", tmp_path)
    hung = await mac.open_builder("local:bench", "agentenv-computer-4567", build)
    monkeypatch.setenv("FAKE_SSH_HANG", "1")
    mac.boot_timeout_s = 1
    with pytest.raises(TimeoutError, match="agentenv-computer-4567 didn't shut down within 1 s, so it isn't kept"):
        await mac.keep_builder(hung)
    await hung.terminate()
    monkeypatch.setenv("FAKE_SSH_STALL", "1")
    with pytest.raises(TimeoutError, match="agentenv-computer-89ab didn't answer ssh within 1 s"):
        await mac.open_builder("local:bench", "agentenv-computer-89ab", build)
    assert "agentenv-computer-89ab" not in _state(mac.fake)["vms"]


def test_it_boots_macos_vm_bundles_on_apple_silicon_alone(monkeypatch):
    monkeypatch.setattr(lume.sys, "platform", "darwin")
    monkeypatch.setattr(lume.platform, "machine", lambda: "arm64")
    assert LumeSandboxProvider.vm_bundle_platforms() == {"macos/arm64"}
    assert LumeSandboxProvider.native_platforms() == frozenset()
    monkeypatch.setattr(lume.sys, "platform", "linux")
    assert LumeSandboxProvider.vm_bundle_platforms() == frozenset()


def _forwards(fake: Path, at_least: int = 1) -> list[dict]:
    """The fake ssh's forwarding calls, a FileNotFoundError until ``at_least`` are logged."""
    calls = [json.loads(line) for line in (fake / "lume.json.ssh").read_text().splitlines()]
    if len(forwards := [call for call in calls if "-N" in call["args"]]) < at_least:
        raise FileNotFoundError(f"fewer than {at_least} forwards logged yet")
    return forwards


def _waited(read, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return read()
        except FileNotFoundError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
