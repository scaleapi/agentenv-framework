"""Computer envs of a base with a disk form: each arch's disk is made from its built container, on the base's disk or on
one made from the base's container and the boot recipe, by a build that runs on the builder's own platform; reused by
its key; probed where a provider boots disks natively, else left unverified for a deploy's first boot with KVM to
probe; and written as the computer's disk form. A put composes every form and a run only those its sandbox's deploys
boot, and a deploy composes first what it boots that no compose started. A deploy that names the disk form boots it in
a VM, its settings in its seed, and a reset boots it afresh. Docker and the registry are faked."""

import asyncio
import base64
import json
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentenv_protocol import CONTROL_TOKEN_ENV

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env import Env
from agent_env.config.runtime import Config
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils import boot, composer, probing
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.published_images import DISK
from agent_env import providers
from agent_env.providers.env_providers import env_computer_provider
from agent_env.providers.env_providers.env_provider import DeployRefused
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.utils.progress import run_progress
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from tst.unit.env.envs._computer import DISK_FORM, _base, _put, disk_layer
from tst.unit.env.envs.test_computer_env import DRIVER_CARD

BOOT_DOCKERFILE = (DISK / "boot" / "Dockerfile").read_text()


def test_each_arch_has_a_disk_made_from_its_container_on_a_base_disk_made_from_the_bases_container(disk_base, driver,
                                                                                                  docker):
    env = Env.get(_put(disk_base, driver).id)

    containers, disks = env.metadata["composed"]["container"], env.metadata["composed"]["disk"]
    assert {arch: r["status"] for arch, r in disks.items()} == {"amd64": "unverified", "arm64": "unverified"}
    assert disks["arm64"]["reason"] == "no link of the sandbox provider boots linux/arm64 disks natively to probe it"
    assert all(r["image"].startswith("localhost:5000/agentenv-computers:") for r in disks.values())
    built = {build["tag"].split(":")[1].split("-", 1)[1]: build for build in docker.disk_builds
             if build["context"] != "tools"}
    for arch in ("amd64", "arm64"):
        made, disk = built[f"base-disk-{arch}"], built[f"disk-{arch}"]
        assert made["dockerfile"] == disk["dockerfile"] == BOOT_DOCKERFILE and made["platform"] == f"linux/{arch}"
        assert made["args"]["ROOTFS"] == disk_base.image().image_name and made["args"]["SIZE_GB"] == "16"
        assert "ESP" not in made["args"] and made["args"]["TOOLS"] == disk["args"]["TOOLS"]
        assert (disk["args"]["ROOTFS"], disk["args"]["ESP"]) == (containers[arch]["image"], made["tag"])
        assert json.loads(base64.b64decode(disk["args"]["ROOTFS_CONFIG"])) == {"Labels": docker.labels[
            f"agentenv-computer:{containers[arch]['key'][7:23]}-{arch}"]}
        assert disk["labels"] == {composer.KEY_LABEL: disks[arch]["key"]}
    tools = [build for build in docker.disk_builds if build["context"] == "tools"]
    assert {(build["platform"], build["tag"]) for build in tools} == {("linux/arm64", built["disk-amd64"]["args"]["TOOLS"]),
                                                                      ("linux/arm64", built["disk-arm64"]["args"]["TOOLS"])}
    assert tools[0]["dockerfile"] == (DISK / "tools" / "Dockerfile").read_text()
    computer = env.computer_artifact()
    assert computer.forms.disk.model_dump(exclude={"image"}) == {**DISK_FORM, "firmware": ["efi"]}
    assert computer.disk_image().image_name.startswith("localhost:5000/agentenv-computers:index-")
    assert {arch: r["status"] for arch, r in computer.metadata["validated_images"]["disk"].items()} == {
        "amd64": "passed", "arm64": "passed"}
    assert env.metadata["computer"]["disks"] == {arch: r["image"] for arch, r in disks.items()}
    assert composer.current(env) and composer.pending(env) == []


def test_a_disk_is_reused_by_its_key_of_its_containers_key_the_base_disk_and_the_recipe(disk_base, driver, docker,
                                                                                       monkeypatch):
    first = Env.get(_put(disk_base, driver).id).metadata["composed"]["disk"]
    made = len(docker.disk_builds)
    other = Env.get(_put(disk_base, driver, id="@local/~/kit/other", environment_name="other").id)
    after_other = len(docker.disk_builds)
    monkeypatch.setattr(composer, "FORMAT", composer.FORMAT + 1)
    raised = Env.get(_put(disk_base, driver, id="@local/~/kit/third", environment_name="third").id)

    reused = other.metadata["composed"]["disk"]
    assert {arch: (r["key"], r["image"], r["reused"]) for arch, r in reused.items()} == {
        arch: (r["key"], r["image"], True) for arch, r in first.items()} and after_other == made
    assert all(raised.metadata["composed"]["disk"][arch]["key"] != first[arch]["key"] for arch in first)
    assert len(docker.disk_builds) > made


def test_a_bases_disk_size_is_in_the_key_and_the_build_of_the_disk_made_for_it(disk_base, driver, docker):
    bigger = _base(disk_base.image(), id="@local/~/kit/bigger", forms={
        **disk_base.forms.model_dump(exclude_none=True), "disk": {**DISK_FORM, "min_disk_gb": 32}})

    small = Env.get(_put(disk_base, driver).id).metadata["composed"]["disk"]["arm64"]
    large = Env.get(_put(bigger, driver, id="@local/~/kit/other", environment_name="other").id)

    assert large.metadata["composed"]["disk"]["arm64"]["key"] != small["key"]
    assert sorted(build["args"]["SIZE_GB"] for build in docker.disk_builds if "base-disk-arm64" in build["tag"]) == [
        "16", "32"]


def test_a_base_with_a_disk_image_or_a_published_one_keeps_its_partition_and_makes_none(disk_base, driver, docker,
                                                                                       monkeypatch):
    registry = docker.registry
    disks = registry.index(amd64=registry.image(disk_layer()), arm64=registry.image(disk_layer(), arch="arm64"))
    image = get_artifact_store().put_document(DockerImageArtifact(id="base-disk", description="",
                                                                  image_name=f"ghcr.io/acme/base-disk:1@{disks}"))
    with_disk = _base(disk_base.image(), id="@local/~/kit/with-disk", forms={
        **disk_base.forms.model_dump(exclude_none=True), "disk": {**DISK_FORM, "image": {"id": image.id,
                                                                                        "version": image.version}}})
    monkeypatch.setattr(composer, "published_disk", lambda sources: "ghcr.io/scaleapi/base-disk:src-1@sha256:" + "a" * 64)
    asked = []
    monkeypatch.setattr(LocalRegistryImageStore, "auth", lambda self, ref: asked.append(ref))

    own = Env.get(_put(with_disk, driver).id).metadata["composed"]["disk"]
    published = Env.get(_put(disk_base, driver, id="@local/~/kit/other", environment_name="other").id)

    esps = {build["args"]["ESP"] for build in docker.disk_builds if "ESP" in (build["args"] or {})}
    assert esps == {image.image_name, "ghcr.io/scaleapi/base-disk:src-1@sha256:" + "a" * 64}
    assert not [build for build in docker.disk_builds if "base-disk" in build["tag"]]
    assert {arch: r["status"] for arch, r in own.items()} == {"amd64": "unverified", "arm64": "unverified"}
    assert own["arm64"]["key"] != published.metadata["composed"]["disk"]["arm64"]["key"]
    assert {image.image_name, "ghcr.io/scaleapi/base-disk:src-1@sha256:" + "a" * 64} <= set(asked)


def test_a_disk_isnt_made_for_a_container_that_didnt_build_and_is_made_by_the_next_build(disk_base, driver, docker):
    docker.fail["linux/amd64"] = "boom"
    env = Env.get(_put(disk_base, driver).id)
    disks, computer = env.metadata["composed"]["disk"], env.computer_artifact()
    docker.fail.clear()

    composer.compose(env, composer.pending(env))

    assert (disks["amd64"]["status"], disks["amd64"]["reason"]) == ("skipped", "its container form didn't build (failed)")
    assert disks["arm64"]["status"] == "unverified" and computer.arch == ["arm64"] and computer.forms.disk is not None
    rebuilt = Env.get(env.id)
    assert rebuilt.metadata["composed"]["disk"]["amd64"]["status"] == "unverified" and composer.current(rebuilt)
    assert rebuilt.computer_artifact().arch == ["amd64", "arm64"]


def test_a_disk_is_probed_on_a_provider_that_boots_its_arch_natively(disk_base, driver, docker, monkeypatch):
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset({"linux/arm64"}))

    env = Env.get(_put(disk_base, driver).id)

    disks = env.metadata["composed"]["disk"]
    assert {arch: r["status"] for arch, r in disks.items()} == {"amd64": "unverified", "arm64": "ready"}
    assert ("disk", "arm64", disks["arm64"]["image"]) in docker.probed
    assert env.computer_artifact().metadata["validated_boot"]["disk"] == {"arm64": disks["arm64"]["probe"]}


def _link(containers=(), disks=(), here=True):
    """A sandbox provider that runs containers of ``containers`` and boots disks of ``disks`` natively, on this machine
    unless ``here`` is False."""
    link = SimpleNamespace(native_platforms=lambda: frozenset(f"linux/{arch}" for arch in containers),
                           disk_image_platforms=lambda: frozenset(f"linux/{arch}" for arch in disks), ON_THIS_MACHINE=here)
    link.links = (link,)
    return link


_SANDBOXES = {"local": [_link(["arm64"])], "local-with-kvm": [_link(["arm64"], ["arm64"])],
              "vm": [_link(disks=["amd64"])], "chain": [_link(["arm64"]), _link(disks=["amd64"])],
              "chain-of-containers": [_link(["arm64"]), _link(["amd64"])], "none-native": [_link(["riscv64"], ["riscv64"])],
              "vm-elsewhere": [_link(["amd64"], ["amd64"], here=False)]}


@pytest.mark.parametrize("sandbox, form, forms", [
    ("local", None, ["container"]), ("local-with-kvm", None, ["container"]), ("local-with-kvm", "disk", ["disk"]),
    ("vm", None, ["disk"]), ("chain", None, ["container", "disk"]), ("chain-of-containers", None, ["container"]),
    ("none-native", None, []), ("vm-elsewhere", None, ["disk"]),
])
def test_the_forms_composed_for_a_sandbox_are_those_its_deploys_boot(disk_base, driver, docker, sandbox, form, forms):
    """On each link the first form it boots natively, as a deploy picks, unless the deploy names one."""
    env = _put(disk_base, driver, compose=False)

    assert composer.booted_forms(env, SimpleNamespace(links=tuple(_SANDBOXES[sandbox])), form) == forms


@pytest.mark.parametrize("sandbox", ["local", "vm", "chain", "vm-elsewhere"])
def test_what_a_run_composes_is_what_each_link_of_its_sandbox_then_boots(disk_base, driver, docker, sandbox):
    """Whatever the deploy's link, its pick of the run's builds is of a form the run composed, and each form the run
    composed is one a link picks: the two share ``boot.first_native``."""
    env = _put(disk_base, driver, compose=False)
    links = _SANDBOXES[sandbox]
    sandbox = SimpleNamespace(links=tuple(links))
    composed = composer.pending(env, forms=Composing.for_run(sandbox).forms(env, sandbox))
    with_containers = composer._with_containers(env, composed)

    picked = {boot.first_native(link, with_containers, env.stored_id) for link in links}

    assert None not in picked and {form for form, _ in picked} == {form for form, _ in composed}


def test_a_runs_compose_looks_up_the_default_sandbox_once_and_only_when_a_computer_env_needs_it(disk_base, driver,
                                                                                                docker, monkeypatch):
    """So a run of a bundle with no computer env never builds a default provider it may not load."""
    looked_up = []
    monkeypatch.setattr(providers, "get_env_sandbox_provider", lambda: looked_up.append(1) or (
        local_sandbox.LocalSandboxProvider()))
    composing = Composing.for_run(None)
    assert looked_up == []
    env = _put(disk_base, driver, compose=False)

    assert composing.forms(env, composing.provider()) == ["container"] and looked_up == [1]


def test_a_compose_of_disks_alone_builds_the_containers_they_are_made_of_rather_than_skip_them(disk_base, driver,
                                                                                               docker):
    env = Env.get(_put(disk_base, driver, compose=False).id)

    composer.compose(env, [("disk", "arm64")])

    composed = Env.get(env.id).metadata["composed"]
    assert composed["container"]["arm64"]["status"] == "ready" and composed["disk"]["arm64"]["status"] == "unverified"
    assert set(composed["container"]) == set(composed["disk"]) == {"arm64"}


def test_a_put_composes_every_form_and_a_run_only_those_its_sandboxs_deploys_boot(disk_base, driver, docker):
    put = Env.get(_put(disk_base, driver).id)
    run = Env.get(_put(disk_base, driver, id="@local/~/kit/run", environment_name="run",
                       composing=Composing.for_run(None)).id)

    assert set(put.metadata["composed"]) == {"container", "disk"} and put.computer_artifact().forms.disk is not None
    assert set(run.metadata["composed"]) == {"container"} and run.computer_artifact().forms.disk is None
    composing = Composing.for_run(None)
    assert composer.pending(run, forms=composing.forms(run, composing.provider())) == [] and composer.current(run)


@pytest.fixture
def vms(monkeypatch):
    """The VMs a deploy's boots of a disk ask the local provider for, each serving the desktop driver's card."""
    made = SimpleNamespace(created=[], reset=[])

    async def create_vm(self, **kwargs):
        made.created.append(kwargs)
        sandbox = local_sandbox.LocalSandbox(port_map={18765: 20001}, sandbox_id=f"local-vm{len(made.created)}")
        sandbox.terminate = AsyncMock()
        return sandbox

    async def reset_vm(self, sandbox, **kwargs):
        made.reset.append((sandbox.sandbox_id, kwargs))

    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "create_vm", create_vm)
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "reset_vm", reset_vm)
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset({"linux/arm64"}))
    monkeypatch.setattr(boot, "ready", AsyncMock(side_effect=lambda url, timeout, token, answering=False: {
        **DRIVER_CARD, "name": "desktop"}))
    monkeypatch.setattr(env_computer_provider, "ready", boot.ready)
    monkeypatch.setattr(env_computer_provider, "_require_display", AsyncMock())
    return made


@pytest.mark.asyncio
async def test_a_deploy_of_the_disk_form_boots_it_in_a_vm_with_its_settings_and_token_in_its_seed(disk_base, driver,
                                                                                                  docker, vms):
    env = _put(disk_base, driver, display={"width": 1600, "height": 900})

    deployed = await Env.get(env.id).deploy(env_params={"form": "disk", "computer_settings": {"timezone": "UTC"}})

    (created,) = vms.created
    disk = Env.get(env.id).metadata["composed"]["disk"]["arm64"]
    assert (created["image"], created["exposed_ports"], created["cpu"], created["memory"]) == (
        disk["image"], [18765], boot.DISK_CPU, boot.DISK_MEMORY_MB)
    assert created["env"]["AGENTENV_SETTING_RESOLUTION"] == "1600x900" and created["env"]["ENVIRONMENT_NAME"] == "desktop"
    assert created["env"][CONTROL_TOKEN_ENV] == deployed.control_token
    assert (deployed.metadata["computer"]["form"], deployed.metadata["computer"]["image"]) == ("disk", disk["image"])
    assert deployed.environment_url == "http://127.0.0.1:20001"


@pytest.mark.asyncio
async def test_an_arch_whose_disk_built_deploys_it_though_the_computer_has_no_disk_form_without_the_other(
        disk_base, driver, docker, vms, monkeypatch):
    built = composer._built_disk
    monkeypatch.setattr(composer, "_built_disk", lambda env, base, arch, *rest: (
        {"status": "failed", "stage": "build", "reason": "boom"} if arch == "amd64" else built(env, base, arch, *rest)))
    env = Env.get(_put(disk_base, driver).id)

    deployed = await env.deploy(env_params={"form": "disk"})

    assert env.computer_artifact().forms.disk is None and env.metadata["composed"]["disk"]["arm64"]["status"] == "ready"
    assert deployed.metadata["computer"]["form"] == "disk" and len(vms.created) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [None, "probe", "boot"])
async def test_first_deploys_of_an_unverified_disk_at_once_share_one_probe_and_its_recorded_result(
        disk_base, driver, docker, vms, monkeypatch, fails):
    """Whether its probe passes, fails, or never boots it, whose record keeps no image."""
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset())
    env = _put(disk_base, driver)
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset({"linux/arm64"}))
    probe, first_boot, waited = docker.probe, env_computer_provider.probe_first_boot, []

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.3)
        if fails == "boot":
            raise RuntimeError("LocalSandboxProvider: no boot")
        return await probe(*args, **kwargs)

    async def counted(*args):
        waited.append(args[1:3])
        return await first_boot(*args)

    monkeypatch.setattr(probing, "_probe", slow)
    monkeypatch.setattr(env_computer_provider, "probe_first_boot", counted)
    if fails == "probe":
        docker.probe_fails["arm64"] = "the click never landed"

    deployed = await asyncio.gather(*(Env.get(env.id).deploy(env_params={"form": "disk"}) for _ in range(3)),
                                    return_exceptions=True)

    assert waited == [("disk", "arm64")] * 3
    assert len([probed for probed in docker.probed if probed[0] == "disk"]) == (fails != "boot")
    recorded = Env.get(env.id).metadata["composed"]["disk"]["arm64"]
    if fails:
        assert recorded["status"] == "failed" and vms.created == []
        assert recorded["reason"] == ("it didn't boot: LocalSandboxProvider: no boot" if fails == "boot" else
                                      "the probe failed: P4 clicks show their effect: the click never landed")
        assert {str(failure) for failure in deployed} == {
            f"{env.id}'s disk/arm64 failed its probe on its first boot: {recorded['reason']}"}
    else:
        assert recorded["status"] == "ready" and len(vms.created) == 3
        assert all(record.metadata["computer"]["form"] == "disk" for record in deployed)


@pytest.mark.asyncio
async def test_a_first_boot_whose_record_another_compose_took_over_probes_the_build_it_picked_and_records_none(
        disk_base, driver, docker, vms, monkeypatch):
    """As a deploy by an agent-env that composes differently does, starting its own compose of the pair."""
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset())
    env = _put(disk_base, driver)
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset({"linux/arm64"}))
    picked, stored = Env.get(env.id), Env.get(env.id)
    taken = {"status": "building", "started_at": "2026-11-05T16:00:00+00:00", "composer": "sha256:another-agent-env"}
    composed = stored.metadata["composed"]
    stored.update_metadata({**stored.metadata, "composed": {**composed, "disk": {**composed["disk"], "arm64": taken}}})

    result = await probing.probe_first_boot(picked, "disk", "arm64", local_sandbox.LocalSandboxProvider())

    assert (result["status"], result["image"]) == ("ready", picked.metadata["composed"]["disk"]["arm64"]["image"])
    assert Env.get(env.id).metadata["composed"]["disk"]["arm64"] == taken


@pytest.mark.asyncio
async def test_deploys_of_a_disk_no_compose_started_compose_it_once_first_reusing_the_containers(disk_base, driver,
                                                                                                docker, vms):
    """As a run that composed for local Docker, which boots containers, leaves an env a later deploy boots as a disk."""
    env = _put(disk_base, driver, composing=Composing.for_run(None))
    containers, built = Env.get(env.id).metadata["composed"]["container"], len(docker.builds)

    deployed = await asyncio.gather(*(Env.get(env.id).deploy(env_params={"form": "disk"}) for _ in range(3)))

    stored = Env.get(env.id)
    disks = stored.metadata["composed"]["disk"]
    assert {arch: r["status"] for arch, r in disks.items()} == {"amd64": "unverified", "arm64": "ready"}
    assert stored.metadata["composed"]["container"] == containers and len(docker.builds) == built
    assert sorted(build["tag"].split(":")[1].split("-", 1)[1] for build in docker.disk_builds
                  if build["context"] != "tools") == ["base-disk-amd64", "base-disk-arm64", "disk-amd64", "disk-arm64"]
    assert stored.computer_artifact().forms.disk is not None and len(vms.created) == 3
    assert all(record.metadata["computer"]["image"] == disks["arm64"]["image"] for record in deployed)


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", [True, False], ids=["shared-store", "per-user-store"])
async def test_a_deploy_composes_its_missing_form_unless_the_runs_fence_would_refuse_the_record(disk_base, driver,
                                                                                                docker, vms,
                                                                                                monkeypatch, refused):
    """Where a configured store isn't the per-user one, a local run refuses writing a registered env's record."""
    env = _put(disk_base, driver, composing=Composing.for_run(None))
    builds, asked = len(docker.disk_builds), []
    monkeypatch.setattr(Config, "local_run_refuses_write", lambda self, entity_id: asked.append(entity_id) or refused)

    if refused:
        with pytest.raises(DeployRefused, match=rf"has no ready build to deploy.*agent-env env build --id {env.id}"):
            await Env.get(env.id).deploy(env_params={"form": "disk"})
    else:
        await Env.get(env.id).deploy(env_params={"form": "disk"})

    assert asked[0] == env.id
    assert ("disk" in Env.get(env.id).metadata["composed"]) is (len(docker.disk_builds) > builds) is (not refused)
    assert len(vms.created) == (not refused)


@pytest.mark.asyncio
async def test_a_deploy_that_finds_the_disk_building_for_another_deploy_waits_for_it_and_boots_it(disk_base, driver,
                                                                                                 docker, vms,
                                                                                                 monkeypatch):
    """Saying what it waits for, from when, and then that it reuses what the other built."""
    env = _put(disk_base, driver, composing=Composing.for_run(None))
    started, waited, said = threading.Event(), threading.Event(), []
    record_started, built_disk = composer.Record.started, composer._built_disk

    def recorded(self, pairs):
        record_started(self, pairs)
        started.set()

    def held(*args):
        waited.wait(10)
        return built_disk(*args)

    def told(line):
        said.append(line)
        waited.set()

    monkeypatch.setattr(composer.Record, "started", recorded)
    monkeypatch.setattr(composer, "_built_disk", held)

    async def later():
        await asyncio.to_thread(started.wait, 10)
        second = Env.get(env.id)
        assert second.metadata["composed"]["disk"]["arm64"]["status"] == "building"
        with run_progress(told):
            return await second.deploy(env_params={"form": "disk"})

    first, second = await asyncio.gather(Env.get(env.id).deploy(env_params={"form": "disk"}), later())

    assert first.metadata["computer"]["form"] == second.metadata["computer"]["form"] == "disk" and len(vms.created) == 2
    since = datetime.fromisoformat(Env.get(env.id).metadata["composed"]["disk"]["arm64"]["started_at"]).strftime("%H:%M")
    assert said == [f"{env.id}: waiting for another deploy composing disk/amd64, disk/arm64 (building since {since} UTC)",
                    f"{env.id}: reusing what another deploy composed: disk/amd64 unverified, disk/arm64 ready"]


@pytest.mark.asyncio
async def test_a_cancelled_deploy_holds_the_lock_until_its_compose_ends_so_the_next_deploy_waits_and_boots_it(
        disk_base, driver, docker, vms, monkeypatch):
    env = _put(disk_base, driver, composing=Composing.for_run(None))
    started, waited, said = threading.Event(), threading.Event(), []
    record_started, built_disk = composer.Record.started, composer._built_disk

    def recorded(self, pairs):
        record_started(self, pairs)
        started.set()

    def held(*args):
        waited.wait(10)
        return built_disk(*args)

    def told(line):
        said.append(line)
        waited.set()

    monkeypatch.setattr(composer.Record, "started", recorded)
    monkeypatch.setattr(composer, "_built_disk", held)
    first = asyncio.create_task(Env.get(env.id).deploy(env_params={"form": "disk"}))
    await asyncio.to_thread(started.wait, 10)
    first.cancel()
    with run_progress(told):
        second = await Env.get(env.id).deploy(env_params={"form": "disk"})

    with pytest.raises(asyncio.CancelledError):
        await first
    assert second.metadata["computer"]["form"] == "disk" and len(vms.created) == 1
    assert said[0].startswith(f"{env.id}: waiting for another deploy composing disk/amd64, disk/arm64")
    assert said[-1] == f"{env.id}: reusing what another deploy composed: disk/amd64 unverified, disk/arm64 ready"


@pytest.mark.asyncio
async def test_a_deploy_that_composes_first_tells_the_run_why_and_each_stage_as_it_goes(disk_base, driver, docker,
                                                                                       vms):
    """Through the task's scheduling and the threads the compose runs in, to the run's progress."""
    env = _put(disk_base, driver, composing=Composing.for_run(None))
    said = []
    task = Task(id="@local/~/kit/t", version=None, steps=[
        DeployEnvTaskStep(id="env", version=None, env_id=env.id, env_params={"form": "disk"})])

    with run_progress(said.append):
        context = await task.run(context=TaskStepContext())

    assert context.deployed_envs[0].metadata["computer"]["form"] == "disk"
    assert said[:2] == [f"{env.id}: composing disk/amd64, disk/arm64 before this deploy on LocalSandboxProvider (no "
                        "compose built them for LocalSandboxProvider yet); later deploys reuse them",
                        f"{env.id}: composing disk/amd64, disk/arm64 on LocalSandboxProvider"]
    assert any(line.startswith(f"{env.id}: disk/arm64: ready (") for line in said)
    assert any(line.startswith(f"{env.id}: disk/amd64: making the disk") for line in said)


@pytest.mark.asyncio
async def test_a_deploy_on_a_sandbox_that_boots_only_disks_composes_them_on_this_machine_and_says_so(disk_base, driver,
                                                                                                    docker,
                                                                                                    monkeypatch):
    """Never on the deploy's own sandbox: the env's default provider, this machine's Docker, builds them."""
    env = _put(disk_base, driver, compose=False)
    composed, said = [], []
    monkeypatch.setattr(composer, "compose", lambda env, pairs, composing: composed.append((pairs, composing)))

    with run_progress(said.append):
        await Env.get(env.id).compose_unstarted(_link(disks=["amd64"]))

    ((pairs, composing),) = composed
    assert pairs == [("disk", "amd64"), ("disk", "arm64")]
    assert isinstance(composing.provider(), local_sandbox.LocalSandboxProvider)
    assert said == [f"{env.id}: composing disk/amd64, disk/arm64 before this deploy on LocalSandboxProvider (no compose "
                    "built them for SimpleNamespace yet); later deploys reuse them"]


@pytest.mark.asyncio
async def test_a_deploy_of_a_disk_bigger_than_the_disk_is_refused_rather_than_booted_smaller(disk_base, driver, docker,
                                                                                            vms):
    env = _put(disk_base, driver)

    with pytest.raises(Exception, match="disk is 16 GB, which a deploy doesn't enlarge, and it asks for 64 GB"):
        await Env.get(env.id).deploy(disk_size_gb=64, env_params={"form": "disk"})

    assert vms.created == []


@pytest.mark.asyncio
async def test_an_unverified_disk_is_probed_on_its_first_boot_with_kvm_before_its_deploy_boots_it(disk_base, driver,
                                                                                                  docker, vms,
                                                                                                  monkeypatch):
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset())
    env = _put(disk_base, driver)
    failing = _put(disk_base, driver, id="@local/~/kit/failing", environment_name="failing")
    monkeypatch.setattr(local_sandbox, "kvm_platforms", lambda: frozenset({"linux/arm64"}))

    await Env.get(env.id).deploy(env_params={"form": "disk"})
    docker.probe_fails["arm64"] = "the click never landed"
    with pytest.raises(RuntimeError, match="failed its probe on its first boot: the probe failed: P4"):
        await Env.get(failing.id).deploy(env_params={"form": "disk"})

    assert Env.get(env.id).metadata["composed"]["disk"]["arm64"]["status"] == "ready"
    assert [probe[0] for probe in docker.probed if probe[0] == "disk"] == ["disk", "disk"]
    assert Env.get(failing.id).metadata["composed"]["disk"]["arm64"]["status"] == "failed"
    assert len(vms.created) == 1


def test_a_sandbox_elsewhere_boots_the_disk_it_carries_not_a_container_in_this_machines_registry(base, disk_base,
                                                                                                 driver, docker,
                                                                                                 monkeypatch):
    """The container form is in the image store's registry on this machine, which such a sandbox can't pull from; the
    disk it carries itself, as a Modal VM does, so a deploy there passes its preflight with the disk form."""
    elsewhere = _link(["amd64"], ["amd64"], here=False)
    monkeypatch.setattr(providers, "build_sandbox_provider", lambda spec: elsewhere)
    with_disk = Env.get(_put(disk_base, driver, compose=False).id)
    containers = Env.get(_put(base, driver, compose=False, id="@local/~/kit/containers", environment_name="c").id)

    assert composer.booted_forms(with_disk, elsewhere) == ["disk"] and with_disk.deploy_refusal(sandbox_type="x") is None
    assert containers.deploy_refusal(sandbox_type="x") == (
        f"computer env {containers.id!r}: SimpleNamespace runs none of its arches (amd64, arm64) natively as container, "
        "and SimpleNamespace can't pull its containers from the image store's registry on this machine")


def test_a_disk_form_is_refused_at_save_where_no_provider_boots_its_disk_natively(disk_base, driver, docker):
    env = Env.get(_put(disk_base, driver).id)

    assert env.deploy_refusal(env_params={"form": "disk"}) == (
        f"computer env {env.id!r}: LocalSandboxProvider runs none of its arches (amd64, arm64) natively as disk")
    assert env.deploy_refusal() is None


@pytest.mark.asyncio
async def test_a_reset_of_a_disk_boots_it_afresh_in_its_vm_with_the_records_settings_and_token(disk_base, driver,
                                                                                             docker, vms, monkeypatch):
    env = _put(disk_base, driver, display={"width": 1600, "height": 900})
    deployed = await Env.get(env.id).deploy(env_params={"form": "disk"})
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(
        return_value=local_sandbox.LocalSandbox(port_map={18765: 20001}, sandbox_id="local-vm1")))
    monkeypatch.setattr(env_computer_provider, "end_recordings", AsyncMock())

    await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(
        TaskStepContext(deployed_envs=[deployed]))

    ((sandbox_id, kwargs),) = vms.reset
    assert (sandbox_id, kwargs["image"]) == ("local-vm1", deployed.metadata["computer"]["image"])
    assert kwargs["env"][CONTROL_TOKEN_ENV] == deployed.control_token
    assert kwargs["env"]["AGENTENV_SETTING_RESOLUTION"] == "1600x900"



@pytest.mark.asyncio
@pytest.mark.parametrize("asks", [1024, None])
async def test_a_disks_init_is_told_the_shared_memory_its_apps_ask_for_at_boot_and_at_a_reset(disk_base, driver, docker,
                                                                                            vms, monkeypatch, asks):
    monkeypatch.setattr(ComputerEnv, "shm_mb", lambda self: asks)
    env = _put(disk_base, driver, display={"width": 1600, "height": 900})
    deployed = await Env.get(env.id).deploy(env_params={"form": "disk"})
    monkeypatch.setattr(local_sandbox.LocalSandboxProvider, "get_sandbox", AsyncMock(
        return_value=local_sandbox.LocalSandbox(port_map={18765: 20001}, sandbox_id="local-vm1")))
    monkeypatch.setattr(env_computer_provider, "end_recordings", AsyncMock())

    await ResetEnvTaskStep(id="reset", version=None, env_id=env.id, fail_task_on_error=True).execute(
        TaskStepContext(deployed_envs=[deployed]))

    ((created,), ((_, reset),)) = vms.created, vms.reset
    told = {boot.SHM_MB_ENV: str(asks)} if asks else {}
    assert {key: created["env"][key] for key in created["env"] if key == boot.SHM_MB_ENV} == told
    assert {key: reset["env"][key] for key in reset["env"] if key == boot.SHM_MB_ENV} == told
    assert "shm_mb" not in created and deployed.metadata["computer"].get("shm_mb") == asks

def test_a_computer_artifact_takes_a_disk_form_and_reads_its_image(base, docker):
    registry = docker.registry
    good = registry.index(amd64=registry.image(disk_layer()), arm64=registry.image(disk_layer(), arch="arm64"))
    bad = registry.index(amd64=registry.image(disk_layer(virtual_size=8 << 30, owner=0)),
                         arm64=registry.image(disk_layer(path="disk/other.img"), arch="arm64"))
    images = {name: get_artifact_store().put_document(DockerImageArtifact(
        id=name, description="", image_name=f"ghcr.io/acme/{name}:1@{index}")) for name, index in (("good", good),
                                                                                                    ("bad", bad))}

    def put(id, disk):
        return ComputerArtifact.put(**{**base.model_dump(include={"arch", "os", "session", "config"}), "id": id,
                                       "forms": {**base.forms.model_dump(exclude_none=True), "disk": {**DISK_FORM, **disk}}})

    derived = put("@local/~/kit/derived", {})
    checked = put("@local/~/kit/good", {"image": {"id": "good", "version": images["good"].version}})
    with pytest.raises(ValueError) as refused:
        put("@local/~/kit/bad", {"image": {"id": "bad", "version": images["bad"].version}})

    assert derived.forms.disk.image is None and derived.disk_image() is None
    assert {r["status"] for r in derived.metadata["validated_images"]["disk"].values()} == {"uninspected"}
    assert {arch: r["status"] for arch, r in checked.metadata["validated_images"]["disk"].items()} == {
        "amd64": "passed", "arm64": "passed"}
    assert str(refused.value) == (
        "computer '@local/~/kit/bad': forms.disk (amd64): disk/disk.img is owned by uid 0, not 107; its virtual size is "
        "8.0 GiB, under min_disk_gb (16); forms.disk (arm64): its first file is disk/other.img, not disk/disk.img")
    assert ComputerArtifact.get(checked.id).forms.disk.image.id == "good"
