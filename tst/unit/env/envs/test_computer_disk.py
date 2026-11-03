"""Computer envs of a base with a disk form: each arch's disk is made from its built container, on the base's disk or on
one made from the base's container and the boot recipe, by a build that runs on the builder's own platform; reused by
its key; probed where a provider boots disks natively, else left unverified for a deploy's first boot with KVM to
probe; and written as the computer's disk form. A deploy that names the disk form boots it in a VM, its settings in its
seed, and a reset boots it afresh. Docker and the registry are faked."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentenv_protocol import CONTROL_TOKEN_ENV

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env import Env
from agent_env.env.envs.computer.utils import boot, composer
from agent_env.env.published_images import DISK
from agent_env.providers.env_providers import env_computer_provider
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.task_step.context import TaskStepContext
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
    monkeypatch.setattr(boot, "ready", AsyncMock(side_effect=lambda url, timeout, token: {
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
