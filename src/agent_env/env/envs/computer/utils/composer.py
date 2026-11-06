"""The composer: a computer env's base, driver and apps built into one image for each form and arch, on a builder the
put's sandbox provider gives (``builder``), tagged in the env's image store by a key of everything that goes into it, so
an image of that key, built for any env and on any builder, is reused. A disk form is made from its arch's container
once that has built: its root filesystem as the disk's, on the base disk's EFI system partition, by a build that runs
only on the builder's own platform. A VM bundle, macOS's, is composed on a Mac: its base's booted, given each layer
over ssh and kept on that Mac by a name of its key (``vm_bundle``).
Each result is recorded in the env's metadata as it finishes (``composed``), and the arches that built are written as
the env's computer, a ``computer`` artifact over an index of their images."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal

from agentenv_protocol.computer import apps as installed_apps
from agentenv_protocol.computer.app import BUILD_SETTINGS, SOCKET

from agent_env.artifact.artifacts.computer import LOCAL_IMAGE_PREFIX, ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.config import get_config
from agent_env.entity.export import Unwritable, checked_path
from agent_env.env.envs.computer.utils import boot, builder, clashes, gate, macos_vm_bundle, packages, probing
from agent_env.env.envs.computer.utils.composed import BUILT, REPOSITORY, Record, failed, now, stale
from agent_env.env.envs.computer.utils.install import Install
from agent_env.env.published_images import DISK, DISK_TOOLS_IMAGE, published_disk, published_pinned
from agent_env.providers.sandbox_providers.host_images import platform_tag
from agent_env.store.ids import derive_id
from agent_env.store.image_store.image_contents import image_config, labels
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.store.image_store.registry_api import NoSuchImageError, pin_digest, write_index
from agent_env.store.local_state import holding_locks
from agent_env.utils.docker.build import DockerBuildError

if TYPE_CHECKING:

    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.env.envs.computer.app_env import ComputerAppEnv
    from agent_env.env.envs.computer.driver_env import ComputerDriverEnv

logger = logging.getLogger(__name__)

#: What the composer and its builders make of a build's inputs beyond what its key shows of them (the base, the
#: Dockerfile, the build files with their executable bits, the lock): raise it with a change that alters what an image
#: holds some other way, its labels or how it's built, and leave it for any other, so images aren't built again for
#: nothing.
FORMAT = 1
#: Where a driver's files go in a composed computer, in a folder named for the driver; an app's, under APPS.
DRIVERS = "/opt/agentenv/drivers"
APPS = str(installed_apps.APPS)
KEY_LABEL = "dev.agentenv.computer-key"
# What a build installed and whether it was emulated, kept on its image for an env that reuses it.
INSTALLED_LABEL, EMULATED_LABEL = "dev.agentenv.installed", "dev.agentenv.emulated"
#: How often a compose says a pair is still at a stage, which can take most of an hour under emulation.
HEARTBEAT_S = 600
_PROTOCOL = "agentenv-framework-protocol"
# What building the protocol from its source reads, beside src/.
_PROTOCOL_FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
#: How many of the files two parts wrote differently a progress line names.
_SHOWN = 10


@dataclass(frozen=True)
class Composing:
    """How a put composes a computer env: on ``sandbox_provider``, by default the env's; emulating an arch no link runs
    natively unless ``emulate`` is False; each pair's stages told to ``on_progress``; the forms of the base, or for a
    run (``for_run``) those its deploys boot (``forms_of``) on ``deploys_on``, by default the provider it composes on."""

    sandbox_provider: SandboxProvider | None = None
    emulate: bool = True
    on_progress: Callable[[str], None] | None = None
    forms_of: Literal["base", "deploys"] = "base"
    deploys_on: SandboxProvider | None = None

    @classmethod
    def for_run(cls, sandbox_provider: SandboxProvider | None, emulate: bool = True) -> Composing:
        """How a run whose deploys are on ``sandbox_provider``, by default the env's, composes: on the env's, this
        machine unless the config names another, the forms its deploys boot."""
        return cls(None, emulate, forms_of="deploys", deploys_on=sandbox_provider)

    def provider(self) -> SandboxProvider:
        """``sandbox_provider``, else the env's default, looked up by each compose that needs it, once: a run with no
        computer env never builds a default it may not load."""
        from agent_env.providers import get_env_sandbox_provider

        return self.sandbox_provider or get_env_sandbox_provider()

    def forms(self, env: ComputerEnv, provider: SandboxProvider) -> list[str]:
        """The forms of ``env``'s base this compose on ``provider``, its ``provider()``, builds."""
        return gate.forms(env.computer) if self.forms_of == "base" else booted_forms(env, self.deploys_on or provider)


def compose(env: ComputerEnv, pairs: list[tuple[str, str]] | None = None,
            composing: Composing = Composing()) -> dict[tuple[str, str], dict]:
    """Build each of ``pairs``, by default each of ``composing``'s forms in each arch the env composes, and the container
    each disk among them is made from unless it's built, in parallel, as ``composing`` says, and probe each one its
    provider runs natively (gate 3), recording each as it starts and finishes, then write the env's computer over every
    pair built so far. Each pair's stages and how it ended go to the log too (``_Progress``). What each pair came to,
    which the record keeps unless the one it holds got further, as a ready one does over a rebuild that failed."""
    base, driver, provider = env.computer, env.driver, composing.provider()
    builds = builder.Builds(provider, env.stored_id, REPOSITORY, composing.emulate)
    pairs = _with_containers(env, pairs if pairs is not None else [
        (form, arch) for form in composing.forms(env, provider) for arch in env.composed_arches()])
    record, results = Record(env), {}
    record.started(pairs)
    with _Progress(env, composing.on_progress) as progress, ThreadPoolExecutor(max_workers=max(len(pairs), 1)) as pool:
        progress.say(f"composing {', '.join(f'{form}/{arch}' for form, arch in pairs)} on "
                     f"{' or '.join(type(link).__name__ for link in provider.links)}")
        containers = {arch: pool.submit(_composed, env, base, driver, form, arch, builds, progress)
                      for form, arch in pairs if form == "container"}
        futures = {future: ("container", arch) for arch, future in containers.items()}
        futures |= {pool.submit(_composed_disk, env, base, driver, arch, containers.get(arch), builds, progress):
                    ("disk", arch) for form, arch in pairs if form == "disk"}
        futures |= {pool.submit(_composed_vm_bundle, env, base, driver, arch, provider, progress): ("vm_bundle", arch)
                    for form, arch in pairs if form == "vm_bundle"}
        for future in as_completed(futures):
            form, arch = futures[future]
            results[form, arch] = result = future.result()
            progress.ended(f"{form}/{arch}", result)
            record.finished(form, arch, result)
    record.computer(_computer(env, base, record.built(), record.built("disk"), record.built("vm_bundle")))
    return results


def current(env: ComputerEnv) -> bool:
    """Whether the env's computer was written over every pair built, as a put or env build that failed writing it
    leaves it not."""
    computer, record = env.metadata.get("computer"), Record(env)
    return computer is not None and computer.get("images") == record.built() and computer.get(
        "disks", {}) == record.built("disk") and computer.get("vm_bundles", {}) == record.built("vm_bundle")


def pending(env: ComputerEnv, sandbox_provider: SandboxProvider | None = None,
            forms: list[str] | None = None) -> list[tuple[str, str]]:
    """The pairs of ``forms``, by default every form of the base, left to compose in ``env``: never started, failed,
    left building by a put that stopped, recorded by an agent-env that composes differently, built but unprobed though
    ``sandbox_provider``, by default the env's, runs the arch natively, or a VM bundle the Mac it would compose on
    doesn't hold, which this agent-env composes and records there, as one that composes differently names it."""
    from agent_env.providers import get_env_sandbox_provider

    provider = sandbox_provider or get_env_sandbox_provider()

    def left(form: str, arch: str, record: dict | None) -> bool:
        status = (record or {}).get("status")
        if status not in BUILT or stale(record):
            return True
        if status == "unverified":
            return bool(boot.native_links(provider, arch, form, env.stored_id))
        if form != "vm_bundle":
            return False
        links = boot.native_links(provider, arch, form, env.stored_id)
        return bool(links) and not asyncio.run(links[0].has_vm_bundle(record["image"]))

    return _pairs(env, gate.forms(env.computer) if forms is None else forms, left)


def unstarted(env: ComputerEnv, sandbox_provider: SandboxProvider, form: str | None = None) -> list[tuple[str, str]]:
    """The pairs a deploy on ``sandbox_provider`` boots that no compose has started, as a run for another sandbox
    leaves them, or that an agent-env that composes differently recorded."""
    return _pairs(env, booted_forms(env, sandbox_provider, form),
                  lambda form, arch, record: record is None or stale(record))


def unstarted_or_building(env: ComputerEnv, sandbox_provider: SandboxProvider,
                          form: str | None = None) -> list[tuple[str, str]]:
    """Those, and the ones a compose is building now, which a deploy waits for."""
    return _pairs(env, booted_forms(env, sandbox_provider, form),
                  lambda form, arch, record: record is None or stale(record) or record.get("status") == "building")


def booted_forms(env: ComputerEnv, sandbox_provider: SandboxProvider, form: str | None = None) -> list[str]:
    """The forms deploys on ``sandbox_provider`` boot, ``form``'s when a deploy names it, in the base's order: on each
    link, the one ``boot.first_native`` picks, as the deploy does."""
    forms = gate.forms(env.computer)
    booted = {picked[0] for link in sandbox_provider.links if (picked := boot.first_native(
        link, [(name, arch) for name in ([form] if form else forms) for arch in env.composed_arches()], env.stored_id))}
    return [name for name in forms if name in booted]


def _pairs(env: ComputerEnv, forms: list[str], left: Callable[[str, str, dict | None], bool]) -> list[tuple[str, str]]:
    """The pairs of ``forms`` in the arches the env composes whose record, None when there's none, is ``left``."""
    composed = env.metadata.get("composed", {})
    return [(form, arch) for form in forms for arch in env.composed_arches()
            if left(form, arch, composed.get(form, {}).get(arch))]


def _with_containers(env: ComputerEnv, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``pairs`` and the container each disk among them is made from, unless it's built by this agent-env's composer, in
    the base's order."""
    containers = env.metadata.get("composed", {}).get("container", {})
    wanted = {*pairs, *(("container", arch) for form, arch in pairs if form == "disk"
                        and (containers.get(arch, {}).get("status") not in BUILT or stale(containers.get(arch))))}
    return [(form, arch) for form in gate.forms(env.computer) for arch in env.composed_arches() if (form, arch) in wanted]


def _composed(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, form: str, arch: str,
              builds: builder.Builds, progress: _Progress) -> dict:
    """``form`` and ``arch`` built or reused, then probed when it built (gate 3)."""
    result = _built(env, base, driver, form, arch, builds, progress)
    if result["status"] != "unverified":
        return result
    return _probed(env, driver, form, arch, result, builds.provider, progress)


def _composed_disk(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, arch: str,
                   container: Future | None, builds: builder.Builds, progress: _Progress) -> dict:
    """The disk form of ``arch``, once its container, composed now or before, has built: built or reused, then probed
    when it built (gate 3)."""
    built = container.result() if container is not None else env.metadata.get("composed", {}).get(
        "container", {}).get(arch, {})
    if built.get("status") not in BUILT:
        return {"status": "skipped", "reason": f"its container form didn't build ({built.get('status', 'never')})",
                "skipped_at": now()}
    result = _built_disk(env, base, arch, built, builds, progress)
    if result["status"] != "unverified":
        return result
    return _probed(env, driver, "disk", arch, result, builds.provider, progress)


def held_or_composed_here(env: ComputerEnv, arch: str, link: SandboxProvider,
                          on_progress: Callable[[str], None] | None = None) -> None:
    """The VM bundle of ``arch`` the env's record names, on ``link``, the Mac a deploy boots it on: held there, or
    composed there first, from the same inputs and so under the same name, and probed at full (gate 3), since one
    composed on another Mac isn't checked by its probe. Looked for holding the lock its compose and probe hold, so a
    deploy waits for one under way and boots none before its probe passed. Refused, composing nothing, when this
    agent-env would compose it under another name."""
    recorded = env.metadata["composed"]["vm_bundle"][arch]["image"]
    with _Progress(env, on_progress) as progress, holding_locks([_bundle_lock(recorded)], lambda: progress.say(
            f"vm_bundle/{arch}: waiting for the compose or probe of {recorded} under way on this Mac")):
        if asyncio.run(link.has_vm_bundle(recorded)):
            return
        progress.say(f"vm_bundle/{arch}: {type(link).__name__} doesn't hold {recorded}; composing it there first")
        result = _composed_vm_bundle(env, env.computer, env.driver, arch, link, progress, held=recorded)
    if result["status"] != "ready":
        raise RuntimeError(f"{env.id}'s vm_bundle/{arch} couldn't be composed and probed on this Mac: "
                           f"{result.get('reason')}")


def _composed_vm_bundle(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, arch: str,
                        provider: SandboxProvider, progress: _Progress, held: str | None = None) -> dict:
    """The VM bundle of ``arch`` composed on the first link of ``provider`` that boots macOS VM bundles: its base's
    cloned and booted with a build folder of everything that goes into it (``macos_vm_bundle``), given each layer over
    ssh, shut down and kept on that Mac by a name of its key; or one of that key reused, kept there already. Then probed
    at full (gate 3), and deleted from that Mac when it fails, so no deploy there boots it, though a record of another
    of its name says ready. All of it holds a lock of its name on that Mac, which a deploy takes to look for it, and
    which keeps two composes of one name from breaking each other, as a builder of a name deletes an earlier one: the
    other waits, then reuses it. ``held`` names the VM bundle whose lock the caller holds, composed only under that
    name."""
    pair = f"vm_bundle/{arch}"
    with (_attempting(env, "vm_bundle", arch) as attempt,
          tempfile.TemporaryDirectory(prefix="agentenv-compose-") as work):
        if not (links := boot.native_links(provider, arch, "vm_bundle", env.stored_id)):
            raise builder.Unbuildable(f"no link of {' or '.join(type(link).__name__ for link in provider.links)} "
                                      f"boots macos/{arch} VM bundles, so it can't be composed here: compose it on a Mac "
                                      "with the lume provider")
        link, build = links[0], Path(work) / macos_vm_bundle.BUILD_FOLDER
        context = build / "context"
        context.mkdir(parents=True)
        protocol = _stage_protocol(context)
        layer = _stage_driver(context, driver, base, {})
        apps = [(app, _stage_app(context, app, base, {})) for app in env.composed_apps()]
        parts = [_part(layer, _services(f"driver-{gate.folder_name(driver.id, 'driver')}", layer.install),
                       {"MCP_PORT": driver.port}, protocol, False)]
        parts += [_part(staged, dict(zip(gate.programs(app, base.os.type), [
            f"venv/bin/python -m agentenv_protocol.computer.serve_app {staged.served}",
            *(service.command for service in staged.install.services)])), {"ENVIRONMENT_NAME": app.environment_name},
            protocol, True) for app, staged in apps]
        if (lock := _vm_bundle_lock(env, base, link, parts, context, arch, progress)) is not None:
            packages.stage_wheels(lock, build, env.stored_id)
            parts = [replace(part, wheels=tuple(wheel.file for wheel in lock.wheels.get(part.folder, [])))
                     for part in parts]
        # A VM bundle's base bakes none in, as a Linux base's image does its defaults, so each value is the env's or the
        # setting's default.
        values = env.computer_values()
        settings = {name: values.get(name, setting.default) for name, setting in base.config.items()
                    if setting.phase == "build"}
        script = macos_vm_bundle.write(build, base, parts, link.ssh_password, settings)
        key = attempt.key = _key(base=base.forms.vm_bundle.image, provision=script, context=_digest(context),
                                 agents=_digest(build / "agents"), prepare=_digest(macos_vm_bundle.PREPARE),
                                 lock=lock.digest() if lock is not None else None, build_settings=settings,
                                 form="vm_bundle", arch=arch, format=FORMAT)
        image = f"{LOCAL_IMAGE_PREFIX}agentenv-computer-{key[7:23]}"
        if held is not None and image != held:
            raise RuntimeError(f"this agent-env composes it as {image}, not the {held} its record names, as another "
                               "agent-env composed it; compose it on this Mac with agent-env run or agent-env env "
                               "build, which record what this one composes")
        with contextlib.nullcontext() if held else holding_locks([_bundle_lock(image)], lambda: progress.at(
                pair, "waiting", f"waiting for the compose or probe of {image} under way on this Mac")):
            if asyncio.run(link.has_vm_bundle(image)):
                progress.at(pair, "reusing", "reusing the VM bundle of the same key, composed on this Mac earlier")
                result = {"status": "unverified", "key": key, "image": image, "reused": True, "built_at": now()}
            else:
                progress.at(pair, "composing", f"composing on {type(link).__name__}")
                asyncio.run(_provisioned(link, base.forms.vm_bundle.image, image.removeprefix(LOCAL_IMAGE_PREFIX),
                                         build))
                result = {"status": "unverified", "key": key, "image": image, "reused": False,
                          "builder": type(link).__name__, "built_at": now()}
            probed = _probed(env, driver, "vm_bundle", arch, result, link, progress)
            if probed["status"] == "failed":
                asyncio.run(link.delete_vm_bundle(image))
            return probed
    return attempt.result


def _vm_bundle_lock(env: ComputerEnv, base: ComputerArtifact, link: SandboxProvider, parts: list[macos_vm_bundle.Part],
                    context: Path, arch: str, progress: _Progress) -> packages.Lock | None:
    """The lock of ``parts``' venvs' wheels, their folders staged in ``context``: the one stored for their inputs, or
    one resolved first in a VM of the base on ``link``, which installs the pinned Python they run on and builds their
    wheels into the folder it shares, then is deleted. One resolve of given inputs runs on a Mac at once; another waits,
    then takes its lock. None when no layer has a venv."""
    resolver = macos_vm_bundle.resolver(parts)
    requirements = {part.folder: builder.sha256_of(context / part.folder / part.install.python)
                    if part.install.python else None for part in parts if part.venv}
    protocol = context / "protocol"
    inputs = packages.vm_bundle_inputs(resolver, _digest(macos_vm_bundle.PREPARE), requirements,
                                       protocol if protocol.is_dir() else None, base.forms.vm_bundle.image, arch)
    if inputs is None:
        return None
    pair = f"vm_bundle/{arch}"
    with holding_locks([f"vm-bundle-resolve:{inputs}"], lambda: progress.at(
            pair, "waiting", "waiting for the resolve of the same wheels under way on this Mac")):
        if (lock := packages.stored(inputs, env.stored_id)) is not None:
            return lock
        progress.at(pair, "resolving", f"resolving its venvs' wheels on {type(link).__name__}")
        with tempfile.TemporaryDirectory(prefix="agentenv-resolve-") as work:
            folder = Path(work) / macos_vm_bundle.BUILD_FOLDER
            shutil.copytree(context, folder / "context")
            shutil.copytree(macos_vm_bundle.PREPARE, folder / "prepare")
            (folder / "resolve.sh").write_text(resolver)
            secret = folder / ".password"
            secret.write_text(link.ssh_password + "\n")
            secret.chmod(0o600)
            asyncio.run(_resolved(link, base.forms.vm_bundle.image, f"agentenv-resolve-{inputs[7:23]}", folder))
            return packages.store_resolved(inputs, arch, {part.folder: folder / packages.staged_wheels(part.folder)
                                                          for part in parts if part.venv}, env.stored_id)


def _bundle_lock(image: str) -> str:
    return f"vm-bundle:{image}"


async def _resolved(link: SandboxProvider, source: str, name: str, folder: Path) -> None:
    """``folder``'s ``resolve.sh`` run in a VM of ``source`` on ``link``, named ``name``, which is deleted after; its
    output in the error when it fails."""
    vm = await link.open_builder(source, name, folder)
    try:
        code, stdout, stderr = await vm.exec_with_output(
            "bash", f"{macos_vm_bundle.SHARED_FILES}/{macos_vm_bundle.BUILD_FOLDER}/resolve.sh")
        if code != 0:
            raise builder.ResolveError(f"resolving exited {code}: {(stderr or stdout).strip()[-2000:]}",
                                       log=stdout + stderr)
    finally:
        await vm.terminate()


async def _provisioned(link: SandboxProvider, source: str, name: str, build: Path) -> None:
    """``name`` composed from ``source`` on ``link``: its builder booted with ``build`` shared, ``provision.sh`` run in it,
    and kept; a builder whose provisioning failed, its output in the error, or that couldn't be kept, deleted."""
    builder_vm = await link.open_builder(source, name, build)
    try:
        code, stdout, stderr = await builder_vm.exec_with_output(
            "bash", f"{macos_vm_bundle.SHARED_FILES}/{macos_vm_bundle.BUILD_FOLDER}/provision.sh")
        if code != 0:
            raise RuntimeError(f"provisioning exited {code}: {(stderr or stdout).strip()[-2000:]}")
        await link.keep_builder(builder_vm)
    except BaseException:
        await builder_vm.terminate()
        raise


def _part(layer: _Layer, programs: dict[str, str], environment: dict[str, Any], protocol: bool,
          app: bool) -> macos_vm_bundle.Part:
    """``layer`` as a VM bundle installs it: its programs, each run directly with its own variables when its service
    says so, and a venv's protocol as on Linux: this checkout's source, else an app's pinned to this release, which a
    driver's requirements name themselves."""
    direct = {service.command: service.environment or {} for service in layer.install.services if service.direct}
    extra = "[app]" if app else ""
    spec = (f'"$build/context/protocol{extra}"' if protocol else
            f"{_PROTOCOL}{extra}=={importlib.metadata.version(_PROTOCOL)}" if app else None)
    return macos_vm_bundle.Part(layer.folder, layer.prefix, layer.install,
                          {name: (command, direct.get(command)) for name, command in programs.items()},
                          environment, _venv(layer), spec if _venv(layer) else None, layer.served, layer.steps)


def _probed(env: ComputerEnv, driver: ComputerDriverEnv, form: str, arch: str, result: dict,
            sandbox_provider: SandboxProvider, progress: _Progress) -> dict:
    """Gate 3: the build of ``result`` booted on the first link of ``sandbox_provider`` that runs ``arch``'s ``form``
    natively, and probed at ``full`` (``probing.checked``). Still unverified, saying why, when no link does."""
    if not (links := boot.native_links(sandbox_provider, arch, form, env.stored_id)):
        how = {"disk": f"boots linux/{arch} disks", "vm_bundle": f"boots macos/{arch} VM bundles"}.get(
            form, f"runs linux/{arch}")
        return {**result, "reason": f"no link of the sandbox provider {how} natively to probe it"}
    progress.at(f"{form}/{arch}", "probing", f"probing on {type(links[0]).__name__}")
    return asyncio.run(probing.checked(links, env, driver, form, arch, result))


def _built(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, form: str, arch: str,
           builds: builder.Builds, progress: _Progress) -> dict:
    """The status of ``form`` and ``arch`` once built and pushed, or reused, or the stage that failed and why, each
    stage reported to ``progress``. A builder from ``builds`` resolves and builds it, opened when first needed."""
    pair = f"{form}/{arch}"
    with (_attempting(env, form, arch) as attempt, tempfile.TemporaryDirectory(prefix="agentenv-compose-") as work,
          contextlib.ExitStack() as opened):
        context, fetched, from_store = Path(work) / builder.CONTEXT, {}, {}
        context.mkdir()
        source, base_identity = _base(base, arch)
        layer = _stage_driver(context, driver, base, from_store)
        apps = [_stage_app(context, app, base, from_store) for app in env.composed_apps()]
        layers = [layer, *apps]
        wants = _wants(layers, _stage_protocol(context))
        platform, held = f"linux/{arch}", []

        def on() -> builder.Builder:
            if not held:
                link = builds.link(platform, base.image())
                native = platform in link.native_platforms()
                progress.at(pair, "starting its builder", f"building natively {builder.where(link)}" if native else
                            f"building emulated, which is slow, {builder.where(link)}; "
                            f"{builder.instead_of_emulating(builds.provider, platform)}")
                held.append(opened.enter_context(builds.open(link, platform, Path(work), base.image(), from_store)))
            return held[0]

        inputs = packages.inputs_digest(wants, context, base_identity, arch)
        lock = packages.stored(inputs, env.stored_id) if inputs is not None else None
        if inputs is not None and lock is None:
            attempt.stage = "resolve"
            resolving = on()
            progress.at(pair, "resolving packages")
            lock, fetched = packages.resolve(inputs, wants, resolving, source, arch, env.stored_id)
            progress.at(pair, "resolved", f"resolved {len(fetched)} files and stored them")
            attempt.stage = "build"
        dockerfile = _dockerfile(source, layers, base, _build_settings(env, base), lock)
        key = attempt.key = _key(base=base_identity, dockerfile=dockerfile, context=_digest(context),
                                 lock=lock.digest() if lock is not None else None, form=form, arch=arch, format=FORMAT)
        store, ref, pinned = _stored(env, key)
        if pinned is not None:
            progress.at(pair, "reusing", "reusing the image of the same key, built earlier")
            kept = labels(pinned, arch, store.auth(pinned))
            return {"status": "unverified", "key": key, "image": pinned, "reused": True,
                    "emulated": kept.get(EMULATED_LABEL) == "true", "lock": inputs,
                    "installed": json.loads(kept.get(INSTALLED_LABEL, "{}")), "built_at": now()}
        building_on = on()
        progress.at(pair, "building the image")
        if lock is not None:
            packages.stage(lock, building_on, env.stored_id, fetched)
        local, emulated = f"agentenv-computer:{key[7:23]}-{arch}", platform != building_on.platform
        building_on.build(f"{dockerfile}LABEL {KEY_LABEL}={key}\n", local, platform, {KEY_LABEL: key})
        if reason := _compare(building_on, local, layers, pair, progress):
            return failed(env, form, arch, attempt.stage, key, reason)
        installed = building_on.installed(local, layer.prefix)
        building_on.build(f"FROM {local}\n", local, platform, {
            INSTALLED_LABEL: json.dumps(installed, sort_keys=True), EMULATED_LABEL: "true" if emulated else "false",
        }, context="label")
        attempt.stage = "push"
        progress.at(pair, "pushing the image")
        building_on.push(local, ref, store)
        return {"status": "unverified", "key": key, "image": pin_digest(ref, store.auth(ref)), "reused": False,
                "emulated": emulated, "builder": building_on.name, "lock": inputs, "installed": installed,
                "built_at": now()}
    return attempt.result


def _built_disk(env: ComputerEnv, base: ComputerArtifact, arch: str, container: dict, builds: builder.Builds,
                progress: _Progress) -> dict:
    """The status of the disk form of ``arch`` once built from ``container``, its container's result, and pushed, or
    reused, or the stage that failed and why. Its key is of the container's key, the base disk and the disk recipe;
    the base disk is the base's own, the one published for it, or else one made on the builder from the base's
    container and the boot recipe. Its build runs no program of ``arch``'s but that recipe's."""
    pair, platform = f"disk/{arch}", f"linux/{arch}"
    with (_attempting(env, "disk", arch) as attempt,
          tempfile.TemporaryDirectory(prefix="agentenv-compose-") as work, contextlib.ExitStack() as opened):
        context = Path(work) / builder.CONTEXT
        shutil.copytree(DISK / "boot", context)
        shutil.copytree(DISK / "tools", context / "tools")
        base_disk, base_identity, recipe = *_base_disk(base, arch), _digest(context)
        key = attempt.key = _key(container=container["key"], base_disk=base_identity, recipe=recipe, form="disk",
                                 arch=arch, format=FORMAT)
        store, ref, pinned = _stored(env, key)
        if pinned is not None:
            progress.at(pair, "reusing", "reusing the disk of the same key, made earlier")
            return {"status": "unverified", "key": key, "image": pinned, "reused": True, "built_at": now()}
        link = builds.link(platform, base.image())
        progress.at(pair, "starting its builder", f"making the disk {builder.where(link)}")
        on = opened.enter_context(builds.open(link, platform, Path(work), base.image(), {}))
        on.login(store.auth(container["image"]))
        if base_disk is not None:
            on.login(get_config().get_image_store().auth(base_disk))
        if (tools := published_pinned(DISK_TOOLS_IMAGE)) is None:
            tools = f"agentenv-disk-tools:{DISK_TOOLS_IMAGE.tag()}"
            on.build((context / "tools" / "Dockerfile").read_text(), tools, on.platform, {},
                     context=f"{builder.CONTEXT}/tools")
        dockerfile = (context / "Dockerfile").read_text()
        if base_disk is None:
            progress.at(pair, "making the base's disk")
            base_disk = _made_base_disk(on, base, arch, f"{base_identity} {recipe}", tools, dockerfile)
        progress.at(pair, "making the disk")
        local = f"agentenv-computer:{key[7:23]}-disk-{arch}"
        on.build(dockerfile, local, platform, {KEY_LABEL: key}, build_args={
            "ROOTFS": container["image"], "ROOTFS_CONFIG": _encoded(_config(on, container["image"], arch)),
            "ESP": base_disk, "TOOLS": tools})
        attempt.stage = "push"
        progress.at(pair, "pushing the disk")
        on.push(local, ref, store)
        return {"status": "unverified", "key": key, "image": pin_digest(ref, store.auth(ref)), "reused": False,
                "builder": on.name, "built_at": now()}
    return attempt.result


def _made_base_disk(on: builder.Builder, base: ComputerArtifact, arch: str, made_from: str, tools: str,
                    dockerfile: str) -> str:
    """The base's disk for ``arch``, made on ``on`` from the base's container with the boot recipe's ``dockerfile`` at
    the base's size, under a tag of ``made_from``, what it's made from."""
    source, _ = _base(base, arch)
    tag = f"agentenv-computer:{hashlib.sha256(made_from.encode()).hexdigest()[:16]}-base-disk-{arch}"
    on.build(dockerfile, tag, f"linux/{arch}", {}, build_args={
        "ROOTFS": source, "ROOTFS_CONFIG": _encoded(_config(on, source, arch)), "TOOLS": tools,
        "SIZE_GB": str(base.forms.disk.min_disk_gb)})
    return tag


@dataclass
class _Attempt:
    """How far a pair's build got, its stage and key, and, when it failed, its result (``_attempting``)."""

    stage: str = "build"
    key: str | None = None
    result: dict | None = None


@contextlib.contextmanager
def _attempting(env: ComputerEnv, form: str, arch: str) -> Iterator[_Attempt]:
    """A pair's build, whose failure becomes its result: skipped, or failed at the stage it reached, saying why."""
    attempt = _Attempt()
    try:
        yield attempt
    except builder.Skipped as e:
        attempt.result = {"status": "skipped", "reason": str(e), "skipped_at": now()}
    except builder.Unbuildable as e:
        attempt.result = failed(env, form, arch, attempt.stage, attempt.key, str(e))
    except builder.ResolveError as e:
        attempt.result = failed(env, form, arch, "resolve", None, str(e), log=e.log or None)
    except DockerBuildError as e:
        # Docker can list a platform it has lost the emulation for, until it restarts; the build says so.
        unemulated = "exec format error" in str(e)
        attempt.result = failed(env, form, arch, attempt.stage, attempt.key,
                                 f"the builder can't run linux/{arch} programs: its Docker has no emulation for it"
                                 if unemulated else "the build failed", log=str(e))
    except subprocess.CalledProcessError as e:
        attempt.result = failed(env, form, arch, attempt.stage, attempt.key,
                                 f"{' '.join(e.cmd[:2])} failed: {(e.stderr or '').strip()}")
    except (RuntimeError, ValueError, OSError) as e:
        attempt.result = failed(env, form, arch, attempt.stage, attempt.key, f"{type(e).__name__}: {e}")


def _key(**inputs: Any) -> str:
    """A build's key: a digest of everything that goes into it."""
    return "sha256:" + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _stored(env: ComputerEnv, key: str) -> tuple[Any, str, str | None]:
    """The env's image store, the ref there of the image of ``key``, and that ref pinned when the store serves one
    already, for any env that built it."""
    store = get_config().get_image_store_for(env.stored_id)
    store.ensure_repository(REPOSITORY)
    ref = store.image_ref(REPOSITORY, key.removeprefix("sha256:"))
    return store, ref, _served(ref, store)


def _base_disk(base: ComputerArtifact, arch: str) -> tuple[str | None, str]:
    """The base's disk for ``arch``, pinned to its digest, and its part of a disk's key: the base's disk form's image,
    or the one published for the base's container sources; else None, and the base's container and disk size, which a
    disk is made from with the boot recipe, the key's recipe."""
    if (image := base.disk_image()) is not None:
        store = get_config().get_image_store()
        pinned = image.pinned_name if "@" in image.pinned_name else pin_digest(image.pinned_name,
                                                                              store.auth(image.pinned_name))
        return pinned, pinned
    if (published := published_disk(base.image().source_digest or "")) is not None:
        return published, published
    return None, f"made from {_base(base, arch)[1]} at {base.forms.disk.min_disk_gb} GB"


def _config(on: builder.Builder, image: str, arch: str) -> dict:
    """The config object of ``image``'s ``arch`` image: from its registry, or from the builder's Docker for one only it
    holds."""
    if names_registry(image):
        return image_config(image, arch, get_config().get_image_store().auth(image))
    return on.image_config(image)


def _encoded(config: dict) -> str:
    return base64.b64encode(json.dumps(config, sort_keys=True).encode()).decode()


def _compare(building_on: builder.Builder, tag: str, layers: list[_Layer], pair: str,
             progress: _Progress) -> str | None:
    """Why ``tag`` is refused: a part's run scripts or install steps certainly undoing what an earlier part's installed
    (``clashes``); None, with any other file two parts wrote differently said to ``progress``. Read only when two or
    more parts have such steps, the only ones that write outside a part's folder."""
    runs = {layer.env_id: _runs(layer) for layer in layers if _runs(layer)}
    if len(runs) < 2:
        return None
    progress.at(pair, "comparing the files its parts' run scripts and install steps wrote")
    written = building_on.written(tag, [command for commands in runs.values() for _, command in commands])
    lost, differ = clashes.compare(list(zip([env_id for env_id, commands in runs.items() for _ in commands], written)))
    if lost:
        return "a part undoes what an earlier one installed: " + "; ".join(lost)
    if differ:
        progress.at(pair, "compared", "not refused, as each may keep the earlier part's change: two parts wrote "
                    f"{', '.join(differ[:_SHOWN])}" + (f" and {len(differ) - _SHOWN} more" if len(differ) > _SHOWN
                                                       else ""))
    return None


def _base(base: ComputerArtifact, arch: str) -> tuple[str, str]:
    """What the composed image is built ``FROM`` for ``arch``, and the base's part of the key, found without building
    anything: a registry image pinned to its digest, or the tag a build of the base's stored build context for that
    arch gets, the same on every machine."""
    image = base.image()
    if names_registry(image.image_name):
        store = get_config().get_image_store()
        pinned = image.pinned_name if "@" in image.pinned_name else pin_digest(image.pinned_name, store.auth(image.pinned_name))
        return pinned, pinned
    return platform_tag(image, f"linux/{arch}"), f"sources:{image.source_digest}"


@dataclass(frozen=True)
class _Layer:
    """A driver's or an app's part of a build: its folder in the build context, its prefix in the image, what it
    installs, the file of its supervisord programs, and the id of its env. An app's venv also has the protocol's ``app`` extra, for the
    app's own server, which listens in a folder of the app's the session user owns."""

    folder: str
    prefix: str
    install: Install
    conf: str
    env_id: str
    app: bool = False
    served: str | None = None
    steps: tuple[tuple[str, str], ...] = ()


def _stage_driver(context: Path, driver: ComputerDriverEnv, base: ComputerArtifact,
                  from_store: dict[str, tuple[str, str]]) -> _Layer:
    """Write the driver's install files under ``context/driver/`` and its supervisord programs beside them, served on
    its port, its folder under ``DRIVERS`` named for its id's last part, its programs and their file under ``driver-``
    as an app's are under ``app-``, so neither ever takes the other's."""
    install, name = driver.install[base.os.type], gate.folder_name(driver.id, "driver")
    layer = _Layer("driver", f"{DRIVERS}/{name}", install, f"agentenv-driver-{name}.conf", driver.id)
    _stage(context, layer, f"driver {driver.id}", driver.files, _services(f"driver-{name}", install),
           {"MCP_PORT": driver.port}, from_store)
    return layer


def _stage_app(context: Path, app: ComputerAppEnv, base: ComputerArtifact,
               from_store: dict[str, tuple[str, str]]) -> _Layer:
    """Write the app's install files under ``context/apps/<name>/``, with the values of its build settings the env
    fixes, and its supervisord programs beside them, run under its name: its services and its own server, which serves
    its class from its venv and launches the app. Its folder under ``APPS`` is named for its environment name."""
    install, name = app.install[base.os.type], gate.app_folder(app)
    file, _, cls = app.module.partition(":")
    served = shlex.quote(f"{PurePosixPath(file).relative_to(install.files)}:{cls}")
    layer = _Layer(f"apps/{name}", f"{APPS}/{name}", install, f"agentenv-app-{name}.conf", app.id, app=True,
                   served=served,
                   steps=tuple((step.name, step.scope) for step in app.declaration.install_steps
                               if step.os is None or base.os.type in step.os))
    programs = dict(zip(gate.programs(app, base.os.type), [
        f"venv/bin/python -m agentenv_protocol.computer.serve_app {served}",
        *(service.command for service in install.services)]))
    _stage(context, layer, f"app {app.id}", app.files, programs, {"ENVIRONMENT_NAME": app.environment_name},
           from_store)
    if app.build_settings:
        (context / layer.folder / BUILD_SETTINGS).parent.mkdir(exist_ok=True)
        (context / layer.folder / BUILD_SETTINGS).write_text(json.dumps(app.build_settings, sort_keys=True))
        # Inline, over any the app's files hold, as an app stored before that folder was refused can.
        from_store.pop(f"{layer.folder}/{BUILD_SETTINGS.as_posix()}", None)
    return layer


def _services(name: str, install: Install) -> dict[str, str]:
    """The commands of ``install``'s services by their programs' names, each named for ``name``."""
    return {f"{name}-{service.name}": service.command for service in install.services}


def _stage(context: Path, layer: _Layer, what: str, files: FileArtifactUniverse, programs: dict[str, str],
           environment: dict[str, Any], from_store: dict[str, tuple[str, str]]) -> None:
    """Write the ``files`` of ``what``'s folder its install names to ``layer``'s folder, a script, a binary or a file its
    install runs executable, each added to ``from_store`` by its path with its object URL and sha256, for a builder to
    fetch; and ``programs``, commands by name, as its supervisord programs, with ``environment``."""
    folder = PurePosixPath(layer.install.files)
    runs = {PurePosixPath(run.script) for run in layer.install.run or []}
    for name, artifact in files.get_file_artifacts().items():
        try:
            path = PurePosixPath(checked_path(name))
        except Unwritable as e:
            raise ValueError(f"{what}: {e}") from None
        if folder == PurePosixPath(".") or path.is_relative_to(folder):
            target = context / layer.folder / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data := artifact.load())
            from_store[target.relative_to(context).as_posix()] = (artifact.object_url, hashlib.sha256(data).hexdigest())
            if data.startswith(builder.RUN_ONLY) or path.relative_to(folder) in runs:
                target.chmod(builder.EXECUTABLE)
    (context / layer.folder).mkdir(parents=True, exist_ok=True)
    (context / layer.conf).write_text(_programs(programs, layer.prefix, environment))


def _programs(commands: dict[str, str], prefix: str, environment: dict[str, Any]) -> str:
    variables = ",".join(f'{key}="{value}"' for key, value in environment.items())
    programs = []
    for name, command in commands.items():
        program, _, rest = command.partition(" ")
        if "/" in program and not program.startswith("/"):
            program = f"{prefix}/{program}"
        programs.append(f"[program:{name}]\ncommand={program} {rest}".rstrip() + "\n"
                        f"directory={prefix}\nenvironment={variables}\nautorestart=true\nredirect_stderr=true\n"
                        f"stdout_logfile=/var/log/agentenv/{name}.log\n")
    return "\n".join(programs)


def _stage_protocol(context: Path) -> bool:
    """Copy the protocol's source to ``context/protocol/`` when this install runs it from one, as a checkout does;
    False when it's installed from an index, as a release is, and the driver's requirements name it."""
    if (source := _protocol_source()) is None:
        return False
    shutil.copytree(source / "src", context / "protocol" / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in _PROTOCOL_FILES:
        if (source / name).is_file():
            shutil.copy(source / name, context / "protocol" / name)
    return True


def _protocol_source() -> Path | None:
    direct = importlib.metadata.distribution(_PROTOCOL).read_text("direct_url.json")
    url = json.loads(direct).get("url", "") if direct else ""
    if not url.startswith("file://"):
        return None
    folder = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path))
    return folder if (folder / "pyproject.toml").is_file() else None


def _build_settings(env: ComputerEnv, base: ComputerArtifact) -> dict[str, Any]:
    """The env's values of the base's build-phase settings, which the composed image fixes."""
    values = env.computer_values()
    return {name: values[name] for name, setting in base.config.items() if setting.phase == "build" and name in values}


def _dockerfile(source: str, layers: list[_Layer], base: ComputerArtifact, build_settings: dict,
                lock: packages.Lock | None) -> str:
    """The composed image's Dockerfile but its key's label, which the key is a digest of: the build-phase settings, the
    lock's .debs, then each layer's files under its prefix, its venv beside them from the lock's wheels, its run scripts
    and an app's install steps, each run as root or the session user, every install with the network closed; then its
    programs, and an app's server's folder owned by the session user."""
    user = base.session.user
    lines = [f"FROM {source}", "USER root",
             *(f"ENV AGENTENV_SETTING_{name.upper()}={json.dumps(str(value))}" for name, value in build_settings.items())]
    if lock is not None and lock.apt:
        debs = f"/tmp/agentenv-{packages.STAGED_APT}"
        lines += [f"COPY {packages.STAGED_APT}/ {debs}/",
                  f"RUN --network=none apt-get install -y --no-install-recommends {debs}/*.deb && rm -rf {debs}"]
    for layer in layers:
        prefix = layer.prefix
        lines.append(f"COPY {layer.folder}/ {prefix}/")
        if _venv(layer):
            staged = packages.staged_wheels(layer.folder)
            wheels = f"/tmp/agentenv-{staged}"
            if lock is not None and lock.wheels.get(layer.folder):
                lines += [f"COPY {staged}/ {wheels}/",
                          f"RUN --network=none python3 -m venv {prefix}/venv && {prefix}/venv/bin/pip install "
                          f"--no-index --no-deps {wheels}/*.whl && {prefix}/venv/bin/pip list --format=json > "
                          f"{prefix}/installed.json && rm -rf {wheels}"]
            else:
                lines.append(f"RUN --network=none python3 -m venv {prefix}/venv && {prefix}/venv/bin/pip list "
                             f"--format=json > {prefix}/installed.json")
        if layer.app:
            lines.append(f"RUN mkdir -p {prefix}/{SOCKET.parent} && chown {user} {prefix}/{SOCKET.parent}")
        for scope, command in _runs(layer):
            lines += _as(scope, user, f"RUN --network=none {command}")
        lines.append(f"COPY {layer.conf} /etc/supervisor/conf.d/{layer.conf}")
        checks = [f"test -d {prefix}", f"test -f /etc/supervisor/conf.d/{layer.conf}"]
        if _venv(layer):
            checks.append(f"test -x {prefix}/venv/bin/python")
        lines.append(f"RUN {' && '.join(checks)}")
    lines.append(f"USER {user}")
    return "\n".join(lines) + "\n"


def _runs(layer: _Layer) -> list[tuple[str, str]]:
    """The commands of ``layer``'s run scripts and its install steps, each with the scope it runs as: the only steps of
    a build that write outside the layer's folder."""
    return [*((run.scope, f"cd {layer.prefix} && {shlex.quote('./' + run.script)}") for run in layer.install.run or []),
            *((scope, f"cd {layer.prefix} && venv/bin/python -m agentenv_protocol.computer.run_install_step "
                      f"{layer.served} {step}") for step, scope in layer.steps)]


def _as(scope: str, user: str, run: str) -> list[str]:
    """``run`` as root, or for ``session``, as ``user``."""
    return [run] if scope == "system" else [f"USER {user}", run, "USER root"]


def _venv(layer: _Layer) -> bool:
    """Whether ``layer`` has a venv: its requirements, or an app's own server."""
    return layer.install.python is not None or layer.app


def _wants(layers: list[_Layer], protocol: bool) -> list[packages.Want]:
    """What each layer asks a lock for; a venv adds the protocol, for an app with its ``app`` extra: from this checkout's
    source when ``protocol``, else an app's pinned to this release, which a driver's requirements name themselves."""
    wants = []
    for layer in layers:
        extra = "[app]" if layer.app else ""
        adds = ((f"/ctx/protocol{extra}",) if protocol else
                (f"{_PROTOCOL}{extra}=={importlib.metadata.version(_PROTOCOL)}",) if layer.app else ())
        wants.append(packages.Want(layer.folder, layer.install, _venv(layer), adds if _venv(layer) else ()))
    return wants


def _served(ref: str, store) -> str | None:
    """``ref`` pinned to the digest the image store serves for it, or None when it serves none."""
    try:
        return pin_digest(ref, store.auth(ref))
    except NoSuchImageError:
        return None


def _computer(env: ComputerEnv, base: ComputerArtifact, built: dict[str, str], disks: dict[str, str],
              bundles: dict[str, str]) -> dict | None:
    """Write the env's computer over the images ``built``, by arch, joined into one index in the env's image store, signed
    in to it: its arches those, its OS and session the base's, its settings the base's with the env's values as their
    defaults; with a disk form over ``disks`` when there's one for each of those arches; for a macOS base, over its VM
    bundles alone. None when nothing built."""
    config = {name: setting.model_dump() for name, setting in base.config.items()}
    for name, value in env.computer_values().items():
        config[name]["default"] = value
    if base.forms.vm_bundle is not None:
        if not bundles:
            return None
        arches = [arch for arch in base.arch if arch in bundles]
        forms = {"vm_bundle": {**base.forms.vm_bundle.model_dump(), "image": bundles[arches[0]]}}
        return _written(env, base, arches, config, forms)
    if not built:
        return None
    arches = [arch for arch in base.arch if arch in built]
    image = _joined(env, "computer_image", "composed computer", built, arches)
    forms = {"container": {"image": {"id": image.id, "version": image.version}, "init": base.forms.container.init,
                           "input_channel": base.forms.container.input_channel}}
    if base.forms.disk is not None and all(arch in disks for arch in arches):
        disk = _joined(env, "computer_disk", "composed computer's disk", disks, arches)
        forms["disk"] = {**base.forms.disk.model_dump(exclude={"image"}), "image": {"id": disk.id, "version": disk.version}}
    return _written(env, base, arches, config, forms)


def _written(env: ComputerEnv, base: ComputerArtifact, arches: list[str], config: dict, forms: dict) -> dict:
    """The env's computer of ``forms``, written, with what each form's probe found as its ``validated_boot``."""
    computer = ComputerArtifact.put(
        id=derive_id(env.stored_id, "computer"), description=f"{base.id} composed with {env.driver.id} for {env.stored_id}",
        arch=arches, os=base.os.model_dump(), session=base.session.model_dump(), config=config, forms=forms)
    composed = env.metadata.get("composed", {})
    if probes := {form: by_arch for form in forms if (by_arch := {
            arch: composed[form][arch]["probe"] for arch in arches if "probe" in composed.get(form, {}).get(arch, {})})}:
        computer.merge_metadata({"validated_boot": probes})
    return {"id": computer.id, "version": computer.version}


def _joined(env: ComputerEnv, name: str, what: str, images: dict[str, str], arches: list[str]) -> DockerImageArtifact:
    """``images`` of ``arches`` joined into one index in the env's image store, signed in to it, as an image artifact."""
    store = get_config().get_image_store_for(env.stored_id)
    index = store.image_ref(REPOSITORY, "index-" + hashlib.sha256(
        ",".join(images[arch] for arch in arches).encode()).hexdigest())
    write_index(index, {f"linux/{arch}": images[arch] for arch in arches}, store.auth(index))
    return DockerImageArtifact.put_ref(derive_id(env.stored_id, name), description=f"{env.stored_id}'s {what}",
                                       image_name=index)


def _digest(folder: Path) -> str:
    """A digest of the files under ``folder``, by their paths, bytes and whether each is executable, which the composer
    alone decides (umask never sets it)."""
    hashed = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        hashed.update(path.relative_to(folder).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest()
                      + (b"x" if path.stat().st_mode & 0o100 else b"-"))
    return "sha256:" + hashed.hexdigest()


class _Progress:
    """What a compose reports as it goes, to ``on_progress`` and the log: a line as each pair reaches a stage, one each
    ``HEARTBEAT_S`` a stage goes on, and how each pair ended, with the time since the compose began."""

    def __init__(self, env: ComputerEnv, on_progress: Callable[[str], None] | None):
        self.env, self.on_progress, self.lock = env, on_progress, threading.Lock()
        self.began, self.stages, self.stopped = time.monotonic(), {}, threading.Event()

    def __enter__(self) -> _Progress:
        self.beating = threading.Thread(target=self._beat, daemon=True)
        self.beating.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stopped.set()
        self.beating.join()

    def say(self, line: str, failed: bool = False) -> None:
        """``line`` to ``on_progress``, and the log; a failure as a warning when nothing else shows it."""
        logger.log(logging.WARNING if failed and self.on_progress is None else logging.INFO,
                   f"{self.env.stored_id}: {line}")
        if self.on_progress is not None:
            with self.lock:
                self.on_progress(line)

    def at(self, pair: str, stage: str, line: str | None = None) -> None:
        now = time.monotonic()
        with self.lock:
            self.stages[pair] = (stage, now, now)
        self.say(f"{pair}: {line or stage}")

    def ended(self, pair: str, result: dict) -> None:
        with self.lock:
            self.stages.pop(pair, None)
        status = f"failed at {result['stage']}" if result["status"] == "failed" else result["status"]
        reason = f": {result['reason']}" if result.get("reason") else ""
        self.say(f"{pair}: {'reused, ' if result.get('reused') else ''}{status}{reason} "
                 f"({took(time.monotonic() - self.began)})", failed=result["status"] == "failed")

    def _beat(self) -> None:
        while not self.stopped.wait(HEARTBEAT_S / 10):
            now = time.monotonic()
            with self.lock:
                due = [(pair, stage, since) for pair, (stage, since, said) in self.stages.items()
                       if now - said >= HEARTBEAT_S]
                for pair, stage, since in due:
                    self.stages[pair] = (stage, since, now)
            for pair, stage, since in due:
                self.say(f"{pair}: still {stage} ({took(now - since)})")


def took(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} s"
