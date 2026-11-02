"""The composer: a computer env's base, driver and apps built into one image for each form and arch, on a builder the
put's sandbox provider gives (``builder``), tagged in the env's image store by a key of everything that goes into it, so
an image of that key, built for any env and on any builder, is reused.
Each result is recorded in the env's metadata as it finishes (``composed``), and the arches that built are written as
the env's computer, a ``computer`` artifact over an index of their images."""

from __future__ import annotations

import asyncio
import contextlib
import copy
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agentenv_protocol.computer import apps as installed_apps
from agentenv_protocol.computer.app import BUILD_SETTINGS, SOCKET
from agentenv_protocol.computer.conformance import run_probe

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.config import get_config
from agent_env.entity.export import Unwritable, checked_path
from agent_env.env.env import Env
from agent_env.env.envs.computer.utils import boot, builder, clashes, gate, packages
from agent_env.env.envs.computer.utils.install import Install
from agent_env.providers.sandbox_providers.host_images import platform_tag
from agent_env.store.base import ConcurrentModificationError
from agent_env.store.ids import derive_id
from agent_env.store.image_store.image_contents import labels
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.store.image_store.registry_api import NoSuchImageError, pin_digest, write_index
from agent_env.utils.docker.build import DockerBuildError

if TYPE_CHECKING:
    from agentenv_protocol.computer.conformance import ProbeReport

    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.env.envs.computer.app_env import ComputerAppEnv
    from agent_env.env.envs.computer.driver_env import ComputerDriverEnv

logger = logging.getLogger(__name__)

#: The image store repository composed computers are pushed to, tagged by their keys.
REPOSITORY = "agentenv-computers"
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
# A pair's statuses once its image is in the store.
_BUILT = ("unverified", "ready")
_WRITE_TRIES = 4
#: How long a probe's computer may run before its sandbox provider stops it.
_PROBE_TTL_S = 1800
#: How long the probe at full may take on a computer that booted: about ten seconds on a healthy one.
_PROBE_TIMEOUT_S = 120
#: How often a compose says a pair is still at a stage, which can take most of an hour under emulation.
_HEARTBEAT_S = 600
_PROTOCOL = "agentenv-framework-protocol"
# What building the protocol from its source reads, beside src/.
_PROTOCOL_FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
#: How many of the files two parts wrote differently a progress line names.
_SHOWN = 10


@dataclass(frozen=True)
class Composing:
    """How a put composes a computer env: on ``sandbox_provider``, by default the env's; emulating an arch no link of
    it runs natively unless ``emulate`` is False, which leaves the arch skipped for a later compose; each pair's
    stages told to ``on_progress``, a line at a time."""

    sandbox_provider: SandboxProvider | None = None
    emulate: bool = True
    on_progress: Callable[[str], None] | None = None


def compose(env: ComputerEnv, pairs: list[tuple[str, str]] | None = None, composing: Composing = Composing()) -> None:
    """Build each of ``pairs``, by default each form the base has in each arch the env composes, in parallel, as
    ``composing`` says, and probe each one its provider runs natively (gate 3), recording each as it starts and
    finishes, then write the env's computer over every pair built so far. Each pair's stages and how it ended go to the
    log too (``_Progress``)."""
    from agent_env.providers import get_env_sandbox_provider

    base, driver = env.computer, env.driver
    provider = composing.sandbox_provider or get_env_sandbox_provider()
    builds = builder.Builds(provider, env.stored_id, REPOSITORY, composing.emulate)
    pairs = pairs if pairs is not None else [(form, arch) for form in gate.forms(base)
                                             for arch in gate.arches(base, driver, env.composed_apps())]
    record = _Record(env)
    record.started(pairs)
    with _Progress(env, composing.on_progress) as progress, ThreadPoolExecutor(max_workers=max(len(pairs), 1)) as pool:
        progress.say(f"composing {', '.join(f'{form}/{arch}' for form, arch in pairs)} on "
                     f"{' or '.join(type(link).__name__ for link in provider.links)}")
        futures = {pool.submit(_composed, env, base, driver, form, arch, builds, progress): (form, arch)
                   for form, arch in pairs}
        for future in as_completed(futures):
            form, arch = futures[future]
            result = future.result()
            progress.ended(f"{form}/{arch}", result)
            record.finished(form, arch, result)
    record.computer(_computer(env, base, record.built()))


def current(env: ComputerEnv) -> bool:
    """Whether the env's computer was written over every pair built, as a put or env build that failed writing it
    leaves it not."""
    return "computer" in env.metadata and env.metadata["computer"].get("images") == _Record(env).built()


def pending(env: ComputerEnv, sandbox_provider: SandboxProvider | None = None) -> list[tuple[str, str]]:
    """The forms and arches of ``env`` left to compose: never started, failed, left building by a put that stopped, or
    built but unprobed though ``sandbox_provider``, by default the env's, runs the arch natively."""
    from agent_env.providers import get_env_sandbox_provider

    composed, provider = env.metadata.get("composed", {}), sandbox_provider or get_env_sandbox_provider()
    statuses = {(form, arch): composed.get(form, {}).get(arch, {}).get("status")
                for form in gate.forms(env.computer)
                for arch in gate.arches(env.computer, env.driver, env.composed_apps())}
    return [(form, arch) for (form, arch), status in statuses.items()
            if status not in _BUILT or (status == "unverified" and boot.native_links(provider, arch))]


def _composed(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, form: str, arch: str,
              builds: builder.Builds, progress: _Progress) -> dict:
    """``form`` and ``arch`` built or reused, then probed when it built (gate 3)."""
    result = _built(env, base, driver, form, arch, builds, progress)
    if result["status"] != "unverified":
        return result
    return _probed(env, driver, form, arch, result, builds.provider, progress)


def _probed(env: ComputerEnv, driver: ComputerDriverEnv, form: str, arch: str, result: dict,
            sandbox_provider: SandboxProvider, progress: _Progress) -> dict:
    """Gate 3: the build of ``result`` booted on the first link of ``sandbox_provider`` that runs ``arch`` natively, and
    probed at ``full``: ready, with the driver's card, or failed at the probe; each with a summary of the probe and its
    report stored. Still unverified, saying why, when no link runs ``arch`` natively."""
    if not (links := boot.native_links(sandbox_provider, arch)):
        return {**result, "reason": f"no link of the sandbox provider runs linux/{arch} natively to probe it"}
    progress.at(f"{form}/{arch}", "probing", f"probing on {type(links[0]).__name__}")
    try:
        report, card, provider = asyncio.run(_probe(links, env, driver, result["image"]))
    except TimeoutError:
        return _failed(env, form, arch, "probe", result["key"], f"the probe didn't finish within {_PROBE_TIMEOUT_S} s")
    except Exception as e:
        return _failed(env, form, arch, "probe", result["key"], f"it didn't boot: {e}")
    stored = FileArtifact.put_bytes(derive_id(env.stored_id, f"probe_report_{form}_{arch}"), description=(
        f"the {form}/{arch} probe of {env.stored_id} v{env.version}"), filename="probe.json",
        content=report.model_dump_json(indent=1).encode(), content_type="application/json")
    probe = {"level": report.level, "provider": provider, "passed": report.passed, "checks": report.counts(),
             "extensions": report.extensions, "report": {"id": stored.id, "version": stored.version},
             "probed_at": _now()}
    if not report.passed:
        failing = "; ".join(f"{c.check} {c.name}: {c.detail}" for c in report.checks if c.status == "failed")
        return {**result, "status": "failed", "stage": "probe", "reason": f"the probe failed: {failing}",
                "failed_at": _now(), "probe": probe}
    return {**result, "status": "ready", "card": card, "probe": probe}


async def _probe(links: list[SandboxProvider], env: ComputerEnv, driver: ComputerDriverEnv,
                 image: str) -> tuple[ProbeReport, dict, str]:
    """``image`` booted at the env's settings with a control token of its own, as a deploy boots it, on the first of
    ``links`` that boots it, probed at ``full`` with that token within ``_PROBE_TIMEOUT_S`` and torn down; the report,
    the driver's card and the provider's name."""
    token = boot.control_token()
    environment = boot.settings_environment(env.computer, _values(env), environment_name=env.environment_name)
    problems = []
    for link in links:
        try:
            booted = await boot.boot(link, image=image, port=driver.port, environment=environment, token=token,
                                     network_policy=None, timeout=_PROBE_TTL_S, attribution=None,
                                     ready_s=boot.ready_timeout(env.composed_apps(), env.computer.os.type))
        except Exception as e:
            problems.append(f"{type(link).__name__}: {e}")
            continue
        try:
            async with asyncio.timeout(_PROBE_TIMEOUT_S):
                return await run_probe(booted.url, "full", token=token), booted.card, type(link).__name__
        finally:
            await boot.discard(booted.sandbox)
    raise RuntimeError("; ".join(problems))


def _built(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, form: str, arch: str,
           builds: builder.Builds, progress: _Progress) -> dict:
    """The status of ``form`` and ``arch`` once built and pushed, or reused, or the stage that failed and why, each
    stage reported to ``progress``. A builder from ``builds`` resolves and builds it, opened when first needed."""
    stage, key, pair = "build", None, f"{form}/{arch}"
    try:
        with tempfile.TemporaryDirectory(prefix="agentenv-compose-") as work, contextlib.ExitStack() as opened:
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
                    link = builds.link(platform)
                    native = platform in link.native_platforms()
                    progress.at(pair, "starting its builder", f"building natively {builder.where(link)}" if native else
                                f"building emulated, which is slow, {builder.where(link)}; "
                                f"{builder.instead_of_emulating(builds.provider, platform)}")
                    held.append(opened.enter_context(builds.open(link, platform, Path(work), base.image(), from_store)))
                return held[0]

            inputs = packages.inputs_digest(wants, context, base_identity, arch)
            lock = packages.stored(inputs, env.stored_id) if inputs is not None else None
            if inputs is not None and lock is None:
                stage = "resolve"
                resolving = on()
                progress.at(pair, "resolving packages")
                lock, fetched = packages.resolve(inputs, wants, resolving, source, arch, env.stored_id)
                progress.at(pair, "resolved", f"resolved {len(fetched)} files and stored them")
                stage = "build"
            dockerfile = _dockerfile(source, layers, base, _build_settings(env, base), lock)
            key = "sha256:" + hashlib.sha256(json.dumps({
                "base": base_identity, "dockerfile": dockerfile, "context": _digest(context),
                "lock": lock.digest() if lock is not None else None, "form": form, "arch": arch,
                "format": FORMAT,
            }, sort_keys=True).encode()).hexdigest()
            store = get_config().get_image_store_for(env.stored_id)
            store.ensure_repository(REPOSITORY)
            ref = store.image_ref(REPOSITORY, key.removeprefix("sha256:"))
            if (pinned := _served(ref, store)) is not None:
                progress.at(pair, "reusing", "reusing the image of the same key, built earlier")
                kept = labels(pinned, arch, store.auth(pinned))
                return {"status": "unverified", "key": key, "image": pinned, "reused": True,
                        "emulated": kept.get(EMULATED_LABEL) == "true", "lock": inputs,
                        "installed": json.loads(kept.get(INSTALLED_LABEL, "{}")), "built_at": _now()}
            building_on = on()
            progress.at(pair, "building the image")
            if lock is not None:
                packages.stage(lock, building_on, env.stored_id, fetched)
            local, emulated = f"agentenv-computer:{key[7:23]}-{arch}", platform != building_on.platform
            building_on.build(f"{dockerfile}LABEL {KEY_LABEL}={key}\n", local, platform, {KEY_LABEL: key})
            if reason := _compare(building_on, local, layers, pair, progress):
                return _failed(env, form, arch, stage, key, reason)
            installed = building_on.installed(local, layer.prefix)
            building_on.build(f"FROM {local}\n", local, platform, {
                INSTALLED_LABEL: json.dumps(installed, sort_keys=True), EMULATED_LABEL: "true" if emulated else "false",
            }, context="label")
            stage = "push"
            progress.at(pair, "pushing the image")
            building_on.push(local, ref, store)
            return {"status": "unverified", "key": key, "image": pin_digest(ref, store.auth(ref)), "reused": False,
                    "emulated": emulated, "builder": building_on.name, "lock": inputs, "installed": installed,
                    "built_at": _now()}
    except builder.Skipped as e:
        return {"status": "skipped", "reason": str(e), "skipped_at": _now()}
    except builder.Unbuildable as e:
        return _failed(env, form, arch, stage, key, str(e))
    except builder.ResolveError as e:
        return _failed(env, form, arch, "resolve", None, str(e), log=e.log or None)
    except DockerBuildError as e:
        # Docker can list a platform it has lost the emulation for, until it restarts; the build says so.
        unemulated = "exec format error" in str(e)
        return _failed(env, form, arch, stage, key, f"the builder can't run linux/{arch} programs: its Docker has no "
                       "emulation for it" if unemulated else "the build failed", log=str(e))
    except subprocess.CalledProcessError as e:
        return _failed(env, form, arch, stage, key, f"{' '.join(e.cmd[:2])} failed: {(e.stderr or '').strip()}")
    except (RuntimeError, ValueError, OSError) as e:
        return _failed(env, form, arch, stage, key, f"{type(e).__name__}: {e}")


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
    values = _values(env)
    return {name: values[name] for name, setting in base.config.items() if setting.phase == "build" and name in values}


def _values(env: ComputerEnv) -> dict[str, Any]:
    """The env's values of its base's settings: its ``computer_settings``, and its display as the resolution."""
    display = env.display
    return {**env.computer_settings, **({"resolution": f"{display['width']}x{display['height']}"} if display else {})}


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


def _failed(env: ComputerEnv, form: str, arch: str, stage: str, key: str | None, reason: str,
            log: str | None = None) -> dict:
    result = {"status": "failed", "stage": stage, "reason": reason, "failed_at": _now()}
    if key is not None:
        result["key"] = key
    if log is not None:
        stored = FileArtifact.put_bytes(derive_id(env.stored_id, f"build_log_{form}_{arch}"), description=(
            f"the {form}/{arch} build of {env.stored_id} v{env.version}, which failed"), filename="build.log",
            content=log.encode(), content_type="text/plain")
        result["log"] = {"id": stored.id, "version": stored.version}
        result["reason"] = f"{reason}: {log.strip().splitlines()[-1] if log.strip() else 'no output'}"
    return result


def _computer(env: ComputerEnv, base: ComputerArtifact, built: dict[str, str]) -> dict | None:
    """Write the env's computer over the images ``built``, by arch, joined into one index in the env's image store, signed
    in to it: its arches those, its OS and session the base's, its settings the base's with the env's values as their
    defaults. None when nothing built."""
    if not built:
        return None
    arches = [arch for arch in base.arch if arch in built]
    store = get_config().get_image_store_for(env.stored_id)
    index = store.image_ref(REPOSITORY, "index-" + hashlib.sha256(
        ",".join(built[arch] for arch in arches).encode()).hexdigest())
    write_index(index, {f"linux/{arch}": built[arch] for arch in arches}, store.auth(index))
    image = DockerImageArtifact.put_ref(derive_id(env.stored_id, "computer_image"), description=(
        f"{env.stored_id}'s composed computer"), image_name=index)
    config = {name: setting.model_dump() for name, setting in base.config.items()}
    for name, value in _values(env).items():
        config[name]["default"] = value
    computer = ComputerArtifact.put(
        id=derive_id(env.stored_id, "computer"), description=f"{base.id} composed with {env.driver.id} for {env.stored_id}",
        arch=arches, os=base.os.model_dump(), session=base.session.model_dump(), config=config,
        forms={"container": {"image": {"id": image.id, "version": image.version}, "init": base.forms.container.init,
                             "input_channel": base.forms.container.input_channel}})
    composed = env.metadata.get("composed", {}).get("container", {})
    if probes := {arch: composed[arch]["probe"] for arch in arches if "probe" in composed.get(arch, {})}:
        computer.merge_metadata({"validated_boot": {"container": probes}})
    return {"id": computer.id, "version": computer.version}


def _digest(folder: Path) -> str:
    """A digest of the files under ``folder``, by their paths, bytes and whether each is executable, which the composer
    alone decides (umask never sets it)."""
    hashed = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        hashed.update(path.relative_to(folder).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest()
                      + (b"x" if path.stat().st_mode & 0o100 else b"-"))
    return "sha256:" + hashed.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Record:
    """The env's ``composed`` metadata, by form and arch. Each result is merged into the stored table pair by pair, the
    table read again when another build wrote it first, so builds of one env at once keep each other's pairs, and a
    result never undoes one that got further (``_rank``)."""

    def __init__(self, env: ComputerEnv):
        self.env, self.lock = env, threading.Lock()

    def started(self, pairs: list[tuple[str, str]]) -> None:
        now = _now()
        self._merge({pair: {"status": "building", "started_at": now} for pair in pairs})

    def finished(self, form: str, arch: str, result: dict) -> None:
        self._merge({(form, arch): result})

    def built(self) -> dict[str, str]:
        """Each arch's image of the container form, of the pairs built or reused."""
        return {arch: result["image"] for arch, result in self.env.metadata.get("composed", {}).get("container", {}).items()
                if result.get("status") in _BUILT}

    def computer(self, ref: dict | None) -> None:
        """Record the env's computer, with the images it was written over."""
        if ref is not None:
            with self.lock:
                self.env.merge_metadata({"computer": {**ref, "images": self.built()}})

    def _merge(self, results: dict[tuple[str, str], dict]) -> None:
        with self.lock:
            for attempt in range(_WRITE_TRIES):
                composed = copy.deepcopy(self.env.metadata.get("composed", {}))
                for (form, arch), result in results.items():
                    stored = composed.setdefault(form, {}).get(arch, {})
                    if _rank(result) >= _rank(stored):
                        composed[form][arch] = {"started_at": stored.get("started_at"), **result}
                try:
                    self.env.update_metadata({**self.env.metadata, "composed": composed})
                    return
                except ConcurrentModificationError:
                    if attempt == _WRITE_TRIES - 1:
                        raise
                    self.env.metadata = Env.get(self.env.stored_id, self.env.version).metadata


class _Progress:
    """What a compose reports as it goes, to ``on_progress`` and the log: a line as each pair reaches a stage, one each
    ``_HEARTBEAT_S`` a stage goes on, and how each pair ended, with the time since the compose began."""

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
                 f"({_took(time.monotonic() - self.began)})", failed=result["status"] == "failed")

    def _beat(self) -> None:
        while not self.stopped.wait(_HEARTBEAT_S / 10):
            now = time.monotonic()
            with self.lock:
                due = [(pair, stage, since) for pair, (stage, since, said) in self.stages.items()
                       if now - said >= _HEARTBEAT_S]
                for pair, stage, since in due:
                    self.stages[pair] = (stage, since, now)
            for pair, stage, since in due:
                self.say(f"{pair}: still {stage} ({_took(now - since)})")


def _took(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} s"


def _rank(result: dict) -> int:
    """How far a pair got: ready, then built and either unprobed or failed at its probe, then anything short of an
    image. A result takes a stored one's place only when it got at least as far."""
    if result.get("status") == "ready":
        return 2
    return 1 if result.get("status") == "unverified" or result.get("stage") == "probe" else 0
