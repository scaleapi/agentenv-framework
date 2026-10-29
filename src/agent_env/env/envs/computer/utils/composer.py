"""The composer: a computer env's base, driver and apps built into one image for each form and arch with this machine's
Docker, tagged in the env's image store by a key of everything that goes into it, so an image of that key, built for any
env, is reused.
Each result is recorded in the env's metadata as it finishes (``composed``), and the arches that built are written as
the env's computer, a ``computer`` artifact over an index of their images."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import io
import json
import logging
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agentenv_protocol.computer import apps as installed_apps
from agentenv_protocol.computer.app import SOCKET
from agentenv_protocol.computer.conformance import run_probe

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact, docker_login, push_local_image
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.config import get_config
from agent_env.entity.export import Unwritable, checked_path
from agent_env.env.env import Env
from agent_env.env.envs.computer.utils import boot, gate
from agent_env.env.envs.computer.utils.install import Install
from agent_env.providers.sandbox_providers.host_images import build_for, host_platform, platform_tag
from agent_env.store.base import ConcurrentModificationError
from agent_env.store.ids import derive_id
from agent_env.store.image_store.image_contents import labels
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.store.image_store.registry_api import NoSuchImageError, pin_digest
from agent_env.utils.docker.build import DockerBuildError, build_image

if TYPE_CHECKING:
    from agentenv_protocol.computer.conformance import ProbeReport

    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.env.envs.computer.app_env import ComputerAppEnv
    from agent_env.env.envs.computer.driver_env import ComputerDriverEnv

logger = logging.getLogger(__name__)

#: The image store repository composed computers are pushed to, tagged by their keys.
REPOSITORY = "agentenv-computers"
#: Where a driver's files go in a composed computer, in a folder named for the driver; an app's, under APPS.
DRIVERS = "/opt/agentenv/drivers"
APPS = str(installed_apps.APPS)
KEY_LABEL = "dev.agentenv.computer-key"
# What a build installed and whether it was emulated, kept on its image for an env that reuses it.
INSTALLED_LABEL, EMULATED_LABEL = "dev.agentenv.installed", "dev.agentenv.emulated"
# What a file starts with when it's only ever run: a script or a Linux binary, made executable in the image, since the
# store keeps no file modes.
_RUN_ONLY = (b"#!", b"\x7fELF")
_EXECUTABLE = 0o755
# A pair's statuses once its image is in the store.
_BUILT = ("unverified", "ready")
_WRITE_TRIES = 4
#: How long a probe's computer may run before its sandbox provider stops it.
_PROBE_TTL_S = 1800
#: How long the probe at full may take on a computer that booted: about ten seconds on a healthy one.
_PROBE_TIMEOUT_S = 120
_PROTOCOL = "agentenv-framework-protocol"
# What building the protocol from its source reads, beside src/.
_PROTOCOL_FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


def compose(env: ComputerEnv, pairs: list[tuple[str, str]] | None = None,
            on_build: Callable[[], None] | None = None) -> None:
    """Build each of ``pairs``, by default each form the base has in each arch the env composes, in parallel, and probe
    each one a configured sandbox provider runs natively (gate 3), recording each as it starts and finishes, then write
    the env's computer over every pair built so far. ``on_build`` is called once, just before the first image is built
    here; reusing them all calls it never."""
    base, driver = env.computer, env.driver
    building = _once(on_build)
    pairs = pairs if pairs is not None else [(form, arch) for form in gate.forms(base)
                                             for arch in gate.arches(base, driver, env.composed_apps())]
    record = _Record(env)
    record.started(pairs)
    with ThreadPoolExecutor(max_workers=max(len(pairs), 1)) as pool:
        futures = {pool.submit(_composed, env, base, driver, form, arch, building): (form, arch) for form, arch in pairs}
        for future in as_completed(futures):
            form, arch = futures[future]
            result = future.result()
            if result["status"] == "failed":
                logger.warning(f"{env.stored_id}: {form}/{arch} failed at {result['stage']}: {result['reason']}")
            else:
                logger.info(f"{env.stored_id}: {form}/{arch} {'reused' if result['reused'] else 'built'}, "
                            f"{result['status']}: {result['image']}")
            record.finished(form, arch, result)
    record.computer(_computer(env, base, record.built()))


def current(env: ComputerEnv) -> bool:
    """Whether the env's computer was written over every pair built, as a put or env build that failed writing it
    leaves it not."""
    return "computer" in env.metadata and env.metadata["computer"].get("images") == _Record(env).built()


def pending(env: ComputerEnv) -> list[tuple[str, str]]:
    """The forms and arches of ``env`` left to compose: never started, failed, left building by a put that stopped, or
    built but unprobed though a configured sandbox provider runs the arch natively."""
    from agent_env.providers import get_env_sandbox_provider

    composed, provider = env.metadata.get("composed", {}), get_env_sandbox_provider()
    statuses = {(form, arch): composed.get(form, {}).get(arch, {}).get("status")
                for form in gate.forms(env.computer)
                for arch in gate.arches(env.computer, env.driver, env.composed_apps())}
    return [(form, arch) for (form, arch), status in statuses.items()
            if status not in _BUILT or (status == "unverified" and boot.native_links(provider, arch))]


def _composed(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, form: str, arch: str,
              building: Callable[[], None]) -> dict:
    """``form`` and ``arch`` built or reused, then probed when it built (gate 3)."""
    result = _built(env, base, driver, form, arch, building)
    return _probed(env, driver, form, arch, result) if result["status"] == "unverified" else result


def _probed(env: ComputerEnv, driver: ComputerDriverEnv, form: str, arch: str, result: dict) -> dict:
    """Gate 3: the build of ``result`` booted on the first configured sandbox provider that runs ``arch`` natively, and
    probed at ``full``: ready, with the driver's card, or failed at the probe; each with a summary of the probe and its
    report stored. Still unverified, saying why, when no configured provider runs ``arch`` natively."""
    from agent_env.providers import get_env_sandbox_provider

    if not (links := boot.native_links(get_env_sandbox_provider(), arch)):
        return {**result, "reason": f"no configured sandbox provider runs linux/{arch} natively to probe it"}
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
    """``image`` booted at the env's settings, as a deploy boots it, on the first of ``links`` that boots it, probed at
    ``full`` within ``_PROBE_TIMEOUT_S`` and torn down; the report, the driver's card and the provider's name."""
    environment = boot.settings_environment(env.computer, _values(env), environment_name=env.environment_name)
    problems = []
    for link in links:
        try:
            booted = await boot.boot(link, image=image, port=driver.port, environment=environment,
                                     network_policy=None, timeout=_PROBE_TTL_S, attribution=None,
                                     ready_s=boot.ready_timeout(env.composed_apps()))
        except Exception as e:
            problems.append(f"{type(link).__name__}: {e}")
            continue
        try:
            async with asyncio.timeout(_PROBE_TIMEOUT_S):
                return await run_probe(booted.url, "full"), booted.card, type(link).__name__
        finally:
            await boot.discard(booted.sandbox)
    raise RuntimeError("; ".join(problems))


def _built(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, form: str, arch: str,
           building: Callable[[], None]) -> dict:
    """The status of ``form`` and ``arch`` once built and pushed, or reused, or the stage that failed and why;
    ``building`` is called before anything builds."""
    stage, key = "build", None
    try:
        with tempfile.TemporaryDirectory(prefix="agentenv-compose-") as work:
            context = Path(work)
            source, base_identity = _base(base, arch)
            layer = _stage_driver(context, driver, base)
            apps = [_stage_app(context, app, base) for app in env.composed_apps()]
            dockerfile = _dockerfile(source, [layer, *apps], base, _build_settings(env, base), _stage_protocol(context))
            key = "sha256:" + hashlib.sha256(json.dumps({
                "base": base_identity, "dockerfile": dockerfile, "context": _digest(context), "form": form,
                "arch": arch, "composer": _composer_digest(),
            }, sort_keys=True).encode()).hexdigest()
            store = get_config().get_image_store_for(env.stored_id)
            store.ensure_repository(REPOSITORY)
            ref = store.image_ref(REPOSITORY, key.removeprefix("sha256:"))
            if (pinned := _served(ref, store)) is not None:
                kept = labels(pinned, arch, store.auth(pinned))
                return {"status": "unverified", "key": key, "image": pinned, "reused": True,
                        "emulated": kept.get(EMULATED_LABEL) == "true",
                        "installed": json.loads(kept.get(INSTALLED_LABEL, "{}")), "built_at": _now()}
            platform = f"linux/{arch}"
            emulated = platform != host_platform()
            if emulated and platform not in _emulated_platforms():
                return _failed(env, form, arch, "build", key, f"this machine can't build {platform}: its Docker has "
                               "no emulation for it")
            building()
            _ready_base(base, arch)
            (context / "Dockerfile").write_text(f"{dockerfile}LABEL {KEY_LABEL}={key}\n")
            local = f"agentenv-computer:{key[7:23]}-{arch}"
            build_image(context / "Dockerfile", context, local, platform=platform, labels={KEY_LABEL: key})
            installed = _installed(local, layer.prefix)
            _label(local, platform, {INSTALLED_LABEL: json.dumps(installed, sort_keys=True),
                                     EMULATED_LABEL: "true" if emulated else "false"})
            stage = "push"
            push_local_image(local, ref, store)
            return {"status": "unverified", "key": key, "image": pin_digest(ref, store.auth(ref)), "reused": False,
                    "emulated": emulated, "installed": installed, "built_at": _now()}
    except DockerBuildError as e:
        # Docker can list a platform it has lost the emulation for, until it restarts; the build says so.
        unemulated = "exec format error" in str(e)
        return _failed(env, form, arch, stage, key, f"this machine can't run linux/{arch} programs: its Docker has no "
                       "emulation for it" if unemulated else "the build failed", log=str(e))
    except subprocess.CalledProcessError as e:
        return _failed(env, form, arch, stage, key, f"{' '.join(e.cmd[:2])} failed: {(e.stderr or '').strip()}")
    except (RuntimeError, ValueError, OSError) as e:
        return _failed(env, form, arch, stage, key, f"{type(e).__name__}: {e}")


def _once(fn: Callable[[], None] | None) -> Callable[[], None]:
    """``fn``, called at most once however many builds call it at once; nothing when it's None."""
    lock, called = threading.Lock(), []

    def once() -> None:
        with lock:
            if fn is not None and not called:
                called.append(True)
                fn()
    return once


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


def _ready_base(base: ComputerArtifact, arch: str) -> None:
    """Have this machine's Docker able to build ``FROM`` the base for ``arch``: signed in to its registry with the image
    store's credentials for it, or holding it built from its stored build context."""
    image = base.image()
    if names_registry(image.image_name):
        docker_login(get_config().get_image_store().auth(image.pinned_name))
    else:
        build_for(image, f"linux/{arch}")


@dataclass(frozen=True)
class _Layer:
    """A driver's or an app's part of a build: its folder in the build context, its prefix in the image, what it
    installs, and the file of its supervisord programs. An app's venv also has the protocol's ``app`` extra, for the
    app's own server, which listens in a folder of the app's the session user owns."""

    folder: str
    prefix: str
    install: Install
    conf: str
    app: bool = False


def _stage_driver(context: Path, driver: ComputerDriverEnv, base: ComputerArtifact) -> _Layer:
    """Write the driver's install files under ``context/driver/`` and its supervisord programs beside them, served on
    its port, its folder under ``DRIVERS`` named for its id's last part."""
    install, name = driver.install[base.os.type], _name(driver.id, "driver")
    layer = _Layer("driver", f"{DRIVERS}/{name}", install, f"agentenv-{name}.conf")
    _stage(context, layer, f"driver {driver.id}", driver.files, _services(name, install), {"MCP_PORT": driver.port})
    return layer


def _stage_app(context: Path, app: ComputerAppEnv, base: ComputerArtifact) -> _Layer:
    """Write the app's install files under ``context/apps/<name>/`` and its supervisord programs beside them, run under
    its name: its services and its own server, which serves its class from its venv and launches the app. Its folder
    under ``APPS`` is named for its id's last part."""
    install, name = app.install[base.os.type], _name(app.id, "app")
    layer = _Layer(f"apps/{name}", f"{APPS}/{name}", install, f"agentenv-app-{name}.conf", app=True)
    file, _, cls = app.module.partition(":")
    served = shlex.quote(f"{PurePosixPath(file).relative_to(install.files)}:{cls}")
    programs = {f"app-{name}": f"venv/bin/python -m agentenv_protocol.computer.serve_app {served}",
                **_services(f"app-{name}", install)}
    _stage(context, layer, f"app {app.id}", app.files, programs, {"ENVIRONMENT_NAME": app.environment_name})
    return layer


def _name(env_id: str, fallback: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", env_id.rsplit("/", 1)[-1].lower()).strip("-") or fallback


def _services(name: str, install: Install) -> dict[str, str]:
    """The commands of ``install``'s services by their programs' names, each named for ``name``."""
    return {f"{name}-{service.name}": service.command for service in install.services}


def _stage(context: Path, layer: _Layer, what: str, files: FileArtifactUniverse, programs: dict[str, str],
           environment: dict[str, Any]) -> None:
    """Write the ``files`` of ``what``'s folder its install names to ``layer``'s folder, a script or binary executable,
    and ``programs``, commands by name, as its supervisord programs, with ``environment``."""
    folder = PurePosixPath(layer.install.files)
    for name, artifact in files.get_file_artifacts().items():
        try:
            path = PurePosixPath(checked_path(name))
        except Unwritable as e:
            raise ValueError(f"{what}: {e}") from None
        if folder == PurePosixPath(".") or path.is_relative_to(folder):
            target = context / layer.folder / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data := artifact.load())
            if data.startswith(_RUN_ONLY):
                target.chmod(_EXECUTABLE)
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


def _dockerfile(source: str, layers: list[_Layer], base: ComputerArtifact, build_settings: dict, protocol: bool) -> str:
    """The composed image's Dockerfile but its key's label, which the key is a digest of: each layer's files under its
    prefix, its venv beside them, with this checkout's protocol when ``protocol``, and its programs; an app's server's
    folder owned by the session user."""
    venvs = [layer for layer in layers if layer.install.python is not None or layer.app]
    lines = [f"FROM {source}", "USER root", *(["COPY protocol /tmp/agentenv-protocol"] if protocol and venvs else [])]
    for layer in layers:
        prefix = layer.prefix
        lines.append(f"COPY {layer.folder}/ {prefix}/")
        if layer in venvs:
            lines.append(f"RUN python3 -m venv {prefix}/venv && {prefix}/venv/bin/pip install --no-cache-dir "
                         f"{_packages(layer, protocol)} && {prefix}/venv/bin/pip list --format=json > "
                         f"{prefix}/installed.json")
        if layer.app:
            lines.append(f"RUN mkdir -p {prefix}/{SOCKET.parent} && chown {base.session.user} {prefix}/{SOCKET.parent}")
        lines.append(f"COPY {layer.conf} /etc/supervisor/conf.d/{layer.conf}")
        checks = [f"test -d {prefix}", f"test -f /etc/supervisor/conf.d/{layer.conf}"]
        if layer in venvs:
            checks.append(f"test -x {prefix}/venv/bin/python")
        lines.append(f"RUN {' && '.join(checks)}")
    if protocol and venvs:
        lines.append("RUN rm -rf /tmp/agentenv-protocol")
    lines += [f"ENV AGENTENV_SETTING_{name.upper()}={json.dumps(str(value))}" for name, value in build_settings.items()]
    lines.append(f"USER {base.session.user}")
    return "\n".join(lines) + "\n"


def _packages(layer: _Layer, protocol: bool) -> str:
    """What ``layer``'s venv installs: its requirements and, for an app, the protocol's ``app`` extra at this release,
    which a driver's requirements name themselves; the protocol from this checkout's source when ``protocol``."""
    extra = "[app]" if layer.app else ""
    protocols = ([f"/tmp/agentenv-protocol{extra}"] if protocol else
                 [f"{_PROTOCOL}{extra}=={importlib.metadata.version(_PROTOCOL)}"] if layer.app else [])
    requirements = ["-r", f"{layer.prefix}/{layer.install.python}"] if layer.install.python is not None else []
    return shlex.join([*protocols, *requirements])


def _installed(local: str, prefix: str) -> dict[str, str]:
    """The packages ``local``'s build installed into the venv of the driver at ``prefix``, by name, read from the image
    without running it, so on any arch; none for a driver with no venv."""
    created = subprocess.run(["docker", "create", local], capture_output=True, text=True, check=True).stdout.strip()
    try:
        copied = subprocess.run(["docker", "cp", f"{created}:{prefix}/installed.json", "-"],
                                capture_output=True)
    finally:
        subprocess.run(["docker", "rm", created], capture_output=True)
    if copied.returncode != 0:
        return {}
    with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as archive:
        listed = json.loads(archive.extractfile(archive.getmembers()[0]).read())
    return {package["name"]: package["version"] for package in listed}


def _label(local: str, platform: str, extra: dict[str, str]) -> None:
    """Add ``extra``'s labels to this machine's image ``local``, which adds no layer."""
    with tempfile.TemporaryDirectory(prefix="agentenv-label-") as work:
        dockerfile = Path(work) / "Dockerfile"
        dockerfile.write_text(f"FROM {local}\n")
        build_image(dockerfile, Path(work), local, platform=platform, labels=extra)


def _served(ref: str, store) -> str | None:
    """``ref`` pinned to the digest the image store serves for it, or None when it serves none."""
    try:
        return pin_digest(ref, store.auth(ref))
    except NoSuchImageError:
        return None


def _emulated_platforms() -> set[str]:
    """The platforms this machine's Docker builds, its own and those it emulates."""
    inspected = subprocess.run(["docker", "buildx", "inspect", "--bootstrap"], capture_output=True, text=True)
    platforms = set()
    for line in inspected.stdout.splitlines():
        if line.strip().startswith("Platforms:"):
            platforms |= {item.strip().rstrip("*") for item in line.split(":", 1)[1].split(",")}
    return platforms


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
    docker_login(store.auth(index))
    _joined(index, [built[arch] for arch in arches])
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


def _joined(index: str, images: list[str]) -> None:
    """Push one index of ``images`` to ``index``, as the release joins its platforms' images."""
    joined = subprocess.run(["docker", "buildx", "imagetools", "create", "-t", index, *images],
                            capture_output=True, text=True)
    if joined.returncode != 0:
        raise RuntimeError(f"joining {', '.join(images)} into {index} failed: {joined.stderr.strip()}")


def _digest(folder: Path) -> str:
    """A digest of the files under ``folder``, by their paths and bytes."""
    hashed = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        hashed.update(path.relative_to(folder).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + hashed.hexdigest()


def _composer_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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


def _rank(result: dict) -> int:
    """How far a pair got: ready, then built and either unprobed or failed at its probe, then anything short of an
    image. A result takes a stored one's place only when it got at least as far."""
    if result.get("status") == "ready":
        return 2
    return 1 if result.get("status") == "unverified" or result.get("stage") == "probe" else 0
