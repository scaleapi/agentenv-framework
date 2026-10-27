"""The composer: a computer env's base and driver built into one image for each form and arch with this machine's Docker,
tagged in the image store by a key of everything that goes into it, so an image of that key, built for any env, is reused.
Each result is recorded in the env's metadata as it finishes (``composed``), and the arches that built are written as
the env's computer, a ``computer`` artifact over an index of their images."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import io
import json
import logging
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact, push_local_image
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.config import get_config
from agent_env.env.envs.computer import gate
from agent_env.env.envs.computer.install import Install
from agent_env.providers.sandbox_providers.host_images import build_for, host_platform
from agent_env.store.ids import derive_id
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.store.image_store.registry_api import NoSuchImageError, pin_digest
from agent_env.utils.docker.build import DockerBuildError, build_image

if TYPE_CHECKING:
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.env.envs.computer_driver import ComputerDriverEnv

logger = logging.getLogger(__name__)

#: The image store repository composed computers are pushed to, tagged by their keys.
REPOSITORY = "agentenv-computers"
#: Where a driver's files go in a composed computer, in a folder named for the driver.
DRIVERS = "/opt/agentenv/drivers"
KEY_LABEL = "dev.agentenv.computer-key"
_PROTOCOL = "agentenv-framework-protocol"
# What building the protocol from its source reads, beside src/.
_PROTOCOL_FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


def compose(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv,
            pairs: list[tuple[str, str]] | None = None) -> None:
    """Build each of ``pairs``, by default every form and arch the env composes, in parallel, recording each as it
    starts and finishes, then write the env's computer over every pair built so far."""
    pairs = pairs if pairs is not None else [(form, arch) for form in env.forms for arch in gate.arches(base, driver)]
    record = _Record(env)
    record.started(pairs)
    with ThreadPoolExecutor(max_workers=max(len(pairs), 1)) as pool:
        futures = {pool.submit(_built, env, base, driver, form, arch): (form, arch) for form, arch in pairs}
        for future in as_completed(futures):
            form, arch = futures[future]
            result = future.result()
            if result["status"] == "failed":
                logger.warning(f"{env.id}: {form}/{arch} failed at {result['stage']}: {result['reason']}")
            else:
                logger.info(f"{env.id}: {form}/{arch} {'reused' if result['reused'] else 'built'}: {result['image']}")
            record.finished(form, arch, result)
    record.computer(_computer(env, base, record.built()))


def pending(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv) -> list[tuple[str, str]]:
    """The forms and arches of ``env`` not yet built: never started, failed, or left building by a put that stopped."""
    composed = env.metadata.get("composed", {})
    return [(form, arch) for form in env.forms for arch in gate.arches(base, driver)
            if composed.get(form, {}).get(arch, {}).get("status") not in ("unverified", "ready")]


def _built(env: ComputerEnv, base: ComputerArtifact, driver: ComputerDriverEnv, form: str, arch: str) -> dict:
    """The status of ``form`` and ``arch`` once built and pushed, or reused, or the stage that failed and why."""
    stage, key = "build", None
    try:
        with tempfile.TemporaryDirectory(prefix="agentenv-compose-") as work:
            context = Path(work)
            source, base_identity = _base(base, arch)
            layer = _stage_driver(context, driver, base)
            dockerfile = _dockerfile(source, layer, driver.install[base.os.type], base, _build_settings(env, base),
                                     _stage_protocol(context))
            key = "sha256:" + hashlib.sha256(json.dumps({
                "base": base_identity, "dockerfile": dockerfile, "context": _digest(context), "form": form,
                "arch": arch, "composer": _composer_digest(),
            }, sort_keys=True).encode()).hexdigest()
            store = get_config().get_image_store()
            store.ensure_repository(REPOSITORY)
            ref = store.image_ref(REPOSITORY, key.removeprefix("sha256:"))
            if (pinned := _served(ref, store)) is not None:
                return {"status": "unverified", "key": key, "image": pinned, "reused": True, "built_at": _now()}
            platform = f"linux/{arch}"
            emulated = platform != host_platform()
            if emulated and platform not in _emulated_platforms():
                return _failed(env, form, arch, "build", key, f"this machine can't build {platform}: its Docker has "
                               "no emulation for it")
            (context / "Dockerfile").write_text(f"{dockerfile}LABEL {KEY_LABEL}={key}\n")
            local = f"agentenv-computer:{key[7:23]}-{arch}"
            build_image(context / "Dockerfile", context, local, platform=platform, labels={KEY_LABEL: key})
            installed = _installed(local, layer)
            stage = "push"
            push_local_image(local, ref, store)
            return {"status": "unverified", "key": key, "image": pin_digest(ref, store.auth(ref)), "reused": False,
                    "emulated": emulated, "installed": installed, "built_at": _now()}
    except DockerBuildError as e:
        return _failed(env, form, arch, stage, key, "the build failed", log=str(e))
    except subprocess.CalledProcessError as e:
        return _failed(env, form, arch, stage, key, f"{' '.join(e.cmd[:2])} failed: {(e.stderr or '').strip()}")
    except (RuntimeError, ValueError, OSError) as e:
        return _failed(env, form, arch, stage, key, f"{type(e).__name__}: {e}")


def _base(base: ComputerArtifact, arch: str) -> tuple[str, str]:
    """What the composed image is built ``FROM`` for ``arch``, and the base's part of the key: a registry image pinned to
    its digest, or one built here from the base's stored build context for that arch."""
    image = base.image()
    if names_registry(image.image_name):
        store = get_config().get_image_store()
        pinned = image.pinned_name if "@" in image.pinned_name else pin_digest(image.pinned_name, store.auth(image.pinned_name))
        return pinned, pinned
    return build_for(image, f"linux/{arch}"), f"sources:{image.source_digest}"


def _stage_driver(context: Path, driver: ComputerDriverEnv, base: ComputerArtifact) -> str:
    """Write the driver's install files under ``context/driver/`` and its supervisord programs beside them; the name
    of its folder under ``DRIVERS``, its id's last part."""
    install = driver.install[base.os.type]
    layer = re.sub(r"[^a-z0-9-]+", "-", driver.id.rsplit("/", 1)[-1].lower()).strip("-") or "driver"
    folder = PurePosixPath(install.files)
    for name, artifact in driver.files.get_file_artifacts().items():
        path = PurePosixPath(name)
        if folder == PurePosixPath(".") or path.is_relative_to(folder):
            target = context / "driver" / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(artifact.load())
    (context / "driver").mkdir(exist_ok=True)
    (context / f"agentenv-{layer}.conf").write_text(_programs(layer, install, driver.port))
    return layer


def _programs(layer: str, install: Install, port: int) -> str:
    prefix = f"{DRIVERS}/{layer}"
    programs = []
    for service in install.services:
        program, _, rest = service.command.partition(" ")
        if "/" in program and not program.startswith("/"):
            program = f"{prefix}/{program}"
        programs.append(f"[program:{layer}-{service.name}]\ncommand={program} {rest}".rstrip() + "\n"
                        f"directory={prefix}\nenvironment=MCP_PORT=\"{port}\"\nautorestart=true\nredirect_stderr=true\n"
                        f"stdout_logfile=/var/log/agentenv/{layer}-{service.name}.log\n")
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
    values = env.settings.get(base.id, {})
    return {name: values[name] for name, setting in base.config.items() if setting.phase == "build" and name in values}


def _dockerfile(source: str, layer: str, install: Install, base: ComputerArtifact, build_settings: dict,
                protocol: bool) -> str:
    """The composed image's Dockerfile but its key's label, which the key is a digest of."""
    prefix = f"{DRIVERS}/{layer}"
    lines = [f"FROM {source}", "USER root", f"COPY driver/ {prefix}/"]
    if install.python is not None:
        packages = f"/tmp/agentenv-protocol -r {prefix}/{install.python}" if protocol else f"-r {prefix}/{install.python}"
        if protocol:
            lines.append("COPY protocol /tmp/agentenv-protocol")
        lines.append(f"RUN python3 -m venv {prefix}/venv && {prefix}/venv/bin/pip install --no-cache-dir {packages} "
                     f"&& {prefix}/venv/bin/pip list --format=json > {prefix}/installed.json"
                     + (" && rm -rf /tmp/agentenv-protocol" if protocol else ""))
    lines.append(f"COPY agentenv-{layer}.conf /etc/supervisor/conf.d/agentenv-{layer}.conf")
    checks = [f"test -d {prefix}", f"test -f /etc/supervisor/conf.d/agentenv-{layer}.conf"]
    if install.python is not None:
        checks.append(f"test -x {prefix}/venv/bin/python")
    lines.append(f"RUN {' && '.join(checks)}")
    lines += [f"ENV AGENTENV_SETTING_{name.upper()}={json.dumps(str(value))}" for name, value in build_settings.items()]
    lines.append(f"USER {base.session.user}")
    return "\n".join(lines) + "\n"


def _installed(local: str, layer: str) -> dict[str, str]:
    """The packages ``local``'s build installed into the driver's venv, by name, read from the image without running it,
    so on any arch; none for a driver with no venv."""
    created = subprocess.run(["docker", "create", local], capture_output=True, text=True, check=True).stdout.strip()
    try:
        copied = subprocess.run(["docker", "cp", f"{created}:{DRIVERS}/{layer}/installed.json", "-"],
                                capture_output=True)
    finally:
        subprocess.run(["docker", "rm", created], capture_output=True)
    if copied.returncode != 0:
        return {}
    with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as archive:
        listed = json.loads(archive.extractfile(archive.getmembers()[0]).read())
    return {package["name"]: package["version"] for package in listed}


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
        stored = FileArtifact.put_bytes(derive_id(env.id, f"build_log_{form}_{arch}"), description=(
            f"the {form}/{arch} build of {env.id} v{env.version}, which failed"), filename="build.log",
            content=log.encode(), content_type="text/plain")
        result["log"] = {"id": stored.id, "version": stored.version}
        result["reason"] = f"{reason}: {log.strip().splitlines()[-1] if log.strip() else 'no output'}"
    return result


def _computer(env: ComputerEnv, base: ComputerArtifact, built: dict[str, str]) -> dict | None:
    """Write the env's computer over the images ``built``, by arch, joined into one index: its arches those, its OS and
    session the base's, its settings the base's with the env's values as their defaults. None when nothing built."""
    if not built:
        return None
    arches = [arch for arch in base.arch if arch in built]
    store = get_config().get_image_store()
    index = store.image_ref(REPOSITORY, "index-" + hashlib.sha256(
        ",".join(built[arch] for arch in arches).encode()).hexdigest())
    _joined(index, [built[arch] for arch in arches])
    image = DockerImageArtifact.put_ref(derive_id(env.id, "computer_image"), description=(
        f"{env.id}'s composed computer"), image_name=index)
    config = {name: setting.model_dump() for name, setting in base.config.items()}
    for name, value in env.settings.get(base.id, {}).items():
        config[name]["default"] = value
    if env.display is not None:
        config["resolution"]["default"] = f"{env.display['width']}x{env.display['height']}"
    computer = ComputerArtifact.put(
        id=derive_id(env.id, "computer"), description=f"{base.id} composed with {env.driver['id']} for {env.id}",
        arch=arches, os=base.os.model_dump(), session=base.session.model_dump(), config=config,
        forms={"container": {"image": {"id": image.id, "version": image.version}, "init": base.forms.container.init,
                             "input_channel": base.forms.container.input_channel}})
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
    """The env's ``composed`` metadata, by form and arch, written by compare-and-swap as each pair starts and finishes,
    one write at a time."""

    def __init__(self, env: ComputerEnv):
        self.env, self.lock = env, threading.Lock()
        self.composed = copy.deepcopy(env.metadata.get("composed", {}))

    def started(self, pairs: list[tuple[str, str]]) -> None:
        with self.lock:
            for form, arch in pairs:
                self.composed.setdefault(form, {})[arch] = {"status": "building", "started_at": _now()}
            self.env.merge_metadata({"composed": copy.deepcopy(self.composed)})

    def finished(self, form: str, arch: str, result: dict) -> None:
        with self.lock:
            started = self.composed.get(form, {}).get(arch, {}).get("started_at")
            self.composed.setdefault(form, {})[arch] = {**result, "started_at": started}
            self.env.merge_metadata({"composed": copy.deepcopy(self.composed)})

    def built(self) -> dict[str, str]:
        """Each arch's image of the container form, of the pairs built or reused."""
        return {arch: result["image"] for arch, result in self.composed.get("container", {}).items()
                if result.get("status") in ("unverified", "ready")}

    def computer(self, ref: dict | None) -> None:
        if ref is not None:
            with self.lock:
                self.env.merge_metadata({"computer": ref})
