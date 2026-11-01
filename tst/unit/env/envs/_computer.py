"""What the computer env tests share: the fake Docker and registry the composer builds with, a fake sandbox provider
whose VMs build, and a base, a driver and an env of them."""

import gzip
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
import urllib.parse

from agentenv_protocol.computer import AppDeclaration, DriverDeclaration
from agentenv_protocol.computer.conformance import CheckResult, ProbeReport

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.config import get_config
from agent_env.env.envs.computer.utils import builder, layers_in_image
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.computer.utils.install import Install, Service
from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.utils.docker.build import DockerBuildError
from tst.util.oci_registry import UBUNTU_LAYER, Registry

OS = {"type": "linux", "distribution": "ubuntu", "version": "24.04", "display": "x11"}


CONFIG = {"resolution": {"default": "1280x800", "phase": "deploy"}, "timezone": {"default": "UTC", "phase": "deploy"},
          "theme": {"default": "light", "phase": "build"}}


SERVE = "venv/bin/python -m agentenv_protocol.computer.serve driver.py:Desk"


CARD_EXTENSION = "urn:agentenv:computer/v1"


CARD = {"name": "desktop", "capabilities": {"extensions": [{"uri": CARD_EXTENSION}]}}


TOOLS = ["computer_screenshot", "computer_click", "computer_type", "computer_pointer_list"]


class FakeDocker:
    """This machine's Docker for the composer, on arm64 emulating amd64: each build recorded with its Dockerfile, its
    context's files and its programs, each push landing in the fake registry as an image of its arch, and each probe
    recorded by the arch and image it booted."""

    def __init__(self, registry: Registry):
        self.registry, self.builds, self.fail, self.emulates, self.labels = registry, [], {}, {"linux/amd64"}, {}
        self.probed, self.probe_fails, self.resolved, self.timeouts = [], {}, [], {}
        self.writes, self.saved = {}, []

    def resolve(self, request, context, source, arch, work):
        """The resolver in the base: each apt package as one .deb, each venv's requirements as one wheel."""
        self.resolved.append((arch, request))
        out, result = work / "out", {"apt": [], "wheels": {}}
        out.mkdir(parents=True, exist_ok=True)
        for name in (re.split("[=/]", wanted)[0] for wanted in request["apt"]):
            (out / "apt").mkdir(parents=True, exist_ok=True)
            (out / "apt" / f"{name}_1.0_{arch}.deb").write_bytes(f"{name} for {arch}".encode())
            result["apt"].append({"file": f"{name}_1.0_{arch}.deb", "package": name, "version": "1.0", "arch": arch})
        for layer in request["python"]:
            requirements = context / layer["folder"].removeprefix("/ctx/") / (layer["requirements"] or "-")
            if not (requirements.is_file() and requirements.read_text().strip()) and not layer["add"]:
                result["wheels"][layer["name"]] = []
                continue
            (out / "wheels" / layer["name"]).mkdir(parents=True)
            (out / "wheels" / layer["name"] / "mss-10.1.0-py3-none-any.whl").write_bytes(b"mss")
            result["wheels"][layer["name"]] = ["mss-10.1.0-py3-none-any.whl"]
        return result

    def resolving(self, resolve=None):
        """A LocalBuilder.resolve that runs ``resolve``, by default this one's, on the builder's context and resolve."""
        def resolving(on, request, source, arch, timeout):
            self.timeouts[arch] = timeout
            return (resolve or self.resolve)(request, on.work / builder.CONTEXT, source, arch, on.work / builder.RESOLVED)
        return resolving

    async def probe(self, links, env, driver, image):
        arch = self._arch(self.registry.manifests[image.rpartition("@")[2]]["config"]["digest"])
        self.probed.append((arch, image))
        if arch in self.probe_fails:
            checks = [CheckResult(check="P4", name="clicks show their effect", status="failed", detail=self.probe_fails[arch])]
        else:
            checks = [CheckResult(check="P0", name="the card", status="passed", extensions=[CARD_EXTENSION])]
        return ProbeReport.of("full", "desk", checks, [CARD_EXTENSION]), CARD, "LocalSandboxProvider"

    def build_image(self, dockerfile, context, tag, *, platform, labels=None, build_args=None):
        if context.name == "label":
            self.labels[tag] = labels
            return
        if platform in self.fail:
            raise DockerBuildError(self.fail[platform])
        self.builds.append({"platform": platform, "tag": tag, "dockerfile": dockerfile.read_text(),
                            "files": sorted(p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()),
                            "conf": next(context.glob("agentenv-*.conf")).read_text(),
                            "confs": {conf.name: conf.read_text() for conf in context.glob("agentenv-*.conf")},
                            "build_settings": {path.parent.parent.name: json.loads(path.read_text())
                                               for path in context.glob("apps/*/.agentenv/build_settings.json")}})

    def save(self, tag, path):
        """``tag`` saved as ``docker save`` does: a base layer with no history, then a layer for each COPY and RUN of its
        Dockerfile, a RUN's holding the files ``writes`` gives for each key its command has."""
        self.saved.append(tag)
        dockerfile = next(build["dockerfile"] for build in reversed(self.builds) if build["tag"] == tag)
        layers = [(None, {"etc/os-release": b"ubuntu"})]
        for line in dockerfile.splitlines():
            instruction, _, rest = line.partition(" ")
            command = rest.removeprefix("--network=none ")
            if instruction == "RUN":
                layers.append((f"RUN /bin/sh -c {command} # buildkit", {
                    path: content for key, files in self.writes.items() if key in command for path, content in files.items()}))
            elif instruction == "COPY":
                layers.append((f"{line} # buildkit", {}))
            elif instruction != "FROM":
                layers.append((line, None))
        saved_image(path, layers)

    def push(self, local, ref, store):
        self.registry.tags[ref] = self.registry.image(UBUNTU_LAYER, arch=local.rsplit("-", 1)[1],
                                                      labels=self.labels.get(local))

    def joined(self, index, images, auth):
        self.registry.tags[index] = self.registry.index(**{
            platform.rsplit("/", 1)[1]: image.rpartition("@")[2] for platform, image in images.items()})

    def _arch(self, config_digest):
        return json.loads(self.registry.blobs[config_digest])["architecture"]


def saved_image(path, layers):
    """A tar of an image as ``docker save`` writes one: each of ``layers``, bottom first, its history's ``created_by``
    (None for a layer with no entry) and its files by path, a file's content as bytes, or with its mode as (bytes, mode),
    and a link's target as text, or None for an entry that made no layer; the last layer gzipped, as some savers keep
    them."""
    blobs, history, made = {}, [], []
    for i, (created_by, files) in enumerate(layers):
        if created_by is not None:
            history.append({"created_by": created_by, **({"empty_layer": True} if files is None else {})})
        if files is None:
            continue
        layer = io.BytesIO()
        with tarfile.open(fileobj=layer, mode="w") as tar:
            for name, content in files.items():
                entry = tarfile.TarInfo(name)
                if isinstance(content, str):
                    entry.type, entry.linkname = tarfile.SYMTYPE, content
                    tar.addfile(entry)
                else:
                    data, entry.mode = content if isinstance(content, tuple) else (content, 0o644)
                    entry.size = len(data)
                    tar.addfile(entry, io.BytesIO(data))
        data = gzip.compress(layer.getvalue()) if i == len(layers) - 1 else layer.getvalue()
        made.append(digest := f"blobs/sha256/{hashlib.sha256(data).hexdigest()}")
        blobs[digest] = data
    config = json.dumps({"history": history}).encode()
    blobs[config_path := f"blobs/sha256/{hashlib.sha256(config).hexdigest()}"] = config
    blobs["manifest.json"] = json.dumps([{"Config": config_path, "Layers": made}]).encode()
    with tarfile.open(path, mode="w") as tar:
        for name, data in blobs.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))


def _base(image, **changes):
    return ComputerArtifact.put(inspect=False, **{
        "id": "@local/~/kit/ubuntu-24.04", "arch": ["amd64", "arm64"], "os": OS, "session": {"user": "user"},
        "config": CONFIG, "forms": {"container": {"image": {"id": image.id, "version": image.version},
                                                  "init": "supervisord", "input_channel": "env"}}, **changes})


def _driver(folder, requirements="mss>=9,<11\n", supports=None, services=None, install=True, extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "driver.py").write_text("class Desk: ...\n")
    (folder / "requirements.txt").write_text(requirements)
    files = FileArtifactUniverse.put_bundled("@local/~/kit/desk__files", files={
        name: folder / name for name in ("driver.py", "requirements.txt")})
    if extra:
        files = FileArtifactUniverse.put(files.id, file_artifacts={**files.get_file_artifacts(), **extra})
    supports = supports or [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}]
    platforms = [{"platform": {"os": "linux", "arch": arch, "display": "x11"}, "provides": [], "tools": TOOLS,
                  "features": {}} for arch in ("amd64", "arm64")]
    services = services or [Service(name="driver", command=SERVE)]
    return ComputerDriverEnv.put(
        id="@local/~/kit/desk", module="driver.py:Desk", files=files,
        declaration=DriverDeclaration.model_validate({"name": "desk", "supports": supports, "platforms": platforms}),
        install={"linux": Install(python="requirements.txt", services=services)} if install else {}, port=18765)


def _put(base, driver, compose=True, **changes):
    return ComputerEnv.put(**{"id": "@local/~/kit/desktop", "computer": base, "driver": driver,
                              "environment_name": "desktop", "compose": compose, **changes})


class FakeVm:
    """A builder VM, amd64: its files under ``root``, each line of a script it's given run by bash here, but for Docker
    and uploads to the store, which the fake's Docker and the store do, and the helper's subcommands that use Docker."""

    platform, _WFT_CHUNK_BYTES = "linux/amd64", 96 * 1024

    def __init__(self, provider, root):
        self.provider, self.root, self.terminated = provider, root, False

    def at(self, path):
        return self.root / str(path).lstrip("/")

    async def write_host_file(self, data, path):
        self.at(path).parent.mkdir(parents=True, exist_ok=True)
        self.at(path).write_bytes(data)

    async def load_object_file(self, url, path):
        self.provider.fetched.append(url)
        self.at(path).parent.mkdir(parents=True, exist_ok=True)
        get_config().get_object_store_at(url).download_to_file(url, str(self.at(path)))

    async def exec_script(self, script, max_retries=0):
        code, out, _ = await self.exec_with_output("sudo", "bash", "-c", script)
        if code:
            raise RuntimeError(f"Script failed (exit {code}):\nstdout: {out}")
        return out

    async def exec_with_output(self, *argv):
        printed = []
        for line in argv[-1].splitlines():
            if line.strip() and line != "set -e":
                code, out = self._line(line)
                printed.append(out)
                if code:
                    return code, "".join(printed), ""
        return 0, "".join(printed), ""

    async def terminate(self):
        self.terminated = True

    def _line(self, line):
        self.provider.ran.append(line)
        words = shlex.split(line)
        if builder.IN_VM.name in line and words[2] in ("buildx", "emulate", "installed"):
            if words[2] == "emulate":
                self.provider.binfmt.append(words[3])
            return 0, json.dumps([{"name": "mss", "version": "10.1.0"}]) if words[2] == "installed" else ""
        if "docker" in words or "curl" in words:
            return self._docker(words)
        ran = subprocess.run(["bash", "-c", line.replace("/tmp/agentenv-compose-", f"{self.root}/tmp/agentenv-compose-")],
                             capture_output=True, text=True)
        return ran.returncode, ran.stdout + ran.stderr

    def _docker(self, words):
        docker, cwd, printed = self.provider.docker, None, ""
        while words:
            command, words = (words[:words.index("&&")], words[words.index("&&") + 1:]) if "&&" in words else (words, [])
            if command[0] == "cd":
                cwd = self.at(command[1])
            elif "login" in command:
                self.provider.logins.append(command[-1])
            elif command[0] == "timeout":
                mounts = [i for i, word in enumerate(command) if word == "-v"]
                context, out = (self.at(command[i + 1].split(":")[0]) for i in mounts)
                platform = command[command.index("--platform") + 1]
                result = docker.resolve(json.loads((out / "request.json").read_text()), context,
                                        command[mounts[1] + 2], platform.split("/")[1], out.parent)
                (out / "result.json").write_text(json.dumps(result))
            elif command[:2] == ["docker", "build"]:
                labels = dict(command[i + 1].split("=", 1) for i, word in enumerate(command) if word == "--label")
                try:
                    docker.build_image(cwd / "Dockerfile", cwd, command[command.index("-t") + 1],
                                       platform=command[command.index("--platform") + 1], labels=labels)
                except DockerBuildError as e:
                    return 1, str(e)
            elif command[:2] == ["docker", "save"]:
                docker.save(command[4], self.at(command[3]))
            elif command[:2] == ["docker", "run"] and "/layers.py" in command:
                saved = command[command.index("-v") + 1].split(":")[0]
                printed = json.dumps(layers_in_image.written(str(self.at(saved)), json.loads(command[-1])))
            elif command[:2] == ["rm", "-f"]:
                self.at(command[2]).unlink(missing_ok=True)
            elif command[:2] == ["docker", "tag"]:
                self.provider.tagged[command[3]] = command[2]
            elif command[:2] == ["docker", "push"]:
                docker.push(self.provider.tagged[command[2]], command[2], None)
            elif command[0] == "curl" and "--upload-file" in command:
                url = urllib.parse.unquote(command[-1].partition("?")[2])
                get_config().get_object_store_at(url).put_file_at(url, str(self.at(command[-2])))
                self.provider.uploaded.append(url)
            else:
                raise AssertionError(f"a builder VM was asked to run {command}")
        return 0, printed


class FakeVmProvider(SandboxProvider):
    """A sandbox provider whose VMs run amd64, recording what each was asked to do."""

    def __init__(self, docker, root):
        self.docker, self.root, self.vms, self.sizes = docker, root, [], []
        self.ran, self.fetched, self.uploaded, self.binfmt, self.logins, self.tagged = [], [], [], [], [], {}

    async def create_vm(self, **size):
        self.sizes.append(size)
        self.vms.append(vm := FakeVm(self, self.root / str(len(self.vms))))
        return vm


MAIL_SETTINGS = {"properties": {
    "signature": {"default": "", "type": "string"},
    "password": {"default": "", "type": "string", "format": "password", "writeOnly": True, "minLength": 32},
    "token": {"anyOf": [{"type": "string", "format": "password", "writeOnly": True}, {"type": "null"}], "default": None},
    "theme": {"enum": ["light", "dark"], "type": "string", "default": "light"}}, "type": "object"}


MAIL_BUILD = {"properties": {"edition": {"default": "home", "enum": ["home", "pro"], "type": "string"},
                             "locale": {"default": "en_US", "type": "string"}}, "type": "object"}


#: The program of the mail app's own server.
MAIL_SERVER = ("[program:app-mail]\ncommand=/opt/agentenv/apps/mail/venv/bin/python -m "
               "agentenv_protocol.computer.serve_app app.py:Mail\ndirectory=/opt/agentenv/apps/mail\n"
               "environment=ENVIRONMENT_NAME=\"mail\"\nautorestart=true\nredirect_stderr=true\n"
               "stdout_logfile=/var/log/agentenv/app-mail.log\n")


def _app(base, driver, folder, compose=True, supports=None, install=None, settings=MAIL_SETTINGS, launch=None,
         extra=None, steps=(), build=MAIL_BUILD, linux_settings=None, name="mail", paths=None, **changes):
    """A mail app on ``base`` and ``driver``, which serves IMAP from a venv of its requirements; ``extra`` files beside
    them, by name, ``steps``, its install steps by name, scope and the OS types they run on, ``build``, the schema of
    its build settings, ``linux_settings``, that of its settings on Linux, ``name``, its id's last part and its
    environment name, and ``paths``, where it keeps its data on Linux."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "app.py").write_text("class Mail: ...\n")
    (folder / "requirements.txt").write_text("agentenv-framework-protocol\n")
    for path, text in (extra or {}).items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_text(text)
    files = FileArtifactUniverse.put_bundled(f"@local/~/kit/{name}__files", files={
        path: folder / path for path in ("app.py", "requirements.txt", *(extra or {}))})
    declaration = AppDeclaration.model_validate({
        "supports": supports or [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}],
        "permissions": [], "settings": settings, "build_settings": build, "skill": None,
        "data": [], "install_steps": [{"name": name, "scope": scope, "os": os or None} for name, scope, *os in steps],
        "ready_checks": [], "os": {"linux": {
            "launch": launch, "settings": linux_settings, "paths": paths, "install": install or {
                "python": "requirements.txt", "services": [{"name": "imap", "command": "venv/bin/python imap.py"}]}}}})
    return ComputerAppEnv.put(**{"id": f"@local/~/kit/{name}", "environment_name": name, "module": "app.py:Mail",
                                 "files": files, "declaration": declaration, "computer": base, "driver": driver,
                                 "compose": compose, **changes})
