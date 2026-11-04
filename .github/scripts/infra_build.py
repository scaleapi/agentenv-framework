"""What the release workflow's jobs that hold a token to push images or publish to PyPI run for the images agent-env
publishes, its infra images and the shipped agents'. It uses only the standard library and the docker CLI, so those
jobs install nothing; what they're handed by the jobs that run agent-env (``infra_images.py``) is checked against the
images agent-env ships before it's built, pushed or recorded.

``check ENTRY`` fails unless ENTRY, a JSON object as ``infra_images.py plan`` lists an image, names an image agent-env
ships, with its Dockerfile, context, repository and a source tag, and for a disk form, the build args naming its base
computer's image and the disk tools by their repositories and source tags, and its size. ``exists REF`` exits 0 when
the registry has REF, 1 when it says it doesn't, and 2 when it can't tell. ``wanted ENTRY`` checks ENTRY, signs in to
GHCR and writes ``build=true`` to the step's output when the registry lacks its tag, ``build=false`` when it has it.
``build ENTRY ARCH`` checks ENTRY and builds it for linux/ARCH as its tag with ``-ARCH`` after it; a disk form from its
base computer's image of ARCH, which it pulls, with that image's config. ``config IMAGE`` prints the config object of
IMAGE's OCI config, base64'd, as a disk form's build takes its base's. ``smoke NAME IMAGE`` checks that IMAGE, a local
build of NAME, runs: a disk form boots in the QEMU launcher image ``--launcher`` names until every program its base's
supervisord.conf declares runs. ``publish MISSING`` signs in to GHCR and tags the two platform builds of each entry of
MISSING, a JSON list, as one multi-platform image under its tag, once the entry is checked, unless the registry has
that tag. ``verify-record PATH`` fails unless PATH records each image agent-env ships once, with the digest the
registry serves under its tag now. ``retry COMMAND...`` runs COMMAND, again while a registry rate-limits it or PyPI
doesn't serve yet the agentenv-framework-protocol an agent image's build installs.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import posixpath
import re
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

REGISTRY = "ghcr.io/scaleapi"
REPO = Path(__file__).resolve().parents[2]
# Each image agent-env ships, by name, and the Dockerfile it's built from, with its folder as the context.
IMAGES = {
    "agentenv-gateway": "src/agent_env/env/gateway/Dockerfile",
    "agentenv-service-db": "src/agent_env/env/envs/service_db/Dockerfile",
    "agentenv-db-web": "src/agent_env/env/envs/service_db/Dockerfile.db-web",
    "agentenv-db-mcp": "src/agent_env/env/envs/service_db/Dockerfile.db-mcp",
    "agentenv-playwright-mcp": "src/agent_env/env/envs/playwright_mcp/Dockerfile",
    "agentenv-claude-code": "src/agent_env/agents/agents/claude-code/Dockerfile",
    "agentenv-codex": "src/agent_env/agents/agents/codex/Dockerfile",
    "agentenv-gemini": "src/agent_env/agents/agents/gemini/Dockerfile",
    "agentenv-hermes": "src/agent_env/agents/agents/hermes/Dockerfile",
    "agentenv-letta-code": "src/agent_env/agents/agents/letta-code/Dockerfile",
    "agentenv-opencode": "src/agent_env/agents/agents/opencode/Dockerfile",
    "agentenv-openhands": "src/agent_env/agents/agents/openhands/Dockerfile",
    "agentenv-pi": "src/agent_env/agents/agents/pi/Dockerfile",
    "agentenv-proxy": "src/agent_env/agents/agents/proxy/Dockerfile",
    "agentenv-pydantic-ai": "src/agent_env/agents/agents/pydantic-ai/Dockerfile",
    "agentenv-mail-mcp": "src/agent_env/examples/mail-mcp/envs/mail/Dockerfile",
    "agentenv-ubuntu-24.04": "src/agent_env/computers/artifacts/ubuntu-24.04/Dockerfile",
    "agentenv-ubuntu-24.04-headless": "src/agent_env/computers/artifacts/ubuntu-24.04-headless/Dockerfile",
    "agentenv-disk-tools": "src/agent_env/computers/disk/linux/tools/Dockerfile",
    "agentenv-qemu-launcher": "src/agent_env/computers/disk/linux/launcher/Dockerfile",
    "agentenv-ubuntu-24.04-disk": "src/agent_env/computers/disk/linux/boot/Dockerfile",
}
# Each disk form's build args, by the images agent-env ships that they name; its SIZE_GB is its base's [forms.disk]
# min_disk_gb, read from the artifact.toml beside the base's Dockerfile.
DISK_ARGS = {"agentenv-ubuntu-24.04-disk": {"ROOTFS": "agentenv-ubuntu-24.04", "TOOLS": "agentenv-disk-tools"}}
_TAG = re.compile(r"^src-[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEALTHY_WITHIN_SECONDS = 120
_HEALTH_POLL_SECONDS = 2
_LOG_TAIL_LINES = 30  # of a container that never turned healthy, in the error
# How long a disk form may take to boot to its desktop: seconds with KVM, minutes under TCG.
_BOOTED_WITHIN_SECONDS = 900
# What supervisord prints on a disk form's console as each program starts.
_RUNNING = re.compile(r"success: (\S+) entered RUNNING")
_PROGRAM = re.compile(r"^\[program:([^\]]+)\]", re.MULTILINE)
_PRINTED_CHARS = 200  # of what docker printed in place of a manifest, in the error
# What a registry answers when it rate-limits a request: ECR Public's 429, and the toomanyrequests error code GHCR and
# Docker Hub send.
_RATE_LIMITED = re.compile(r"429 Too Many Requests|toomanyrequests", re.IGNORECASE)
# What pip in an agent image's build answers for the agentenv-framework-protocol its release published moments earlier,
# until PyPI's index, which a CDN caches, lists it: pinned (an agent's), or a floor it names (the browser's recorder).
_NOT_YET_SERVED = re.compile(r"No matching distribution found for agentenv-framework-protocol(\[[^\]]*\])?(==|>=)")
# How long to wait before each try again, so a burst of builds starting together spreads out.
_RETRY_WAITS_SECONDS = (30, 60, 120)

# Each image's check, beyond its build: the gateway imports its server, Postgres turns healthy, the sidecars and the
# browser answer their own commands, Claude Code's, Codex's, Gemini's, Hermes', OpenCode's and pi's agents run their CLIs and
# import their servers, OpenHands' imports its server, which loads the OpenHands SDK, Pydantic AI's imports its
# server, which loads Pydantic AI, the proxy imports its server, the mail example's env imports its server, and the
# base computer's desktop comes up, every program running.
_SMOKES = {
    "agentenv-gateway": ["docker", "run", "--rm", "--entrypoint", "python", "{image}", "-c", "import gateway.entrypoint"],
    "agentenv-db-web": ["docker", "run", "--rm", "--entrypoint", "/usr/bin/pgweb", "{image}", "--version"],
    "agentenv-db-mcp": ["docker", "run", "--rm", "{image}", "--help"],
    "agentenv-playwright-mcp": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                                'npx --no-install @playwright/mcp@"$PLAYWRIGHT_MCP_VERSION" --help >/dev/null'
                                " && ls -d /root/.cache/ms-playwright/chromium-*"],
    "agentenv-claude-code": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                             'claude --version && python -c "import a2a_server"'],
    "agentenv-codex": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                       'codex --version && mcp-remote --help >/dev/null && python -c "import a2a_server"'],
    "agentenv-gemini": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                        'gemini --version && python -c "import a2a_server"'],
    "agentenv-hermes": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                        'hermes --version && python -c "import a2a_server"'],
    "agentenv-letta-code": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                            'letta --version && python -c "import a2a_server"'],
    "agentenv-opencode": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                          'opencode --version && python -c "import a2a_server"'],
    "agentenv-openhands": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c", 'python -c "import a2a_server"'],
    "agentenv-pi": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                    'pi --version && python3 -c "import a2a_server"'],
    "agentenv-proxy": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c", 'python -c "import a2a_server"'],
    "agentenv-pydantic-ai": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c", 'python -c "import a2a_server"'],
    "agentenv-mail-mcp": ["docker", "run", "--rm", "--entrypoint", "python", "{image}", "-c", "import server"],
    "agentenv-ubuntu-24.04": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                              "start-desktop init & for _ in $(seq 60); do sleep 1;"
                              " s=$(supervisorctl -c /etc/supervisor/supervisord.conf status);"
                              " echo \"$s\" | grep -q ' RUNNING ' && ! echo \"$s\" | grep -qv ' RUNNING ' && exit 0;"
                              " done; supervisorctl -c /etc/supervisor/supervisord.conf status; exit 1"],
    "agentenv-ubuntu-24.04-headless": [
        "docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
        "start-computer & for _ in $(seq 60); do sleep 1;"
        " s=$(supervisorctl -c /etc/supervisor/supervisord.conf status);"
        " echo \"$s\" | grep -q ' RUNNING ' && ! echo \"$s\" | grep -qv ' RUNNING ' && exit 0;"
        " done; supervisorctl -c /etc/supervisor/supervisord.conf status; exit 1"],
    "agentenv-disk-tools": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                            "command -v disk image-env mkfs.ext4 mkfs.vfat mcopy sfdisk qemu-img"],
    "agentenv-qemu-launcher": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                               "qemu-system-x86_64 --version && qemu-system-aarch64 --version"
                               " && test -f /usr/share/OVMF/OVMF_CODE_4M.fd && test -f /usr/share/AAVMF/AAVMF_CODE.fd"
                               " && command -v socat zstd"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check").add_argument("entry")
    commands.add_parser("exists").add_argument("ref")
    commands.add_parser("wanted").add_argument("entry")
    build = commands.add_parser("build")
    build.add_argument("entry")
    build.add_argument("arch", choices=("amd64", "arm64"))
    commands.add_parser("config").add_argument("image")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("name", choices=sorted(IMAGES))
    smoke.add_argument("image")
    smoke.add_argument("--launcher")
    commands.add_parser("publish").add_argument("missing")
    commands.add_parser("verify-record").add_argument("path", type=Path)
    commands.add_parser("retry").add_argument("argv", nargs=argparse.REMAINDER, metavar="COMMAND")
    args = parser.parse_args()
    if args.command == "retry":
        return _tried_again(lambda: _streamed(args.argv)).returncode
    if args.command == "exists":
        return _exists(args.ref)
    if args.command == "wanted":
        return _wanted(args.entry)
    if args.command == "build":
        return _build(args.entry, args.arch)
    if args.command == "config":
        print(_config(args.image))
        return 0
    if args.command == "smoke":
        if args.name in DISK_ARGS and args.launcher is None:
            parser.error(f"{args.name} is booted in a QEMU launcher, which --launcher names")
        return _smoke(args.name, args.image, args.launcher)
    if args.command == "publish":
        return _publish(args.missing)
    problems = _check(args.entry) if args.command == "check" else _verify_record(args.path)
    for problem in problems:
        print(f"::error::{problem}")
    return 1 if problems else 0


def _wanted(entry: str) -> int:
    if not _checked(entry):
        return 1
    _sign_in()
    ref = _ref(json.loads(entry))
    if (code := _exists(ref)) == 2:
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"build={'true' if code == 1 else 'false'}\n")
    if code == 0:
        print(f"{ref} is published already, so it isn't built again")
    return 0


def _build(entry: str, arch: str) -> int:
    """Build ENTRY for linux/``arch``, with no provenance attestation, which would show in the multi-platform image as an
    unknown platform."""
    if not _checked(entry):
        return 1
    image, target, build_args = json.loads(entry), f"linux/{arch}", []
    if image["name"] in DISK_ARGS:
        args, rootfs = image["args"], f"{image['args']['ROOTFS']}-{arch}"
        if code := _tried_again(lambda: _streamed(["docker", "pull", "--platform", target, rootfs])).returncode:
            return code
        for arg, value in {"ROOTFS": rootfs, "TOOLS": args["TOOLS"], "SIZE_GB": args["SIZE_GB"],
                           "ROOTFS_CONFIG": _config(rootfs)}.items():
            build_args += ["--build-arg", f"{arg}={value}"]
    command = ["docker", "build", "--platform", target, "--provenance=false", "-f", image["dockerfile"],
               "-t", f"{_ref(image)}-{arch}", *build_args,
               "--label", f"org.opencontainers.image.source=https://github.com/{os.environ['GITHUB_REPOSITORY']}",
               "--label", f"org.opencontainers.image.revision={os.environ['GITHUB_SHA']}",
               "--label", f"io.agentenv.source-tag={image['tag']}", image["context"]]
    return _tried_again(lambda: _streamed(command)).returncode


def _publish(missing: str) -> int:
    _sign_in()
    for image in json.loads(missing):
        if not _checked(json.dumps(image)):
            return 1
        ref = _ref(image)
        if (code := _exists(ref)) == 0:
            print(f"{ref} is published already, so it's left as it is")
            continue
        if code != 1:
            return 1
        created = _tried_again(lambda: _streamed([
            "docker", "buildx", "imagetools", "create", "-t", ref,
            "--annotation", f"index:org.opencontainers.image.source=https://github.com/{os.environ['GITHUB_REPOSITORY']}",
            "--annotation", f"index:org.opencontainers.image.revision={os.environ['GITHUB_SHA']}",
            f"{ref}-amd64", f"{ref}-arm64"]))
        if created.returncode:
            return created.returncode
    return 0


def _checked(entry: str) -> bool:
    problems = _check(entry)
    for problem in problems:
        print(f"::error::{problem}")
    return not problems


def _ref(image: dict) -> str:
    return f"{image['repository']}:{image['tag']}"


def _sign_in() -> None:
    """Sign in to GHCR with the job's token, as the workflow's actor."""
    subprocess.run(["docker", "login", "ghcr.io", "--username", os.environ["GITHUB_ACTOR"], "--password-stdin"],
                   input=os.environ["GHCR_TOKEN"], text=True, check=True)


def _check(entry: str) -> list[str]:
    try:
        image = json.loads(entry)
    except ValueError:
        return [f"{entry!r} isn't a JSON object"]
    name = image.get("name") if isinstance(image, dict) else None
    if name not in IMAGES:
        return [f"{name!r} isn't an image agent-env ships"]
    expected = {"name": name, "repository": f"{REGISTRY}/{name}", "dockerfile": IMAGES[name],
                "context": posixpath.dirname(IMAGES[name])}
    problems = [f"{name}: {key} is {image.get(key)!r}, not {value!r}"
                for key, value in expected.items() if image.get(key) != value]
    if not _TAG.match(str(image.get("tag"))):
        problems.append(f"{name}: {image.get('tag')!r} isn't a source tag")
    if name in DISK_ARGS:
        problems += _check_args(name, image.get("args"))
    if extra := sorted(set(image) - {*expected, "tag", *(("args",) if name in DISK_ARGS else ())}):
        problems.append(f"{name}: unexpected {', '.join(extra)}")
    return problems


def _check_args(name: str, args: object) -> list[str]:
    """The problems with a disk form's build args: each image arg must name the image agent-env ships it's for, by a
    source tag, and SIZE_GB must be its base's size."""
    wanted = DISK_ARGS[name]
    if not isinstance(args, dict) or set(args) != {*wanted, "SIZE_GB"}:
        return [f"{name}: args is {args!r}, not {', '.join(sorted({*wanted, 'SIZE_GB'}))}"]
    problems = []
    for arg, image in wanted.items():
        repository, _, tag = str(args[arg]).rpartition(":")
        if repository != f"{REGISTRY}/{image}" or not _TAG.match(tag):
            problems.append(f"{name}: {arg} is {args[arg]!r}, not {REGISTRY}/{image} under a source tag")
    if args["SIZE_GB"] != (size := str(_disk_size(name))):
        problems.append(f"{name}: SIZE_GB is {args['SIZE_GB']!r}, not its base's min_disk_gb, {size}")
    return problems


def _disk_size(name: str) -> int:
    """The min_disk_gb of the base computer a disk form is made of, as its artifact.toml declares it."""
    toml = Path(IMAGES[DISK_ARGS[name]["ROOTFS"]]).with_name("artifact.toml")
    return tomllib.loads((REPO / toml).read_text())["forms"]["disk"]["min_disk_gb"]


def _tried_again(attempt: Callable[[], subprocess.CompletedProcess]) -> subprocess.CompletedProcess:
    """``attempt()``, and again after each of ``_RETRY_WAITS_SECONDS`` while what it printed says a later try can work
    (``_why_again``)."""
    for wait in _RETRY_WAITS_SECONDS:
        result = attempt()
        if result.returncode == 0 or (why := _why_again(result)) is None:
            return result
        print(f"::warning::{why}; trying again in {wait}s", flush=True)
        time.sleep(wait)
    return attempt()


def _why_again(result: subprocess.CompletedProcess) -> str | None:
    """Why a failed run can work later, or None: a registry rate-limited it, as public.ecr.aws does past one anonymous
    pull a second, which builds starting together pass; or a build's pip can't find the agentenv-framework-protocol its
    release has just published, which PyPI's index lists a little after the upload."""
    printed, command = f"{result.stdout}{result.stderr}", f"`{' '.join(result.args)[:_PRINTED_CHARS]}`"
    if _RATE_LIMITED.search(printed):
        return f"a registry rate-limited {command}"
    if _NOT_YET_SERVED.search(printed):
        return f"PyPI doesn't list yet the agentenv-framework-protocol {command} installs"
    return None


def _streamed(argv: list[str]) -> subprocess.CompletedProcess:
    """Run ``argv``, printing what it prints as it prints it, which is kept too, to look for a rate limit in. Each line is
    flushed, since the workflow's log is a pipe, which Python would otherwise fill before passing on."""
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    printed = []
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        printed.append(line)
    return subprocess.CompletedProcess(argv, process.wait(), "".join(printed), "")


def _served(ref: str) -> tuple[str | None, str]:
    """The digest the registry serves under ``ref``, or None, and what docker said when it serves none. The manifest as
    JSON, since buildx before 0.35 doesn't know the ``.Manifest.Digest`` template and prints the whole listing."""
    result = _tried_again(lambda: subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", ref, "--format", "{{json .Manifest}}"],
        capture_output=True, text=True))
    if result.returncode != 0:
        return None, result.stderr.strip()
    try:
        return json.loads(result.stdout)["digest"], ""
    except (ValueError, KeyError, TypeError):
        return None, f"docker printed no manifest: {result.stdout.strip()[:_PRINTED_CHARS]}"


def _exists(ref: str) -> int:
    digest, error = _served(ref)
    if digest is not None:
        return 0
    if ": not found" in error:
        return 1
    print(f"::error::couldn't tell whether the registry has {ref}: {error}")
    return 2


def _verify_record(path: Path) -> list[str]:
    try:
        record = json.loads(path.read_text())
        images = record["images"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"{path} isn't a record of published images: {type(e).__name__}: {e}"]
    if set(record) != {"images"} or not isinstance(images, list):
        return [f"{path} isn't a record of published images: it holds {sorted(record)}"]
    problems, seen = [], []
    for image in images:
        if not isinstance(image, dict) or set(image) != {"repository", "tag", "digest"}:
            problems.append(f"{path}: {image!r} isn't an image's repository, tag and digest")
            continue
        repository, tag, digest = image["repository"], image["tag"], image["digest"]
        name = repository.removeprefix(f"{REGISTRY}/") if isinstance(repository, str) else None
        if name not in IMAGES or repository != f"{REGISTRY}/{name}":
            problems.append(f"{path}: {repository!r} isn't an image agent-env ships")
            continue
        seen.append(name)
        if not _TAG.match(str(tag)) or not _DIGEST.match(str(digest)):
            problems.append(f"{path}: {name} has tag {tag!r} and digest {digest!r}, not a source tag and a sha256")
            continue
        served, error = _served(f"{repository}:{tag}")
        if served != digest:
            problems.append(f"{path}: {repository}:{tag} is recorded as {digest}, and the registry serves "
                            f"{served or f'nothing ({error})'}")
    if missing := sorted(set(IMAGES) - set(seen)):
        problems.append(f"{path} records no {', '.join(missing)}")
    if repeated := sorted({name for name in seen if seen.count(name) > 1}):
        problems.append(f"{path} records {', '.join(repeated)} more than once")
    return problems


def _config(image: str) -> str:
    """The config object of ``image``'s OCI config, a local image's, as base64'd JSON."""
    inspected = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config}}", image],
                               capture_output=True, text=True, check=True)
    return base64.b64encode(inspected.stdout.strip().encode()).decode()


def _smoke(name: str, image: str, launcher: str | None) -> int:
    if name == "agentenv-service-db":
        return _healthy(image)
    if name in DISK_ARGS:
        return _boots(name, image, launcher)
    command = [part.replace("{image}", image) for part in _SMOKES[name]]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"::error::{name}: `{' '.join(command)}` exited {result.returncode}:\n{result.stdout}{result.stderr}")
    return result.returncode


def _boots(name: str, image: str, launcher: str) -> int:
    """Boot ``image``, a build of the disk form ``name``, in ``launcher``, with KVM where it can, and wait for every
    program its base computer's supervisord.conf declares to run, which its console shows."""
    programs = set(_PROGRAM.findall((REPO / Path(IMAGES[DISK_ARGS[name]["ROOTFS"]]).with_name("supervisord.conf"))
                                    .read_text()))
    arch = subprocess.run(["docker", "image", "inspect", "--format", "{{.Architecture}}", image], capture_output=True,
                          text=True, check=True).stdout.strip()
    machine = f"{image}-machine"
    subprocess.run(["docker", "build", "-q", "-t", machine, "-"], text=True, check=True, capture_output=True,
                   input=f"FROM --platform=linux/{arch} {image} AS disk\nFROM {launcher}\nCOPY --from=disk /disk/ /disk/\n")
    kvm = ["--device", "/dev/kvm"] if Path("/dev/kvm").exists() else []
    container = subprocess.run(["docker", "run", "-d", *kvm, "-e", f"VM_ARCH={arch}", machine], capture_output=True,
                               text=True, check=True).stdout.strip()
    try:
        deadline, logs = time.monotonic() + _BOOTED_WITHIN_SECONDS, ""
        while time.monotonic() < deadline:
            logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True, errors="replace")
            logs = logs.stdout + logs.stderr
            if programs <= set(_RUNNING.findall(logs)):
                native = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine()) == arch
                print(f"{image} booted to its desktop {'with KVM' if kvm and native else 'under TCG'}")
                return 0
            time.sleep(_HEALTH_POLL_SECONDS)
        tail = "\n".join(logs.splitlines()[-_LOG_TAIL_LINES:])
        print(f"::error::{image} didn't boot to its desktop within {_BOOTED_WITHIN_SECONDS}s:\n{tail}")
        return 1
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


def _healthy(image: str) -> int:
    """Start ``image`` and wait for its health check to pass, as compose does before starting what depends on it."""
    container = subprocess.run(["docker", "run", "-d", image], capture_output=True, text=True, check=True).stdout.strip()
    try:
        status, deadline = "", time.monotonic() + _HEALTHY_WITHIN_SECONDS
        while time.monotonic() < deadline:
            status = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", container],
                                    capture_output=True, text=True).stdout.strip()
            if status == "healthy":
                return 0
            time.sleep(_HEALTH_POLL_SECONDS)
        logs = subprocess.run(["docker", "logs", "--tail", str(_LOG_TAIL_LINES), container], capture_output=True,
                              text=True)
        print(f"::error::{image} wasn't healthy within {_HEALTHY_WITHIN_SECONDS}s (last: {status}):\n"
              f"{logs.stdout}{logs.stderr}")
        return 1
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
