"""What the release workflow's jobs that hold a token to push images or publish to PyPI run for the images agent-env
publishes, its infra images and the shipped agents'. It uses only the standard library and the docker CLI, so those
jobs install nothing; what they're handed by the jobs that run agent-env (``infra_images.py``) is checked against the
images agent-env ships before it's built, pushed or recorded.

``check ENTRY`` fails unless ENTRY, a JSON object as ``infra_images.py plan`` lists an image, names an image agent-env
ships, with its Dockerfile, context, repository and a source tag. ``exists REF`` exits 0 when the registry has REF, 1
when it says it doesn't, and 2 when it can't tell. ``smoke NAME IMAGE`` checks that IMAGE, a local build of NAME, runs.
``verify-record PATH`` fails unless PATH records each image agent-env ships once, with the digest the registry serves
under its tag now. ``retry COMMAND...`` runs COMMAND, again while a registry rate-limits it or PyPI doesn't serve
yet the agentenv-framework-protocol an agent image's build installs.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

REGISTRY = "ghcr.io/scaleapi"
# Each image agent-env ships, by name, and the Dockerfile it's built from, with its folder as the context.
IMAGES = {
    "agentenv-gateway": "src/agent_env/env/gateway/Dockerfile",
    "agentenv-service-db": "src/agent_env/env/envs/service_db/Dockerfile",
    "agentenv-db-web": "src/agent_env/env/envs/service_db/Dockerfile.db-web",
    "agentenv-db-mcp": "src/agent_env/env/envs/service_db/Dockerfile.db-mcp",
    "agentenv-website-browser": "src/agent_env/env/envs/website_browser/Dockerfile",
    "agentenv-claude-code": "src/agent_env/agents/agents/claude-code/Dockerfile",
    "agentenv-codex": "src/agent_env/agents/agents/codex/Dockerfile",
    "agentenv-gemini": "src/agent_env/agents/agents/gemini/Dockerfile",
    "agentenv-opencode": "src/agent_env/agents/agents/opencode/Dockerfile",
    "agentenv-openhands": "src/agent_env/agents/agents/openhands/Dockerfile",
    "agentenv-pi": "src/agent_env/agents/agents/pi/Dockerfile",
    "agentenv-proxy": "src/agent_env/agents/agents/proxy/Dockerfile",
}
_TAG = re.compile(r"^src-[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEALTHY_WITHIN_SECONDS = 120
_HEALTH_POLL_SECONDS = 2
_LOG_TAIL_LINES = 30  # of a container that never turned healthy, in the error
_PRINTED_CHARS = 200  # of what docker printed in place of a manifest, in the error
# What a registry answers when it rate-limits a request: ECR Public's 429, and the toomanyrequests error code GHCR and
# Docker Hub send.
_RATE_LIMITED = re.compile(r"429 Too Many Requests|toomanyrequests", re.IGNORECASE)
# What pip in an agent image's build answers for the agentenv-framework-protocol its release published moments earlier,
# until PyPI's index, which a CDN caches, lists it.
_NOT_YET_SERVED = re.compile(r"No matching distribution found for agentenv-framework-protocol(\[[^\]]*\])?==")
# How long to wait before each try again, so a burst of builds starting together spreads out.
_RETRY_WAITS_SECONDS = (30, 60, 120)

# Each image's check, beyond its build: the gateway imports its server, Postgres turns healthy, the sidecars and the
# browser answer their own commands, Claude Code's, Codex's, Gemini's, OpenCode's and pi's agents run their CLIs and
# import their servers, OpenHands' imports its server, which loads the OpenHands SDK, and the proxy imports its server.
_SMOKES = {
    "agentenv-gateway": ["docker", "run", "--rm", "--entrypoint", "python", "{image}", "-c", "import gateway.entrypoint"],
    "agentenv-db-web": ["docker", "run", "--rm", "--entrypoint", "/usr/bin/pgweb", "{image}", "--version"],
    "agentenv-db-mcp": ["docker", "run", "--rm", "{image}", "--help"],
    "agentenv-website-browser": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                                 'npx --no-install @playwright/mcp@"$PLAYWRIGHT_MCP_VERSION" --help >/dev/null'
                                 " && ls -d /root/.cache/ms-playwright/chromium-*"],
    "agentenv-claude-code": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                             'claude --version && python -c "import a2a_server"'],
    "agentenv-codex": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                       'codex --version && mcp-remote --help >/dev/null && python -c "import a2a_server"'],
    "agentenv-gemini": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                        'gemini --version && python -c "import a2a_server"'],
    "agentenv-opencode": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                          'opencode --version && python -c "import a2a_server"'],
    "agentenv-openhands": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c", 'python -c "import a2a_server"'],
    "agentenv-pi": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c",
                    'pi --version && python3 -c "import a2a_server"'],
    "agentenv-proxy": ["docker", "run", "--rm", "--entrypoint", "sh", "{image}", "-c", 'python -c "import a2a_server"'],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check").add_argument("entry")
    commands.add_parser("exists").add_argument("ref")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("name", choices=sorted(IMAGES))
    smoke.add_argument("image")
    commands.add_parser("verify-record").add_argument("path", type=Path)
    commands.add_parser("retry").add_argument("argv", nargs=argparse.REMAINDER, metavar="COMMAND")
    args = parser.parse_args()
    if args.command == "retry":
        return _tried_again(lambda: _streamed(args.argv)).returncode
    if args.command == "exists":
        return _exists(args.ref)
    if args.command == "smoke":
        return _smoke(args.name, args.image)
    problems = _check(args.entry) if args.command == "check" else _verify_record(args.path)
    for problem in problems:
        print(f"::error::{problem}")
    return 1 if problems else 0


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
    if extra := sorted(set(image) - {*expected, "tag"}):
        problems.append(f"{name}: unexpected {', '.join(extra)}")
    return problems


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


def _smoke(name: str, image: str) -> int:
    if name == "agentenv-service-db":
        return _healthy(image)
    command = [part.replace("{image}", image) for part in _SMOKES[name]]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"::error::{name}: `{' '.join(command)}` exited {result.returncode}:\n{result.stdout}{result.stderr}")
    return result.returncode


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
