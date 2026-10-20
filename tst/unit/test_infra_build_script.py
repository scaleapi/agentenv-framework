"""The script the release workflow's jobs that push images or publish to PyPI run, ``.github/scripts/infra_build.py``:
standard library only, and every entry it's handed checked against the images agent-env ships."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_env.env import bootstrap
from agent_env.env.published_images import REGISTRY, write_release_digests
from tst.util.scripts import SCRIPTS, load_script

REPO = SCRIPTS.parents[1]
SCRIPT = SCRIPTS / "infra_build.py"
_DIGEST = "sha256:" + "e" * 64
_TAG = "src-" + "a" * 64


@pytest.fixture
def script():
    return load_script(SCRIPT.name)


def _run(script, monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", [SCRIPT.name, *args])
    return script.main()


def test_it_imports_only_the_standard_library():
    """Those jobs install nothing, so a dependency's code never runs where a token can push or publish."""
    imported = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert imported - {"__future__"} <= sys.stdlib_module_names, imported - sys.stdlib_module_names


def test_its_images_are_the_ones_agent_env_ships(script):
    assert script.REGISTRY == REGISTRY
    assert script.IMAGES == {image.name: str(image.dockerfile.relative_to(REPO)) for image in bootstrap.PUBLISHED_IMAGES}
    assert set(script._SMOKES) | {"agentenv-service-db"} == set(script.IMAGES)


def test_check_accepts_every_image_the_plan_lists(script, monkeypatch, capsys):
    plan = load_script("infra_images.py")

    for image in bootstrap.PUBLISHED_IMAGES:
        assert _run(script, monkeypatch, "check", json.dumps(plan._described(image))) == 0, capsys.readouterr().out


def _gateway(**changes) -> str:
    entry = {"name": "agentenv-gateway", "repository": f"{REGISTRY}/agentenv-gateway", "tag": _TAG,
             "dockerfile": "src/agent_env/env/gateway/Dockerfile", "context": "src/agent_env/env/gateway"}
    return json.dumps({key: value for key, value in {**entry, **changes}.items() if value is not None})


@pytest.mark.parametrize("entry, problem", [
    (_gateway(name="agentenv-evil"), "'agentenv-evil' isn't an image agent-env ships"),
    (_gateway(repository="ghcr.io/someone/agentenv-gateway"), "agentenv-gateway: repository is 'ghcr.io/someone/"),
    (_gateway(dockerfile="tst/data/agentenv_mcp/Dockerfile"), "agentenv-gateway: dockerfile is 'tst/data/agentenv_mcp"),
    (_gateway(context="."), "agentenv-gateway: context is '.'"),
    (_gateway(tag="latest"), "agentenv-gateway: 'latest' isn't a source tag"),
    (_gateway(tag=None), "agentenv-gateway: None isn't a source tag"),
    (_gateway(platform="linux/riscv64"), "agentenv-gateway: unexpected platform"),
    ("[]", "None isn't an image agent-env ships"),
    ("not json", "'not json' isn't a JSON object"),
])
def test_check_refuses_anything_else(script, monkeypatch, capsys, entry, problem):
    assert _run(script, monkeypatch, "check", entry) == 1

    assert f"::error::{problem}" in capsys.readouterr().out


def _docker(served: dict[str, str], calls: list[list[str]] | None = None, statuses: list[str] | None = None):
    """Docker as the registry answers: ``served`` maps a ref to its digest, or to an error docker prints."""
    def run(command, **kw):
        if calls is not None:
            calls.append(command)
        if command[:3] == ["docker", "buildx", "imagetools"]:
            answer = served.get(command[4], "ERROR: not found")
            if answer.startswith("ERROR"):
                return subprocess.CompletedProcess(command, 1, "", f"ERROR: {command[4]}: {answer[7:]}\n")
            return subprocess.CompletedProcess(command, 0, json.dumps({"mediaType": "index", "digest": answer}), "")
        if command[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(command, 0, "c123\n", "")
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, statuses.pop(0) if statuses else "unhealthy", "")
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, "FATAL: data directory has wrong ownership", "")
        return subprocess.CompletedProcess(command, 0, "", "")
    return run


@pytest.mark.parametrize("answer, code", [(_DIGEST, 0), ("ERROR: not found", 1), ("ERROR: connection refused", 2)])
def test_exists_says_yes_no_or_that_it_cant_tell(script, monkeypatch, capsys, answer, code):
    ref = f"{REGISTRY}/agentenv-gateway:{_TAG}"
    monkeypatch.setattr(script.subprocess, "run", _docker({ref: answer}))

    assert _run(script, monkeypatch, "exists", ref) == code

    assert ("::error::couldn't tell whether the registry has" in capsys.readouterr().out) is (code == 2)


def _record(tmp_path, digests: dict[str, str] | None = None) -> Path:
    path = tmp_path / "published_images.json"
    write_release_digests(digests or {image.ref(): _DIGEST for image in bootstrap.PUBLISHED_IMAGES}, path)
    return path


def test_verify_record_accepts_what_the_registry_serves(script, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(script.subprocess, "run", _docker({image.ref(): _DIGEST for image in bootstrap.PUBLISHED_IMAGES}))

    assert _run(script, monkeypatch, "verify-record", str(_record(tmp_path))) == 0, capsys.readouterr().out


def test_verify_record_refuses_a_digest_the_registry_doesnt_serve_under_its_tag(script, monkeypatch, tmp_path, capsys):
    gateway, *rest = bootstrap.PUBLISHED_IMAGES
    served = {image.ref(): _DIGEST for image in rest} | {gateway.ref(): "sha256:" + "0" * 64}
    monkeypatch.setattr(script.subprocess, "run", _docker(served))

    assert _run(script, monkeypatch, "verify-record", str(_record(tmp_path))) == 1

    assert f"{gateway.ref()} is recorded as {_DIGEST}, and the registry serves sha256:{'0' * 64}" in capsys.readouterr().out


@pytest.mark.parametrize("record, problem", [
    ({"images": []}, "records no agentenv-db-mcp, agentenv-db-web"),
    ({"images": [], "extra": 1}, "it holds ['extra', 'images']"),
    ({"images": [{"repository": "ghcr.io/someone/thing", "tag": _TAG, "digest": _DIGEST}]},
     "'ghcr.io/someone/thing' isn't an image agent-env ships"),
    ({"images": [{"repository": f"{REGISTRY}/agentenv-gateway", "tag": "latest", "digest": _DIGEST}]},
     "agentenv-gateway has tag 'latest'"),
    ({"images": [{"repository": f"{REGISTRY}/agentenv-gateway", "tag": _TAG, "digest": _DIGEST, "more": 1}]},
     "isn't an image's repository, tag and digest"),
    ([], "isn't a record of published images: TypeError"),
])
def test_verify_record_refuses_a_record_that_isnt_one(script, monkeypatch, tmp_path, capsys, record, problem):
    monkeypatch.setattr(script.subprocess, "run", _docker({}))
    path = tmp_path / "published_images.json"
    path.write_text(json.dumps(record))

    assert _run(script, monkeypatch, "verify-record", str(path)) == 1

    assert problem in capsys.readouterr().out


def test_verify_record_refuses_an_image_recorded_twice(script, monkeypatch, tmp_path, capsys):
    path = _record(tmp_path)
    record = json.loads(path.read_text())
    record["images"].append(record["images"][0])
    path.write_text(json.dumps(record))
    monkeypatch.setattr(script.subprocess, "run", _docker({image.ref(): _DIGEST for image in bootstrap.PUBLISHED_IMAGES}))

    assert _run(script, monkeypatch, "verify-record", str(path)) == 1

    assert f"records {record['images'][0]['repository'].split('/')[-1]} more than once" in capsys.readouterr().out


@pytest.mark.parametrize("name, expected", [
    ("agentenv-gateway", ["docker", "run", "--rm", "--entrypoint", "python", "img", "-c", "import gateway.entrypoint"]),
    ("agentenv-db-mcp", ["docker", "run", "--rm", "img", "--help"]),
])
def test_a_smoke_runs_the_images_own_check(script, monkeypatch, name, expected):
    calls = []
    monkeypatch.setattr(script.subprocess, "run", _docker({}, calls))

    assert _run(script, monkeypatch, "smoke", name, "img") == 0

    assert calls == [expected]


def test_a_failing_smoke_names_the_command_and_its_output(script, monkeypatch, capsys):
    monkeypatch.setattr(script.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 1, "", "ModuleNotFoundError: httpx"))

    assert _run(script, monkeypatch, "smoke", "agentenv-gateway", "img") == 1

    out = capsys.readouterr().out
    assert "::error::agentenv-gateway: `docker run --rm --entrypoint python img" in out and "httpx" in out


def test_the_service_db_smoke_waits_for_its_health_check_and_removes_the_container(script, monkeypatch):
    calls = []
    monkeypatch.setattr(script.subprocess, "run", _docker({}, calls, ["starting", "starting", "healthy"]))
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)

    assert _run(script, monkeypatch, "smoke", "agentenv-service-db", "pg") == 0

    assert calls[0] == ["docker", "run", "-d", "pg"] and calls[-1] == ["docker", "rm", "-f", "c123"]


def test_a_service_db_that_never_turns_healthy_fails_with_its_logs(script, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(script.subprocess, "run", _docker({}, calls, []))
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(script, "_HEALTHY_WITHIN_SECONDS", 0.01)

    assert _run(script, monkeypatch, "smoke", "agentenv-service-db", "pg") == 1

    out = capsys.readouterr().out
    assert "::error::pg wasn't healthy within 0.01s" in out and "wrong ownership" in out
    assert calls[-1] == ["docker", "rm", "-f", "c123"]


def test_exists_cant_tell_when_docker_prints_no_manifest(script, monkeypatch, capsys):
    ref = f"{REGISTRY}/agentenv-gateway:{_TAG}"
    monkeypatch.setattr(script.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 0, "Name: a listing\n", ""))

    assert _run(script, monkeypatch, "exists", ref) == 2

    assert "docker printed no manifest: Name: a listing" in capsys.readouterr().out


_LIMITED = ("#2 ERROR: failed to copy: httpReadSeeker: failed open: unexpected status code https://public.ecr.aws/v2/"
            "docker/library/python/manifests/sha256:c208: 429 Too Many Requests\n")


class _Process:
    """What ``subprocess.Popen`` returns, printing ``printed`` and exiting ``code``."""

    def __init__(self, printed: str, code: int):
        self.stdout, self._code = iter(printed.splitlines(keepends=True)), code

    def wait(self) -> int:
        return self._code


def _attempts(script, monkeypatch, answers: list[tuple[str, int]]) -> tuple[list[list[str]], list[float]]:
    """Each run of the command prints and exits as the next of ``answers``; returns the commands run and the waits."""
    runs, waits = [], []

    def popen(argv, **kw):
        runs.append(argv)
        return _Process(*answers[len(runs) - 1])

    monkeypatch.setattr(script.subprocess, "Popen", popen)
    monkeypatch.setattr(script.time, "sleep", waits.append)
    return runs, waits


def test_retry_runs_a_command_once_when_it_works_and_prints_what_it_printed(script, monkeypatch, capsys):
    runs, waits = _attempts(script, monkeypatch, [("#1 DONE 0.0s\n", 0)])

    assert _run(script, monkeypatch, "retry", "docker", "build", "--platform", "linux/amd64", ".") == 0

    assert runs == [["docker", "build", "--platform", "linux/amd64", "."]] and waits == []
    assert capsys.readouterr().out == "#1 DONE 0.0s\n"


def test_retry_tries_again_while_a_registry_rate_limits_it(script, monkeypatch, capsys):
    runs, waits = _attempts(script, monkeypatch, [(_LIMITED, 1), ("toomanyrequests: retry-after: 1s\n", 1),
                                                  ("pushed\n", 0)])

    assert _run(script, monkeypatch, "retry", "docker", "push", "img") == 0

    assert len(runs) == 3 and waits == list(script._RATE_LIMIT_WAITS_SECONDS[:2])
    assert capsys.readouterr().out.count("::warning::a registry rate-limited `docker push img`; trying again in") == 2


def test_retry_gives_up_after_its_waits(script, monkeypatch, capsys):
    runs, waits = _attempts(script, monkeypatch, [(_LIMITED, 1)] * (len(script._RATE_LIMIT_WAITS_SECONDS) + 1))

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 1

    assert len(runs) == len(script._RATE_LIMIT_WAITS_SECONDS) + 1 and waits == list(script._RATE_LIMIT_WAITS_SECONDS)


def test_retry_doesnt_try_again_a_command_that_fails_otherwise(script, monkeypatch, capsys):
    runs, waits = _attempts(script, monkeypatch, [("ERROR: failed to solve: process \"/bin/sh -c pip install\"\n", 1)])

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 1

    assert len(runs) == 1 and waits == []


def test_a_rate_limited_registry_lookup_is_tried_again(script, monkeypatch, capsys):
    ref = f"{REGISTRY}/agentenv-gateway:{_TAG}"
    answers = [subprocess.CompletedProcess([], 1, "", "ERROR: toomanyrequests: retry-after: 1s\n"),
               subprocess.CompletedProcess([], 0, json.dumps({"digest": _DIGEST}), "")]
    monkeypatch.setattr(script.subprocess, "run", lambda command, **kw: answers.pop(0))
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)

    assert _run(script, monkeypatch, "exists", ref) == 0
