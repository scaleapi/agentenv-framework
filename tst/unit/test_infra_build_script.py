"""The script the release workflow's jobs that push images or publish to PyPI run, ``.github/scripts/infra_build.py``:
standard library only, and every entry it's handed checked against the images agent-env ships."""

from __future__ import annotations

import ast
import base64
import configparser
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_env.env import bootstrap
from agent_env.env.published_images import DISK_IMAGES, REGISTRY, write_release_digests
from tst.util.scripts import SCRIPTS, load_script

REPO = SCRIPTS.parents[1]
SCRIPT = SCRIPTS / "infra_build.py"
_DIGEST = "sha256:" + "e" * 64
_TAG = "src-" + "a" * 64


@pytest.fixture
def script():
    return load_script(SCRIPT.name)


@pytest.fixture
def workflow(monkeypatch, tmp_path) -> Path:
    """What a workflow step's environment holds: its output file, which is returned, the actor and the job's token, the
    repository and the commit."""
    output = tmp_path / "output"
    output.touch()
    for name, value in {"GITHUB_OUTPUT": str(output), "GITHUB_ACTOR": "bot", "GHCR_TOKEN": "t0ken",
                        "GITHUB_REPOSITORY": "scaleapi/agentenv-framework", "GITHUB_SHA": "c0ffee"}.items():
        monkeypatch.setenv(name, value)
    return output


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
    assert set(script._SMOKES) | {"agentenv-service-db", *script.DISK_ARGS} == set(script.IMAGES)
    assert script.DISK_ARGS == {image.name: {arg: built.name for arg, built in image.args} for image in DISK_IMAGES}


def test_check_accepts_every_image_the_plan_lists(script, monkeypatch, capsys):
    plan = load_script("infra_images.py")

    for image in bootstrap.PUBLISHED_IMAGES:
        assert _run(script, monkeypatch, "check", json.dumps(plan._described(image))) == 0, capsys.readouterr().out


def _gateway(**changes) -> str:
    entry = {"name": "agentenv-gateway", "repository": f"{REGISTRY}/agentenv-gateway", "tag": _TAG,
             "dockerfile": "src/agent_env/env/gateway/Dockerfile", "context": "src/agent_env/env/gateway"}
    return json.dumps({key: value for key, value in {**entry, **changes}.items() if value is not None})


def _disk(**args) -> str:
    image = "agentenv-ubuntu-24.04-disk"
    return json.dumps({"name": image, "repository": f"{REGISTRY}/{image}", "tag": _TAG,
                       "dockerfile": "src/agent_env/computers/disk/linux/boot/Dockerfile",
                       "context": "src/agent_env/computers/disk/linux/boot", "args": {
                           "ROOTFS": f"{REGISTRY}/agentenv-ubuntu-24.04:{_TAG}", "TOOLS": f"{REGISTRY}/agentenv-disk-tools:{_TAG}",
                           "SIZE_GB": "16", **args}})


@pytest.mark.parametrize("entry, problem", [
    (_gateway(name="agentenv-evil"), "'agentenv-evil' isn't an image agent-env ships"),
    (_gateway(args={"ROOTFS": "evil"}), "agentenv-gateway: unexpected args"),
    (_disk(ROOTFS="ghcr.io/someone/ubuntu:latest"), "agentenv-ubuntu-24.04-disk: ROOTFS is 'ghcr.io/someone/ubuntu:latest'"),
    (_disk(TOOLS=f"{REGISTRY}/agentenv-disk-tools:latest"), "agentenv-ubuntu-24.04-disk: TOOLS is 'ghcr.io/scaleapi/"),
    (_disk(EXTRA="x"), "agentenv-ubuntu-24.04-disk: args is "),
    (_disk(SIZE_GB="32"), "agentenv-ubuntu-24.04-disk: SIZE_GB is '32', not its base's min_disk_gb, 16"),
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


@pytest.mark.parametrize("answer, code, output", [(_DIGEST, 0, "build=false\n"), ("ERROR: not found", 0, "build=true\n"),
                                                  ("ERROR: connection refused", 1, "")], ids=["served", "missing", "unknown"])
def test_wanted_signs_in_and_builds_only_a_tag_the_registry_says_it_lacks(script, monkeypatch, capsys, workflow, answer,
                                                                          code, output):
    calls, tokens = [], []
    docker = _docker({f"{REGISTRY}/agentenv-gateway:{_TAG}": answer}, calls)

    def run(command, **kw):
        tokens.append(kw.get("input"))
        return docker(command, **kw)

    monkeypatch.setattr(script.subprocess, "run", run)

    assert _run(script, monkeypatch, "wanted", _gateway()) == code

    assert (calls[0], tokens[0]) == (["docker", "login", "ghcr.io", "--username", "bot", "--password-stdin"], "t0ken")
    assert workflow.read_text() == output
    assert ("is published already, so it isn't built again" in capsys.readouterr().out) is (answer == _DIGEST)


@pytest.mark.parametrize("command", [["wanted"], ["build", "amd64"]])
def test_wanted_and_build_refuse_an_entry_before_signing_in_or_building(script, monkeypatch, capsys, workflow, command):
    calls, runs = [], []
    monkeypatch.setattr(script.subprocess, "run", _docker({}, calls))
    monkeypatch.setattr(script.subprocess, "Popen", lambda argv, **kw: runs.append(argv))

    assert _run(script, monkeypatch, command[0], _gateway(context="."), *command[1:]) == 1

    assert calls == runs == [] and workflow.read_text() == ""
    assert "::error::agentenv-gateway: context is '.'" in capsys.readouterr().out


def test_build_builds_an_image_for_its_platform_under_its_tag_with_its_source_on_it(script, monkeypatch, workflow):
    runs, waits = _attempts(script, monkeypatch, [(_LIMITED, 1), ("#1 DONE\n", 0)])

    assert _run(script, monkeypatch, "build", _gateway(), "arm64") == 0

    assert runs[1] == ["docker", "build", "--platform", "linux/arm64", "--provenance=false", "-f",
                       "src/agent_env/env/gateway/Dockerfile", "-t", f"{REGISTRY}/agentenv-gateway:{_TAG}-arm64",
                       "--label", "org.opencontainers.image.source=https://github.com/scaleapi/agentenv-framework",
                       "--label", "org.opencontainers.image.revision=c0ffee", "--label", f"io.agentenv.source-tag={_TAG}",
                       "src/agent_env/env/gateway"]
    assert runs[0] == runs[1] and waits == [script._RETRY_WAITS_SECONDS[0]]


def test_build_makes_a_disk_form_of_its_bases_image_of_the_arch_with_that_images_config_the_tools_and_its_size(
        script, monkeypatch, workflow):
    runs, _ = _attempts(script, monkeypatch, [("pulled\n", 0), ("#1 DONE\n", 0)])
    inspected = []
    config = '{"Env": ["A=1"]}'
    monkeypatch.setattr(script.subprocess, "run", lambda command, **kw: inspected.append(command) or (
        subprocess.CompletedProcess(command, 0, config + "\n", "")))
    rootfs = f"{REGISTRY}/agentenv-ubuntu-24.04:{_TAG}-amd64"

    assert _run(script, monkeypatch, "build", _disk(), "amd64") == 0

    assert runs[0] == ["docker", "pull", "--platform", "linux/amd64", rootfs]
    assert inspected == [["docker", "image", "inspect", "--format", "{{json .Config}}", rootfs]]
    assert runs[1][runs[1].index("-t"):runs[1].index("--label")] == [
        "-t", f"{REGISTRY}/agentenv-ubuntu-24.04-disk:{_TAG}-amd64", "--build-arg", f"ROOTFS={rootfs}",
        "--build-arg", f"TOOLS={REGISTRY}/agentenv-disk-tools:{_TAG}", "--build-arg", "SIZE_GB=16",
        "--build-arg", f"ROOTFS_CONFIG={base64.b64encode(config.encode()).decode()}"]


def test_publish_tags_each_new_images_platform_builds_as_one_and_leaves_a_published_tag_as_it_is(script, monkeypatch,
                                                                                                 capsys, workflow):
    published, new = f"{REGISTRY}/agentenv-gateway:{_TAG}", f"{REGISTRY}/agentenv-ubuntu-24.04-disk:{_TAG}"
    calls = []
    monkeypatch.setattr(script.subprocess, "run", _docker({published: _DIGEST}, calls))
    runs, _ = _attempts(script, monkeypatch, [("pushed\n", 0)])

    assert _run(script, monkeypatch, "publish", json.dumps([json.loads(_gateway()), json.loads(_disk())])) == 0

    assert calls[0] == ["docker", "login", "ghcr.io", "--username", "bot", "--password-stdin"]
    assert runs == [["docker", "buildx", "imagetools", "create", "-t", new,
                     "--annotation", "index:org.opencontainers.image.source=https://github.com/scaleapi/agentenv-framework",
                     "--annotation", "index:org.opencontainers.image.revision=c0ffee", f"{new}-amd64", f"{new}-arm64"]]
    assert f"{published} is published already, so it's left as it is" in capsys.readouterr().out


@pytest.mark.parametrize("first, served", [
    (_gateway(context="."), {}),
    (_gateway(), {f"{REGISTRY}/agentenv-gateway:{_TAG}": "ERROR: connection refused"}),
], ids=["refused", "unknown"])
def test_publish_tags_nothing_more_after_an_entry_it_refuses_or_whose_tag_it_cant_look_up(script, monkeypatch, workflow,
                                                                                         first, served):
    monkeypatch.setattr(script.subprocess, "run", _docker(served))
    runs, _ = _attempts(script, monkeypatch, [])

    assert _run(script, monkeypatch, "publish", json.dumps([json.loads(first), json.loads(_disk())])) == 1

    assert runs == []


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
    ({"images": []}, "records no agentenv-browser-use, agentenv-claude-code, agentenv-codex"),
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


@pytest.mark.parametrize("kvm, machine, how", [(True, "aarch64", "with KVM"), (False, "aarch64", "under TCG"),
                                               (True, "x86_64", "under TCG")], ids=["kvm", "no-kvm", "another-arch"])
def test_a_disk_forms_smoke_boots_it_in_the_launcher_until_its_desktop_runs(script, monkeypatch, capsys, kvm, machine,
                                                                           how):
    console = "".join(f"success: {program} entered RUNNING\n"
                      for program in ("clock", "egress", "session", "xvfb", "dbus", "desktop"))
    calls, logs = [], iter(["agentenv-init: starting", console])

    def run(command, **kw):
        calls.append((command, kw.get("input")))
        answers = {"inspect": "arm64\n", "run": "c1\n", "logs": next(logs) if command[1] == "logs" else ""}
        return subprocess.CompletedProcess(command, 0, answers.get(command[1], answers.get(command[2], "")), "")

    monkeypatch.setattr(script.subprocess, "run", run)
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(script.Path, "exists", lambda self: kvm)
    monkeypatch.setattr(script.platform, "machine", lambda: machine)

    assert _run(script, monkeypatch, "smoke", "agentenv-ubuntu-24.04-disk", "disk:1-arm64", "--launcher", "launcher:1") == 0

    commands = [command for command, _ in calls]
    assert calls[1] == (["docker", "build", "-q", "-t", "disk:1-arm64-machine", "-"],
                        "FROM --platform=linux/arm64 disk:1-arm64 AS disk\nFROM launcher:1\nCOPY --from=disk /disk/ /disk/\n")
    assert commands[2] == ["docker", "run", "-d", *(["--device", "/dev/kvm"] if kvm else []), "-e", "VM_ARCH=arm64",
                           "disk:1-arm64-machine"]
    assert commands[-1] == ["docker", "rm", "-f", "c1"] and commands.count(["docker", "logs", "c1"]) == 2
    assert f"booted to its desktop {how}" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["recorder", "session"])
def test_a_disk_forms_smoke_waits_for_every_program_its_bases_root_and_session_supervisords_declare(
        script, monkeypatch, capsys, tmp_path, missing):
    base = tmp_path / Path(script.IMAGES["agentenv-ubuntu-24.04"])
    base.parent.mkdir(parents=True)
    base.with_name("root.conf").write_text("[supervisord]\nnodaemon=true\n\n[program:session]\ncommand=supervisord\n")
    base.with_name("supervisord.conf").write_text("[supervisord]\nnodaemon=true\n\n[program:xvfb]\ncommand=Xvfb\n\n"
                                                  "[program:recorder]\ncommand=rec\n")
    monkeypatch.setattr(script, "REPO", tmp_path)
    monkeypatch.setattr(script, "_BOOTED_WITHIN_SECONDS", 0.01)
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    console = "".join(f"success: {program} entered RUNNING\n" for program in ("session", "xvfb", "recorder")
                      if program != missing)
    monkeypatch.setattr(script.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(
        command, 0, {"image": "amd64\n", "run": "c1\n", "logs": console}.get(command[1], ""), ""))

    assert _run(script, monkeypatch, "smoke", "agentenv-ubuntu-24.04-disk", "disk:1", "--launcher", "launcher:1") == 1

    assert "::error::disk:1 didn't boot to its desktop within 0.01s" in capsys.readouterr().out


def test_the_sessions_supervisord_logs_to_the_console_where_a_disk_forms_smoke_reads_its_programs_start(script):
    root = configparser.ConfigParser(interpolation=None)
    root.read_string((script.REPO / Path(script.IMAGES["agentenv-ubuntu-24.04"]).with_name("root.conf")).read_text())

    assert (root["program:session"]["stdout_logfile"], root["program:session"]["stdout_logfile_maxbytes"]) == (
        "/dev/stdout", "0")
    assert root["supervisord"]["nodaemon"] == "true"
    init = (script.REPO / Path(script.IMAGES["agentenv-ubuntu-24.04-disk"]).with_name("init")).read_text()
    assert "ln -sf /proc/self/fd/1 /newroot/dev/stdout\n" in init


def test_a_disk_forms_smoke_needs_the_launcher_to_boot_it_in(script, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exited:
        _run(script, monkeypatch, "smoke", "agentenv-ubuntu-24.04-disk", "disk:1")

    assert exited.value.code == 2 and "--launcher" in capsys.readouterr().err


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

    assert len(runs) == 3 and waits == list(script._RETRY_WAITS_SECONDS[:2])
    assert capsys.readouterr().out.count("::warning::a registry rate-limited `docker push img`; trying again in") == 2


def test_retry_tries_again_while_pypi_doesnt_list_the_protocol_a_build_installs(script, monkeypatch, capsys):
    """An agent image's build installs the agentenv-framework-protocol its release published moments earlier."""
    missing = ("#13 1.522 ERROR: No matching distribution found for agentenv-framework-protocol[agent]==0.1.359\n"
               "ERROR: failed to solve: process \"/bin/sh -c pip install\" did not complete successfully: exit code: 1\n")
    runs, waits = _attempts(script, monkeypatch, [(missing, 1), ("#13 DONE 4.1s\n", 0)])

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 0

    assert len(runs) == 2 and waits == [script._RETRY_WAITS_SECONDS[0]]
    assert ("::warning::PyPI doesn't list yet the agentenv-framework-protocol `docker build .` installs; trying again in"
            in capsys.readouterr().out)


def test_retry_tries_again_while_pypi_doesnt_list_the_protocol_floor_a_build_names(script, monkeypatch, capsys):
    """The Playwright MCP browser's recorder names a protocol floor, which its release may be the first to publish."""
    missing = "ERROR: No matching distribution found for agentenv-framework-protocol[computer]>=0.1.382\n"
    runs, waits = _attempts(script, monkeypatch, [(missing, 1), ("#13 DONE 4.1s\n", 0)])

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 0

    assert len(runs) == 2 and waits == [script._RETRY_WAITS_SECONDS[0]]


def test_retry_tries_again_a_push_ghcr_refuses_while_another_platform_creates_the_package(script, monkeypatch, capsys):
    """An image's first release pushes both platforms at once; the second waits on GHCR linking the new package."""
    runs, waits = _attempts(script, monkeypatch, [("denied: permission_denied: write_package\n", 1), ("pushed\n", 0)])

    assert _run(script, monkeypatch, "retry", "docker", "push", "img") == 0

    assert len(runs) == 2 and waits == [script._RETRY_WAITS_SECONDS[0]]
    assert ("::warning::GHCR doesn't let this repository write yet the package `docker push img` pushes to; trying again in"
            in capsys.readouterr().out)


def test_retry_doesnt_try_again_a_denied_command_that_isnt_a_push(script, monkeypatch, capsys):
    runs, waits = _attempts(script, monkeypatch, [("denied: permission_denied: write_package\n", 1)])

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 1

    assert len(runs) == 1 and waits == []


def test_retry_doesnt_try_again_a_build_missing_another_package(script, monkeypatch, capsys):
    runs, waits = _attempts(script, monkeypatch, [("ERROR: No matching distribution found for nonexistent-pkg==1.0\n", 1)])

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 1

    assert len(runs) == 1 and waits == []


def test_retry_gives_up_after_its_waits(script, monkeypatch, capsys):
    runs, waits = _attempts(script, monkeypatch, [(_LIMITED, 1)] * (len(script._RETRY_WAITS_SECONDS) + 1))

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 1

    assert len(runs) == len(script._RETRY_WAITS_SECONDS) + 1 and waits == list(script._RETRY_WAITS_SECONDS)


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


class _Log:
    """A pipe standing in for the workflow's log: what was written, and what had been flushed each time."""

    def __init__(self):
        self.written, self.flushed = "", []

    def write(self, text: str) -> int:
        self.written += text
        return len(text)

    def flush(self) -> None:
        self.flushed.append(self.written)


def test_retry_passes_on_each_line_and_its_warning_as_it_goes(script, monkeypatch):
    """Through a pipe, Python holds output until the buffer fills, so a build or a wait would look stuck."""
    runs, waits = _attempts(script, monkeypatch, [(_LIMITED, 1), ("#1 DONE\n#2 DONE\n", 0)])
    log = _Log()
    monkeypatch.setattr(script.sys, "stdout", log)

    assert _run(script, monkeypatch, "retry", "docker", "build", ".") == 0

    assert log.flushed[0] == _LIMITED
    assert log.flushed[1].endswith("trying again in 30s\n")
    assert log.flushed[2:] == [log.flushed[1] + "#1 DONE\n", log.flushed[1] + "#1 DONE\n#2 DONE\n"]
