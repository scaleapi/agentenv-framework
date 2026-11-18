"""The agents bundle's proxy: its card, how it installs on a VM's host, and how the episode writes it."""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.bundle.installed import write_installed
from agent_env.task.episode.steps import PROXY_AGENT_ID
from agent_env.task_step.task_steps.install_agent import _build_param_resolvers

from tst.unit.agents.proxy.agent import AGENT_DIR, a2a_server
from tst.util.proxy_bake import BAKE_RECIPE, VERSIONS

README = Path(__file__).resolve().parents[4] / "README.md"


def _extensions() -> dict:
    card = a2a_server.AGENT_CARD.model_dump(mode="json", exclude_none=True, by_alias=True)
    return {ext["uri"]: ext for ext in card["capabilities"]["extensions"]}


def test_the_card_offers_the_proxy_and_an_install_on_a_host_or_into_a_container():
    extensions = _extensions()
    assert {"urn:agentenv:proxy/v1", "urn:agentenv:mcp-config/v1", "urn:agentenv:trajectory/v1",
            "urn:agentenv:install/v1"} <= set(extensions)
    params = extensions["urn:agentenv:install/v1"]["params"]
    assert params["install_commands_host"] and params["install_commands"]


@pytest.mark.parametrize(("commands", "required", "container", "script"), [
    ("install_commands_host", "required_params_host", None,
     "bash {work_dir}/agent/install.sh host {agent_name} {a2a_port} {host_workdir} {host_user}"),
    ("install_commands", "required_params", "task", "bash {work_dir}/agent/install.sh container {container} {a2a_port}"),
])
def test_each_install_asks_only_for_params_install_agent_supplies_and_runs_the_folders_script(commands, required,
                                                                                              container, script):
    params = _extensions()["urn:agentenv:install/v1"]["params"]
    supplied = set(_build_param_resolvers(container_name=container, work_dir="/w", agent_ctx_tar="/w/a.tgz",
                                          a2a_port=8000, config=None, agent_name="a"))
    placeholders = {name for command in params[commands] for name in re.findall(r"\{(\w+)\}", command)}
    assert placeholders == set(params[required]) and placeholders <= supplied
    assert params[commands][-1] == script


def test_the_install_script_parses_and_the_image_keeps_it_in_its_build_context():
    subprocess.run(["bash", "-n", str(AGENT_DIR / "install.sh")], check=True)
    for script in ("bake.sh", "versions.sh"):
        subprocess.run(["sh", "-n", str(AGENT_DIR / script)], check=True)
    assert "COPY a2a_server.py install.sh ./" in (AGENT_DIR / "Dockerfile").read_text()


def test_the_readme_and_bake_sh_give_the_bake_recipe_for_the_versions_the_install_brings():
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", VERSIONS["UV_IMAGE_DIGEST"])
    assert "\n".join(BAKE_RECIPE) in README.read_text()
    assert "\n".join(f"#   {line}" for line in BAKE_RECIPE) in (AGENT_DIR / "bake.sh").read_text()


def test_the_image_works_in_app_as_agent():
    lines = (AGENT_DIR / "Dockerfile").read_text().splitlines()
    last = {key: next(line.split(maxsplit=1)[1] for line in reversed(lines) if line.startswith(f"{key} "))
            for key in ("USER", "WORKDIR")}
    assert last == {"USER": "agent", "WORKDIR": "/app"}


def test_the_episode_writes_it_from_the_bundle_taking_no_credentials(local_stores, cli_routing):
    version = write_installed(PROXY_AGENT_ID)
    agent = A2AAgent.get(PROXY_AGENT_ID)

    assert (agent.version, agent.metadata) == (version, {"takes_credentials": False})
    assert agent.docker_image_artifact.context_only
    assert write_installed(PROXY_AGENT_ID) == version


@pytest.mark.parametrize("baked", [None, "0" * 64], ids=["no-proxy-baked", "other-requirements-baked"])
def test_a_container_without_this_proxy_baked_in_gets_uv_and_a_venv(tmp_path, baked):
    calls, exit_code = _install(tmp_path, baked)

    assert exit_code != 0 and calls[-1].startswith("curl -fsSL --retry 3 https://github.com/astral-sh/uv/releases/")
    assert not [call for call in calls if call.startswith("docker exec -d")]


def test_a_container_with_this_proxy_baked_in_only_starts_it(tmp_path):
    requirements = hashlib.sha256((AGENT_DIR / "requirements.txt").read_bytes()).hexdigest()

    calls, exit_code = _install(tmp_path, requirements)

    assert exit_code == 0 and not [call for call in calls if call.startswith("curl")]
    assert [call.split()[3] for call in calls if call.startswith("docker cp")] == [
        "task:/opt/agentenv-proxy-8000/a2a_server.py"]
    assert calls[-1].endswith("exec /opt/agentenv-proxy/venv/bin/python /opt/agentenv-proxy-8000/a2a_server.py "
                              "> /opt/agentenv-proxy-8000/proxy.log 2>&1")


@pytest.mark.parametrize(("cached", "downloads"), [("good", 0), (b"not uv", 1), (None, 1)],
                         ids=["a-good-copy", "a-corrupt-copy", "no-copy"])
def test_a_full_install_takes_uv_from_the_hosts_copy_unless_it_isnt_good(tmp_path, cached, downloads):
    archive = _uv_archive(tmp_path)

    calls, exit_code = _install(tmp_path, None, archive=archive, cached=archive.read_bytes() if cached == "good" else cached)

    assert exit_code == 0 and len([call for call in calls if call.startswith("curl")]) == downloads
    assert _cached(tmp_path).read_bytes() == archive.read_bytes()
    assert [call.split()[3] for call in calls if call.startswith("docker cp")] == [
        "task:/opt/agentenv-proxy-8000/bin/uv", "task:/opt/agentenv-proxy-8000/requirements.txt",
        "task:/opt/agentenv-proxy-8000/a2a_server.py"]
    assert calls[-1].endswith("exec /opt/agentenv-proxy-8000/venv/bin/python /opt/agentenv-proxy-8000/a2a_server.py "
                              "> /opt/agentenv-proxy-8000/proxy.log 2>&1")


def _install(tmp_path, baked: str | None, *, archive: Path | None = None,
             cached: bytes | None = None) -> tuple[list[str], int]:
    """``install.sh container task 8000`` against a fake docker whose container's image baked a proxy holding the
    requirements digest ``baked``, or none, with the host holding ``cached`` as its copy of uv, and a curl that serves
    ``archive`` as uv, pinned to its checksum, or fails without one: every docker and curl call, and the exit code."""
    agent, bin_dir, log = tmp_path / "agent", tmp_path / "bin", tmp_path / "calls.log"
    shutil.copytree(AGENT_DIR, agent, ignore=shutil.ignore_patterns("__pycache__"))
    if archive is not None:
        versions = agent / "versions.sh"
        versions.write_text(re.sub(r"^UV_SHA256_X86_64=\S+$", f"UV_SHA256_X86_64={_sha256(archive.read_bytes())}",
                                   versions.read_text(), flags=re.M))
    if cached is not None:
        _cached(tmp_path).parent.mkdir(parents=True)
        _cached(tmp_path).write_bytes(cached)
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(f"""#!/bin/sh
printf 'docker %s\\n' "$*" >> {log}
case "$*" in
  *"cat /opt/agentenv-proxy/baked"*) [ -n "$FAKE_BAKED" ] && echo "$FAKE_BAKED" ;;
  *"uname -m"*) echo x86_64 ;;
esac
exit 0
""")
    (bin_dir / "curl").write_text(f"""#!/bin/sh
printf 'curl %s\\n' "$*" >> {log}
[ -n "$FAKE_UV" ] || exit 22
while [ $# -gt 1 ]; do [ "$1" = -o ] && cp "$FAKE_UV" "$2"; shift; done
""")
    for stub in ("docker", "curl"):
        (bin_dir / stub).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_BAKED": baked or "",
           "FAKE_UV": str(archive or ""), "HOME": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path / "cache")}
    done = subprocess.run(["bash", str(agent / "install.sh"), "container", "task", "8000"], capture_output=True, env=env)
    return log.read_text().splitlines(), done.returncode


def _uv_archive(tmp_path) -> Path:
    """A uv release archive whose uv is a stub."""
    archive = tmp_path / "uv.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        uv = tarfile.TarInfo("uv-x86_64-unknown-linux-musl/uv")
        uv.size, uv.mode = 10, 0o755
        tar.addfile(uv, io.BytesIO(b"#!/bin/sh\n"))
    return archive


def _cached(tmp_path) -> Path:
    return tmp_path / "cache" / "agentenv-proxy" / f"{VERSIONS['UV_VERSION']}-uv-x86_64-unknown-linux-musl.tar.gz"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
