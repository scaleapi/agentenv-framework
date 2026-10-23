"""The agent's folder is its build context: the Dockerfile copies only its own files, pins Node by its checksum and the
CLI by package-lock.json, keeps TLS checks on, and installs the protocol at or above its floor, which a release doesn't
pin; install.sh brings the same Node and the same CLI."""

from __future__ import annotations

import ast
import json
import re

from tst.unit.agents.gemini.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
INSTALL_SH = (AGENT_DIR / "install.sh").read_text()


def _copied() -> list[str]:
    return [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
            for name in line.split()]


def test_every_file_the_image_copies_is_in_the_folder():
    copied = _copied()

    assert copied and all((AGENT_DIR / name).is_file() for name in copied), copied
    # install.sh and run.sh are install/v1's; a build context pruned to what the Dockerfile reads keeps them so.
    assert {"a2a_server.py", "gemini_bridge.py", "gemini_changelog.py", "gemini_config.py", "gemini_kit.py",
            "entrypoint.sh", "install.sh", "run.sh", "package.json", "package-lock.json"} <= set(copied)


def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_and_an_install_carry_every_module_the_server_imports():
    modules = {path.stem: path for path in AGENT_DIR.glob("*.py")}
    needed = {"a2a_server.py"} | {f"{name}.py" for path in modules.values() for name in _imported(path) & set(modules)}
    installed = set(re.search(r'^AGENT_FILES="([^"]+)"$', INSTALL_SH, re.MULTILINE).group(1).split())

    assert needed <= set(_copied()) and needed <= installed, (needed, installed)


def test_the_image_installs_the_protocol_alone_from_its_requirements():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert requirements == ["agentenv-framework-protocol[agent]>=0.1.351"]


def test_node_is_pinned_by_version_and_checksum_for_the_image_and_an_install():
    args = dict(re.findall(r"^ARG (NODE_\w+)=(\S+)$", DOCKERFILE, re.MULTILINE))

    assert re.fullmatch(r"\d+\.\d+\.\d+", args["NODE_VERSION"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", args[name]) for name in ("NODE_SHA256_X64", "NODE_SHA256_ARM64"))
    assert "sha256sum -c" in DOCKERFILE
    for name in ("NODE_VERSION", "NODE_SHA256_X64", "NODE_SHA256_ARM64"):
        assert f"dockerfile_arg {name}" in INSTALL_SH


def test_the_cli_is_installed_from_the_lockfile_exactly():
    package = json.loads((AGENT_DIR / "package.json").read_text())
    lock = json.loads((AGENT_DIR / "package-lock.json").read_text())

    assert all(re.fullmatch(r"\d+\.\d+\.\d+", version) for version in package["dependencies"].values())
    assert lock["packages"][""]["dependencies"] == package["dependencies"]
    for name, version in package["dependencies"].items():
        assert lock["packages"][f"node_modules/{name}"]["version"] == version
    assert all(entry["resolved"].startswith("https://registry.npmjs.org/")
               for path, entry in lock["packages"].items() if path)
    for text in (DOCKERFILE, INSTALL_SH):
        assert "npm ci --ignore-scripts --omit=dev" in text


def test_tls_checks_stay_on():
    for name in ("Dockerfile", "entrypoint.sh", "install.sh", "run.sh", "gemini_config.py", "gemini_bridge.py",
                 "a2a_server.py"):
        assert "NODE_TLS_REJECT_UNAUTHORIZED=0" not in (AGENT_DIR / name).read_text()
        assert '"NODE_TLS_REJECT_UNAUTHORIZED": "0"' not in (AGENT_DIR / name).read_text()
        assert "verify=False" not in (AGENT_DIR / name).read_text()


def test_the_image_starts_through_cmd_which_modal_passes_its_env_to():
    assert 'CMD ["/opt/agent/entrypoint.sh"]' in DOCKERFILE
    assert not re.search(r"^ENTRYPOINT", DOCKERFILE, re.MULTILINE)
