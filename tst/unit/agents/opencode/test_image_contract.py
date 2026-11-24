"""The agent's folder is its build context: the Dockerfile copies only its own files, pins Node by its checksum and the
CLI by package-lock.json, keeps TLS checks on, and installs the protocol at or above its floor, which a release
doesn't pin; install.sh brings the same Node and the same CLI."""

from __future__ import annotations

import ast
import json
import re
import tomllib

from tst.unit.agents.opencode.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
SOURCES = AGENT_DIR.parents[2]
SHARED = tomllib.loads((AGENT_DIR / "shared_sources.toml").read_text())
INSTALL_SH = (AGENT_DIR / "install.sh").read_text()


def _copied() -> list[str]:
    return [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
            for name in line.split()]


def _source(name: str):
    """Where the build context's ``name`` comes from: the folder, or a folder shared_sources.toml takes from agent-env's
    own sources at that name."""
    first, _, rest = name.partition("/")
    return SOURCES / SHARED[first] / rest if first in SHARED else AGENT_DIR / name


def _in_context(name: str) -> bool:
    path = _source(name)
    return path.is_dir() if name.endswith("/") else path.is_file()


def test_every_file_the_image_copies_is_in_the_folder():
    copied = _copied()

    assert copied and all(_in_context(name) for name in copied), copied
    # install.sh and run.sh are install/v1's; a build context pruned to what the Dockerfile reads keeps them so.
    assert {"a2a_server.py", "opencode_bridge.py", "opencode_kit.py", "entrypoint.sh", "install.sh", "run.sh",
            "package.json", "package-lock.json", "requirements.txt"} <= set(copied)


def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_and_an_install_carry_every_module_the_server_imports():
    """The folder's own modules and those the build context takes from agent-env's sources, the egress rule and the
    clock page's format among them, each by its name in the context."""
    shared = {path.stem: f"{name}/{path.name}" for name, source in SHARED.items()
              for path in (SOURCES / source).glob("*.py")}
    modules = {**shared, **{path.stem: path.name for path in AGENT_DIR.glob("*.py")}}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(_source(name)) & set(modules)
    installed = set(re.search(r'^AGENT_FILES="([^"]+)"$', INSTALL_SH, re.MULTILINE).group(1).split())

    assert {"agent_clock/agent_egress.py", "clock/clockpage.py"} <= needed
    assert needed <= set(_copied()) and needed <= installed, (needed, installed)


def test_the_requirements_are_the_protocol_floor_the_output_format_validator_the_relays_http_and_the_ca_bundle():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert requirements == ["agentenv-framework-protocol[agent]>=0.1.643", "jsonschema>=4.18,<5", "aiohttp>=3.9,<4",
                            "certifi"]


def test_node_is_pinned_by_version_and_checksum_for_the_image_and_an_install():
    args = dict(re.findall(r"^ARG (NODE_\w+)=(\S+)$", DOCKERFILE, re.MULTILINE))

    assert re.fullmatch(r"\d+\.\d+\.\d+", args["NODE_VERSION"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", args[name]) for name in ("NODE_SHA256_X64", "NODE_SHA256_ARM64"))
    assert "sha256sum -c" in DOCKERFILE
    for name in ("NODE_VERSION", "NODE_SHA256_X64", "NODE_SHA256_ARM64"):
        assert f"dockerfile_arg {name}" in INSTALL_SH


def test_the_cli_is_installed_from_the_lockfile_exactly_with_its_build_for_the_target_alone():
    package = json.loads((AGENT_DIR / "package.json").read_text())
    lock = json.loads((AGENT_DIR / "package-lock.json").read_text())

    assert all(re.fullmatch(r"\d+\.\d+\.\d+", version) for version in package["dependencies"].values())
    assert lock["packages"][""]["dependencies"] == package["dependencies"]
    for name, version in package["dependencies"].items():
        assert lock["packages"][f"node_modules/{name}"]["version"] == version
    assert {f"node_modules/opencode-linux-{cpu}" for cpu in ("x64", "arm64")} <= set(lock["packages"])
    assert all(entry["resolved"].startswith("https://registry.npmjs.org/")
               for path, entry in lock["packages"].items() if path)
    for text in (DOCKERFILE, INSTALL_SH):
        # The npm bin is a stub until opencode-ai's own postinstall links the build in; no other package's scripts run.
        assert "npm ci --ignore-scripts --omit=dev" in text
        assert "node node_modules/opencode-ai/postinstall.mjs" in text
        assert "-name 'opencode-*' ! -name opencode-ai -exec rm -rf {} +" in text


def test_tls_checks_stay_on():
    for name in ("Dockerfile", "entrypoint.sh", "install.sh", "run.sh", "opencode_bridge.py", "a2a_server.py"):
        assert "NODE_TLS_REJECT_UNAUTHORIZED=0" not in (AGENT_DIR / name).read_text()
        assert '"NODE_TLS_REJECT_UNAUTHORIZED": "0"' not in (AGENT_DIR / name).read_text()


def test_the_image_starts_through_cmd_which_modal_passes_its_env_to():
    assert 'CMD ["/usr/local/bin/start-agent"]' in DOCKERFILE
    assert not re.search(r"^ENTRYPOINT", DOCKERFILE, re.MULTILINE)


def test_opencodes_state_is_fixed_under_data_whatever_the_uid():
    env = dict(re.findall(r"(\w+)=(\S+)", re.search(r"^ENV (.+?)\n(?=\S)", DOCKERFILE, re.MULTILINE | re.DOTALL).group(1)))

    assert {key: env[key] for key in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
                                      "XDG_STATE_HOME")} == {
        "HOME": "/data", "XDG_CONFIG_HOME": "/data/config", "XDG_CACHE_HOME": "/data/cache",
        "XDG_DATA_HOME": "/data/share", "XDG_STATE_HOME": "/data/state"}
    assert "poppler-utils" in DOCKERFILE
