"""The agent's folder is its build context: every file the Dockerfile copies is in it, or in a folder its
``shared_sources.toml`` takes from agent-env's sources, and the image and an install both carry every module the A2A
server imports, the egress rule and the clock page's format among them, and the mod the bridge installs."""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib

from tst.unit.agents.letta_code.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
SOURCES = AGENT_DIR.parents[2]
SHARED = tomllib.loads((AGENT_DIR / "shared_sources.toml").read_text())
INSTALL_SH = (AGENT_DIR / "install.sh").read_text()


def _copied() -> list[str]:
    return [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
            for name in line.split()]


def _installed() -> set[str]:
    return set(re.search(r'^AGENT_FILES="([^"]+)"$', INSTALL_SH, re.MULTILINE).group(1).split())


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
    assert {"a2a_server.py", "letta_code_bridge.py", "letta_code_kit.py", "mcp_tools.ts", "entrypoint.sh", "install.sh",
            "run.sh", "package.json", "package-lock.json", "requirements.txt"} <= set(copied)


def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_and_an_install_carry_every_module_the_server_imports():
    shared = {path.stem: f"{name}/{path.name}" for name, source in SHARED.items()
              for path in (SOURCES / source).glob("*.py")}
    modules = {**shared, **{path.stem: path.name for path in AGENT_DIR.glob("*.py")}}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(_source(name)) & set(modules)
    # Not imported: the bridge copies it into each context as the mod that registers the env's MCP tools.
    needed.add("mcp_tools.ts")

    assert {"agent_clock/agent_egress.py", "clock/clockpage.py", "letta_code_bridge.py"} <= needed
    assert needed <= set(_copied()) and needed <= _installed(), (needed, _installed())
    assert all(_in_context(name) for name in _installed())


def test_an_install_puts_every_file_by_its_own_name_in_the_one_folder_run_sh_imports_from():
    """A file the build context takes from a shared folder, ``agent_clock/agent_egress.py`` say, lands beside the
    server, as the image copies it, not in a folder of that name off the path."""
    start = INSTALL_SH.index("for file in $AGENT_FILES; do")
    loop = INSTALL_SH[start:INSTALL_SH.index("\ndone\n", start) + len("\ndone")]
    agent_files = re.search(r"^AGENT_FILES=.*$", INSTALL_SH, re.MULTILINE).group(0)
    script = f'{agent_files}\nhere=/ctx root=/r\nput() {{ printf "%s %s\\n" "$1" "$2"; }}\n{loop}\n'

    placed = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout.splitlines()

    assert sorted(placed) == sorted(f"/ctx/{name} /r/app/{name.rsplit('/', 1)[-1]}" for name in _installed())
    assert 'PYTHONPATH="$root/app"' in (AGENT_DIR / "run.sh").read_text()
