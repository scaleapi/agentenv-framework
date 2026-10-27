"""The unit-mcp2 job's test split, as pytest arguments: ``leave-out`` prints the ``--ignore`` options, ``fastmcp`` the
protocol test files that import the standalone fastmcp, which run in a pass of their own because fastmcp 4 adds mcp 1's
camel-case attributes to mcp's types once imported.

Left out: the tests that import the gateway server, which serves on mcp 1's FastMCP inside its own image, directly or
through a test module that does; the gateway's and openenv's own tests; and the two packaging tests that read every
extra's installed metadata, openenv's included, which the unit job runs."""

import re
import sys
from pathlib import Path

UNIT = Path("tst/unit")
GATEWAY_TESTS = UNIT / "env" / "gateway"
ALWAYS = [GATEWAY_TESTS, UNIT / "openenv", UNIT / "test_optional_extras_stay_optional.py",
          UNIT / "test_runtime_dependencies_are_imported.py"]
GATEWAY_IMPORT = re.compile(r"agent_env\.env\.gateway(\.gateway\b| import gateway\b)|mcp\.server\.fastmcp")
FASTMCP_IMPORT = re.compile(r"^(import fastmcp|from fastmcp|fastmcp = pytest\.importorskip)", re.MULTILINE)


def main(what: str) -> None:
    if what == "leave-out":
        print(" ".join(f"--ignore={path}" for path in [*ALWAYS, *_gateway_importers()]))
    else:
        tests = sorted(Path("packages/agentenv-protocol/tests").glob("test_*.py"))
        print(" ".join(str(path) for path in tests if FASTMCP_IMPORT.search(path.read_text())))


def _gateway_importers() -> list[Path]:
    sources = {path: path.read_text() for path in UNIT.rglob("*.py") if not path.is_relative_to(GATEWAY_TESTS)}
    found = {path for path, text in sources.items() if GATEWAY_IMPORT.search(text)}
    modules = {".".join(path.with_suffix("").parts) for path in found}
    found |= {path for path, text in sources.items() if any(f"from {module} import" in text for module in modules)}
    return sorted(found)


if __name__ == "__main__":
    main(sys.argv[1])
