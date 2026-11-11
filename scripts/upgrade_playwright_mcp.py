"""Pin the page driver's Playwright MCP to a version: its install's downloads in env.toml (Node's Linux tarballs and
every npm package Playwright MCP needs, each with its sha256) and its tools, as that version lists them with ``--caps
vision``, in playwright-mcp-tools.json. Run from anywhere, with network:

    python scripts/upgrade_playwright_mcp.py 0.0.85 [--node 24.22.2]

Node keeps the version env.toml pins unless ``--node`` names another. The install unpacks the exact versions written
here, so a package that names a range for a dependency is refused. The tools that are new or gone are printed: look at
each new one, hide it in playwright_mcp.py's HIDDEN_TOOLS if it runs the agent's own code, and add it to the unit test's
REVIEWED_TOOLS."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "src" / "agent_env" / "drivers" / "envs" / "page"
ENV_TOML = PAGE / "env.toml"
MANIFEST = PAGE / "playwright-mcp-tools.json"
REGISTRY = "https://registry.npmjs.org"
NODE_DIST = "https://nodejs.org/dist"
PACKAGE = "@playwright/mcp"
CAPS = ["vision"]
#: Node's Linux builds, by the architecture a computer names.
NODE_LINUX = {"amd64": "linux-x64", "arm64": "linux-arm64"}
EXACT = re.compile(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?")
TIMEOUT_S = 120


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", help="the @playwright/mcp version to pin, such as 0.0.85")
    parser.add_argument("--node", help="the Node version to pin; by default the one env.toml pins")
    args = parser.parse_args(argv)
    node = args.node or re.search(r"nodejs\.org/dist/v([\d.]+)/", ENV_TOML.read_text()).group(1)
    sums = _node_sums(node)
    packages = _packages(args.version)
    downloads = [{"url": f"{NODE_DIST}/v{node}/node-v{node}-{build}.tar.gz", "sha256": sums[build],
                  "to": "node.tar.gz", "arch": arch} for arch, build in NODE_LINUX.items()]
    downloads += [{"url": p["url"], "sha256": hashlib.sha256(p["bytes"]).hexdigest(), "to": _download_name(p["name"])}
                  for p in packages]
    tools = _listed_tools(node, sums, packages)
    before = {t["name"] for t in json.loads(MANIFEST.read_text(encoding="utf-8"))["tools"]}
    _write_downloads(downloads)
    MANIFEST.write_text(json.dumps({"version": args.version, "caps": CAPS, "tools": tools}, indent=1) + "\n",
                        encoding="utf-8")
    after = {t["name"] for t in tools}
    print(f"pinned @playwright/mcp {args.version} on Node {node}: "
          + ", ".join(f"{p['name']} {p['version']}" for p in packages))
    print("new tools, to look at:", ", ".join(sorted(after - before)) or "none")
    print("tools gone:", ", ".join(sorted(before - after)) or "none")


def _packages(version: str) -> list[dict]:
    """@playwright/mcp at ``version`` and every package it needs, each with its tarball, checked against npm's
    integrity."""
    wanted, found = [(PACKAGE, version)], {}
    while wanted:
        name, version = wanted.pop(0)
        if name in found:
            if found[name]["version"] != version:
                raise SystemExit(f"{name} is needed at both {found[name]['version']} and {version}")
            continue
        listed = json.loads(_get(f"{REGISTRY}/{urllib.parse.quote(name, safe='@')}/{version}"))
        tarball = _get(listed["dist"]["tarball"])
        algorithm, digest = listed["dist"]["integrity"].split("-", 1)
        if base64.b64encode(hashlib.new(algorithm, tarball).digest()).decode() != digest:
            raise SystemExit(f"{name} {version}'s tarball doesn't match npm's integrity")
        found[name] = {"name": name, "version": version, "url": listed["dist"]["tarball"], "bytes": tarball}
        for dependency, spec in (listed.get("dependencies") or {}).items():
            if not EXACT.fullmatch(spec):
                raise SystemExit(f"{name} {version} needs {dependency} {spec}, a range; the install takes exact versions")
            wanted.append((dependency, spec))
    return list(found.values())


def _node_sums(node: str) -> dict[str, str]:
    """Each of Node ``node``'s builds' sha256 by its name, such as linux-x64, from nodejs.org's SHASUMS256.txt."""
    sums = {}
    for line in _get(f"{NODE_DIST}/v{node}/SHASUMS256.txt").decode().splitlines():
        digest, file = line.split()
        if (m := re.fullmatch(rf"node-v{re.escape(node)}-(.+)\.tar\.gz", file)):
            sums[m.group(1)] = digest
    return sums


def _listed_tools(node: str, sums: dict[str, str], packages: list[dict]) -> list[dict]:
    """Playwright MCP's ``tools/list``, run on this machine's Node of the same version, laid out as the install lays it
    out, attached to no browser."""
    build = _host()
    archive = _get(f"{NODE_DIST}/v{node}/node-v{node}-{build}.tar.gz")
    if hashlib.sha256(archive).hexdigest() != sums[build]:
        raise SystemExit(f"Node {node}'s {build} build doesn't match SHASUMS256.txt")
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root / "node", filter="data")
        executable = next((root / "node").glob("node-*/bin/node"))
        for package in packages:
            with tarfile.open(fileobj=io.BytesIO(package["bytes"])) as tar:
                for member in tar.getmembers():
                    member.name = member.name.split("/", 1)[1] if "/" in member.name else ""
                tar.extractall(root / "node_modules" / package["name"],
                               members=[m for m in tar.getmembers() if m.name], filter="data")
        messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "upgrade_playwright_mcp", "version": "0"}}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        out = subprocess.run([str(executable), str(root / "node_modules" / PACKAGE / "cli.js"), "--cdp-endpoint",
                              "http://127.0.0.1:9/", "--caps", ",".join(CAPS)], capture_output=True, text=True,
                             input="\n".join(map(json.dumps, messages)) + "\n", timeout=TIMEOUT_S, check=True).stdout
    return next(r["result"]["tools"] for r in map(json.loads, out.splitlines()) if r.get("id") == 2)


def _host() -> str:
    """This machine's Node build, such as darwin-arm64."""
    machine = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
    return f"{sys.platform}-{machine}"


def _download_name(package: str) -> str:
    return package.lstrip("@").replace("/", "-") + ".tgz"


def _write_downloads(downloads: list[dict]) -> None:
    lines = "".join("    { " + ", ".join(f'{key} = "{value}"' for key, value in d.items()) + " },\n" for d in downloads)
    text, count = re.subn(r"^downloads = \[\n.*?^\]\n", lambda _: f"downloads = [\n{lines}]\n", ENV_TOML.read_text(),
                          count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise SystemExit(f"{ENV_TOML} has no downloads list to replace")
    ENV_TOML.write_text(text)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "agentenv-framework"}),
                                timeout=TIMEOUT_S) as response:
        return response.read()


if __name__ == "__main__":
    main(sys.argv[1:])
