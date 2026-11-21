"""Write HCU's instructions and tools from a local release executable, without a desktop or network:

    python scripts/dump_hcu_mcp.py /path/to/computer-use src/agent_env/drivers/envs/hcu/mcp.json

Review the manifest alongside the release's URL and sha256 pins in env.toml and driver.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def dump(binary: Path, output: Path) -> None:
    parameters = StdioServerParameters(command=str(binary.resolve()), args=["mcp"])
    async with stdio_client(parameters) as (read, write), ClientSession(read, write) as session:
        initialized = await session.initialize()
        tools = (await session.list_tools()).tools
    kept = ("name", "description", "inputSchema", "annotations", "outputSchema")
    manifest = {"instructions": initialized.instructions or "", "tools": [
        {name: value for name, value in tool.model_dump(by_alias=True, exclude_none=True).items() if name in kept}
        for tool in tools]}
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{initialized.serverInfo.name} {initialized.serverInfo.version}: {', '.join(tool.name for tool in tools)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    asyncio.run(dump(arguments.binary, arguments.output))


if __name__ == "__main__":
    main()
