# An X11 computer driver

[`driver.py`](driver.py) is a complete computer driver for an X11 desktop, written on
`ComputerDriverEnvironment` from `agentenv_protocol.computer`: `mss` captures the screen and XTEST,
through `python-xlib`, sends the input. It implements the seven required operations and the pointer
and keyboard groups, declares five buttons, triple clicks and 50-point drags, and adds a ready check
of its own for XTEST. The base class serves everything else from those: the `computer_*` tools,
frames and coordinate spaces, the input queue, the framework extensions and the card, and its Linux
defaults give the driver `computer-exec/v1` and `data/add`.

The image runs the driver on Xvfb beside [`fixture.py`](fixture.py), the Fixture Desk: a Tk window
with a name field, a checkbox, a 100-item list, a grid of cells, a drawing pad and a Save button. It
records whatever reaches it in `~/fixture/state.json`, so what an agent did can be read back from
the app rather than from the driver's answers, and shows the text of `~/inbox/message.txt`, which
`data/add` can write. [`start.sh`](start.sh) restarts Xvfb and the app if either dies.

## Run it

```bash
cd packages/agentenv-protocol
docker build -f examples/computer_x11/Dockerfile -t x11-desktop .
docker run --rm -p 18765:18765 x11-desktop
```

The card is at `http://localhost:18765/.well-known/agent-env.json`, readiness at
`/agentenv/ext/computer-ready/status`, and the tools at the MCP endpoint `/mcp`:

```python
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main() -> None:
    async with streamablehttp_client("http://localhost:18765/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            shot = await session.call_tool("computer_screenshot", {})
            print(shot.structuredContent["summary"])
            await session.call_tool("computer_click", {"x": 300, "y": 52})
            typed = await session.call_tool("computer_type", {"text": "Ada Lovelace", "screenshot": True})
            print(typed.structuredContent["summary"])


asyncio.run(main())
```

`AGENTENV_COMPUTER_SETTINGS` takes the driver's settings as one JSON value, for example
`-e AGENTENV_COMPUTER_SETTINGS='{"default_space": "norm", "data_paths": ["fixture"]}'`.

## In agent-env

As a bundle env (`envs/<name>/Dockerfile`), the image is built from the env's own folder, so copy
the three files there and install `agentenv-framework-protocol[computer]` in place of this checkout.
Name the env with `environment_name` in its `env.toml`. A task then deploys it with `deploy_env`
and gives it to an agent with `deploy_agent`, as for any MCP env.

`tests/test_computer_x11_example.py` runs the driver against fake `mss` and `Xlib` modules, so it
keeps up with the SDK without an X server.
