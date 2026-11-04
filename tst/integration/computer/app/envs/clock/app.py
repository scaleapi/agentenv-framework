"""A clock app for the computer env tests, composed beside notes: its service answers on a port of its own with the
time zone its settings name, and writes it to ~/.clock.json. It installs tzdata, which its run script lists the zones
of, and declares no window. Its data plane is its own hooks over its alarms, kept in ~/.clock-alarms.json. Its own
server also answers, for the driver's /svc/clock/ route, with the headers it got, a 20 MB stream and a WebSocket
that echoes until "bye"."""
import json
from pathlib import Path

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

from agentenv_protocol import DataPart, add_data, get_data, reset_data
from agentenv_protocol.computer import (ComputerAppEnvironment, Packages, RunScript, Service, computer_app, linux,
                                        part_bytes)

PORT = 8137
ALARMS = ".clock-alarms.json"
MB = 1 << 20


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["tzdata"]), python="requirements.txt", run=[RunScript(script="setup.sh")],
       services=[Service(name="clock", command="venv/bin/python serve.py")], ports=[PORT])
class Clock(ComputerAppEnvironment):
    class Settings(BaseModel):
        zone: str = "UTC"

    @add_data
    async def load(self, parts: list) -> None:
        for part in parts:
            (Path.home() / ALARMS).write_bytes(await part_bytes(part))

    @get_data
    def alarms(self) -> list:
        path = Path.home() / ALARMS
        return [DataPart(data=json.loads(path.read_text()) if path.exists() else {"alarms": []})]

    @reset_data
    def clear(self) -> None:
        (Path.home() / ALARMS).unlink(missing_ok=True)

    def create_app(self):
        mcp = super().create_app()
        mcp.custom_route("/headers", methods=["GET"])(_headers)
        mcp.custom_route("/big", methods=["GET"])(_big)
        mcp._custom_starlette_routes.append(WebSocketRoute("/echo", _echo))
        return mcp


async def _headers(request: Request) -> Response:
    return JSONResponse({"host": request.headers.get("host"), "headers": dict(request.headers.items())})


async def _big(request: Request) -> Response:
    async def chunks():
        for n in range(20):
            yield bytes([65 + n]) * MB
    return StreamingResponse(chunks(), media_type="application/octet-stream")


async def _echo(websocket: WebSocket) -> None:
    await websocket.accept(subprotocol="cdp" if "cdp" in websocket.scope.get("subprotocols", []) else None)
    await websocket.send_text(json.dumps({"prefix": websocket.headers.get("x-forwarded-prefix"),
                                          "authorization": websocket.headers.get("authorization")}))
    while (message := await websocket.receive())["type"] != "websocket.disconnect":
        if message.get("text") == "bye":
            await websocket.close(code=4001)
            return
        await (websocket.send_text(message["text"]) if message.get("text") is not None
               else websocket.send_bytes(message["bytes"][::-1]))
