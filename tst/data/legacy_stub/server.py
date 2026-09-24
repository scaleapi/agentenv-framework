"""Minimal legacy MCP-style server (NO agent-env card) for testing data-loader.sh's
legacy branch: serves /api/reset (loads mock_data_path), /api/export_state, /api/health."""
import json
import os

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

_STATE: dict = {}


async def reset(request):
    body = await request.json() if await request.body() else {}
    path = body.get("mock_data_path")
    _STATE.clear()
    if path:
        with open(path) as f:
            _STATE.update(json.load(f))
    return JSONResponse({"ok": True, "loaded": _STATE})


async def export_state(request):
    return JSONResponse(_STATE)


async def health(request):
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[
    Route("/api/reset", reset, methods=["POST"]),
    Route("/export-state", export_state, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MCP_PORT", "18765")))
