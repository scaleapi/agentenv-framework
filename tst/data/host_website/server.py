"""A plain website, not on WebsiteEnvironment, that builds its addresses from its Host header, as many real apps do, for
integration-testing the gateway's site routes: /login redirects to /home at its Host, and /api/host says the Host and
X-Forwarded-Host it was reached with. Its setting ``card_domains``, names split by commas, makes it serve a card
declaring them; without it, it serves none."""
import html
import json
import os

import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from agentenv_protocol.website import WEBSITE_SETTINGS_ENV
from agentenv_protocol.website.readiness import WEBSITE_EXTENSION_URI

DOMAINS = [name for name in str(json.loads(os.environ.get(WEBSITE_SETTINGS_ENV) or "{}").get("card_domains", "")
                                ).split(",") if name]


async def card(_):
    if not DOMAINS:
        return Response(status_code=404)
    return JSONResponse({"name": os.environ.get("ENVIRONMENT_NAME", "shop"), "capabilities": {"extensions": [
        {"uri": WEBSITE_EXTENSION_URI, "params": {"domains": DOMAINS}}]}})


async def health(_):
    return JSONResponse({"status": "ok"})


async def login(request):
    return RedirectResponse(f"https://{request.headers.get('host', '')}/home", status_code=302)


async def home(request):
    return HTMLResponse(f"<!doctype html><title>home</title><p>{html.escape(request.headers.get('host', ''))}</p>")


async def host(request):
    return JSONResponse({"host": request.headers.get("host"),
                         "forwarded_host": request.headers.get("x-forwarded-host")})


app = Starlette(routes=[Route("/.well-known/agent-env.json", card), Route("/api/health", health),
                        Route("/login", login), Route("/home", home), Route("/api/host", host)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), proxy_headers=False, log_level="warning")
