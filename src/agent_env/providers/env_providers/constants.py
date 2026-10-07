"""The gateway deployment's layout: its compose service name, VM app dir and compose path, and website names.

Imports nothing from agent_env, so modules in the ``agent_env.env`` import chain can use these while
``agent_env.providers`` is still importing.
"""

GATEWAY_SERVICE_NAME = "gateway"
# The compose services a gateway deploy runs beside its envs' MCP servers, so no env can be named one of them: the
# gateway, the local Postgres state's database and its two browse UIs, and the website browser (named in
# agent_env.env.envs.website_browser).
GATEWAY_SERVICE_NAMES = frozenset({GATEWAY_SERVICE_NAME, "servicedb", "pgweb", "db-mcp", "website-browser"})
GATEWAY_APP_DIR = "/app"

AGENT_ENV_WEBSITE_BACKEND_PORT = 8000
AGENT_ENV_WEBSITE_BACKEND_SUFFIX = "website-backend"
AGENT_ENV_WEBSITE_FRONTEND_SUFFIX = "website-frontend"
DOCKER_COMPOSE_PATH = f"{GATEWAY_APP_DIR}/docker-compose.yml"
