"""Test doubles for the state-provider surface.

Core ships only ``local_postgres``; every external backend is installed through a
``[state.providers]`` config entry. The gateway's local-vs-external branches are therefore
exercised against this minimal external double rather than a concrete backend.
"""

import pytest

from agent_env.config import reset_config
from agent_env.providers.state import (
    DatabaseStateProvider,
    EnvStateInstance,
    StateContext,
)

EXTERNAL_STATE_TYPE = "external_db_test"
EXTERNAL_READINESS_SERVICE = "external-db-ready"


class ExternalDbStateProvider(DatabaseStateProvider):
    """A database that lives outside the gateway: no co-deployed store container, just a readiness
    probe the clients wait on — the shape every out-of-tree DB backend has."""

    type = EXTERNAL_STATE_TYPE

    def url_for_environment(self, environment_name: str, *, instance: EnvStateInstance) -> str:
        return f"{instance._db_url_base}&options=-csearch_path%3D%22{environment_name}%22"

    def env_state_docker_service_name(self) -> str:
        return EXTERNAL_READINESS_SERVICE

    def render_healthcheck_service(self, environment_names, *, instance=None) -> list[str]:
        return [
            f"  {EXTERNAL_READINESS_SERVICE}:",
            "    image: public.ecr.aws/docker/library/busybox",
            '    command: ["sleep", "infinity"]',
            "    healthcheck:",
            '      test: ["CMD", "true"]',
            "    networks:",
            "      - env-network",
            "",
        ]

    async def acquire(self, ctx: StateContext) -> EnvStateInstance:
        raise NotImplementedError

    async def _teardown(self, instance: EnvStateInstance) -> None:
        return None


@pytest.fixture
def registered_external_provider(monkeypatch, tmp_path):
    """Resolvable through ``build_state_provider``, the way an installed SDK registers a backend."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[state.providers.{EXTERNAL_STATE_TYPE}]\n'
                   f'impl = "tst.unit.providers.state.fakes:ExternalDbStateProvider"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reset_config()
    yield
    reset_config()
