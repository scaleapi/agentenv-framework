"""Env-state providers — abstract WHERE an environment's data/state lives"""

from agent_env.providers.state.env_state_provider import (
    DatabaseStateProvider,
    DEFAULT_STATE_TTL_SECONDS,
    EnvStateInstance,
    EnvStateProvider,
    StateContext,
    acquire_state_for_deploy,
    admin_config_from_env,
    build_state_provider,
    LOCAL_POSTGRES_STATE_TYPE,
)
from agent_env.providers.state.local_postgres import (
    LocalPostgresStateContext,
    LocalPostgresStateProvider,
    LocalPostgresStoreSpec,
)
from agent_env.providers.state.store import (
    ENV_STATE_INSTANCES_COLLECTION,
    EnvStateInstanceStore,
    get_env_state_instance_store,
    register_env_state_instance,
    reset_env_state_instance_store,
    retire_env_state_instance,
    set_env_state_instance_store,
    update_env_state_instance_metadata,
)

__all__ = [
    "DatabaseStateProvider",
    "DEFAULT_STATE_TTL_SECONDS",
    "EnvStateInstance",
    "EnvStateProvider",
    "StateContext",
    "acquire_state_for_deploy",
    "admin_config_from_env",
    "build_state_provider",
    "LOCAL_POSTGRES_STATE_TYPE",
    "LocalPostgresStateContext",
    "LocalPostgresStateProvider",
    "LocalPostgresStoreSpec",
    "ENV_STATE_INSTANCES_COLLECTION",
    "EnvStateInstanceStore",
    "get_env_state_instance_store",
    "set_env_state_instance_store",
    "reset_env_state_instance_store",
    "register_env_state_instance",
    "retire_env_state_instance",
    "update_env_state_instance_metadata",
]
