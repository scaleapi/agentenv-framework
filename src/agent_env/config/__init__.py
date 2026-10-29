"""The ``.agentenv/config.toml`` configuration surface: discovery, parsing,
``env:`` / ``secret:`` interpolation, and impl-pointer construction.

Consumed today by the store layer; designed to back the other ``.agentenv``
seams (task steps, sandbox, model gateway) as they are wired up.
"""

from agent_env.config.errors import ConfigError
from agent_env.config.loader import (
    ENV_REF_PREFIX,
    SECRET_REF_PREFIX,
    build_store,
    discover_config_path,
    interpolate,
    load_config_file,
    load_impl,
)
from agent_env.config.runtime import (
    Config,
    configure,
    gateway_env_id,
    get_config,
    get_runner,
    reset_config,
    run_env_ids,
    service_db_env_id,
    set_document_store,
    set_image_store,
    set_object_store,
    set_runner,
    set_secret_store,
)

__all__ = [
    "ConfigError",
    "ENV_REF_PREFIX",
    "SECRET_REF_PREFIX",
    "build_store",
    "discover_config_path",
    "interpolate",
    "load_config_file",
    "load_impl",
    "Config",
    "configure",
    "gateway_env_id",
    "get_config",
    "get_runner",
    "reset_config",
    "run_env_ids",
    "service_db_env_id",
    "set_document_store",
    "set_image_store",
    "set_object_store",
    "set_runner",
    "set_secret_store",
]
