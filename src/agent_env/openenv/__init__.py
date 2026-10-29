"""Every task of a bundle as an OpenEnv env (https://github.com/openenv/OpenEnv): ``build_app`` returns the server
any OpenEnv client resets, steps and scores. The only part of agent-env that imports ``openenv``, which the ``openenv``
extra installs."""

try:
    from .app import build_app
except ModuleNotFoundError as exc:
    if exc.name not in ("fastapi", "openenv"):
        raise
    raise ImportError("agent_env.openenv requires pip install 'agentenv-framework[openenv]'") from exc

__all__ = ["build_app"]
