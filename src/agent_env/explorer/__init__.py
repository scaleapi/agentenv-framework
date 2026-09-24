"""The local agent-env explorer (FastAPI). Requires the ``explorer`` extra."""

from agent_env.explorer.plugin import ExplorerPlugin, load_plugins

__all__ = ["ExplorerPlugin", "load_plugins", "create_app"]


def __getattr__(name: str):
    if name == "create_app":                 # lazy: keeps FastAPI off the import path
        from agent_env.explorer.app import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
