"""Run dispatch: the seam between "run this task" and ``Task.run()``."""

from agent_env.runner.runner import RunHandle, RunRecord, Runner, RunStatus

__all__ = ["RunHandle", "RunRecord", "Runner", "RunStatus", "LocalRunner", "get_runner"]


def __getattr__(name: str):
    # Lazy so importing the package does not pull the store/config chain.
    if name == "LocalRunner":
        from agent_env.runner.local_runner import LocalRunner
        return LocalRunner
    if name == "get_runner":
        from agent_env.config import get_config
        return get_config().get_runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
