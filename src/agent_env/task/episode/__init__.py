"""Run a task with some of its agents played from outside AgentEnv, by a training loop's model or any other caller.

AgentEnv runs everything else exactly as ``Task.run`` does, so its scores are the ones ``agent-env run`` would report.
"""

from .run import Episode, EpisodeError, Turn
from .spec import EpisodeSpec

__all__ = ["Episode", "EpisodeError", "EpisodeSpec", "Turn"]
