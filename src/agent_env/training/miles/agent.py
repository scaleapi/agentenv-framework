"""What Miles calls for each sample of the rows ``python -m agent_env.training.miles`` writes: the agent function
(``--custom-agent-function-path agent_env.training.miles.agent.run``), whose session is Miles' session server, the
reward hook (``--custom-rm-path agent_env.training.miles.agent.reward``) and the dynamic-sampling filter
(``--dynamic-sampling-filter-path agent_env.training.miles.agent.keep_scored_groups``). Imports nothing of Miles: it
calls them by their arguments."""

from agent_env.training._chat import keep_scored_groups, reward
from agent_env.training._chat import play as run

__all__ = ["keep_scored_groups", "reward", "run"]
