"""OpenCode's atif/1 recording: its image, Haiku through the turn's model relay, each context's session store under
/data/sessions (its opencode.db holds the sub-agents' steps the CLI's events leave out), the pinned CLI's version, and
beside the common scenarios a sub-agent run through the task tool, and a high reasoning effort, whose thinking the CLI
prints for a task that builds atif/1."""

SPEC = {
    "image": "agentenv-opencode",
    "model": "anthropic/claude-haiku-4-5",
    "files": "/data/sessions",
    "version": ["opencode", "--version"],
    "own": {
        "subagent": {"prompts": [
            "Use your task tool to start a general sub-agent that runs `wc -l /etc/passwd` and reports the count back "
            "to you. Then reply with only that number."]},
        "subagents": {"prompts": [
            "In one response, call your task tool twice, so two general sub-agents run at once: one runs "
            "`echo one`, the other `echo two`, each reporting its output back. Then reply with both outputs on one "
            "line."]},
        "reasoning": {"config": {"effort": "high"}, "prompts": [
            "Think it through, then reply with only the answer: how many positive integers below 1000 are divisible "
            "by 3 or 5 but not by 15?"]},
    },
}
