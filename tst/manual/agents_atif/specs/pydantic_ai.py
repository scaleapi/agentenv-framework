"""Pydantic AI's spec for ``record.py``: the agent's image runs Pydantic AI in its own process, calling Claude Haiku
through Chat Completions at the endpoint, streamed, and persists each context's messages under
``~/.pydantic-ai-a2a/conversations``. Beside the common scenarios, two of its own: a high reasoning effort on a turn with
no tool call, which Claude answers with thinking that comes back through LiteLLM as ``reasoning_content``, unsigned;
and a ``delegate_task`` run, which runs the agent again with messages of its own that only the ``delegated`` aux file
holds. The common ``retried`` scenario's 503s are retried by the OpenAI client inside one model call, which the messages
never show, so it checks only that the copy still sums to the turn."""

SPEC = {
    "image": "agentenv-pydantic-ai",
    "model": "anthropic/claude-haiku-4-5",
    "files": "/home/agent/.pydantic-ai-a2a",
    "version": ["python", "-c", "import pydantic_ai; print(pydantic_ai.__version__)"],
    "own": {
        "reasoning": {"config": {"effort": "high"}, "prompts": [
            "Think it through, then reply with only the answer: how many positive integers below 1000 are divisible "
            "by 3 or 5 but not by 15?"]},
        "subagent": {"prompts": [
            "Use your delegate_task tool with agent_name \"self\" to have a sub-agent run `wc -l /etc/passwd` with its "
            "shell tool and report the count back to you. Then reply with only that number."]},
    },
}
