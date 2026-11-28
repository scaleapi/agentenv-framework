"""OpenHands' recording: the agent's image runs the OpenHands SDK in its own process, calling the model through the
LiteLLM SDK as ``litellm_proxy/<model>``, and persists each context's conversation, events and state, under
``~/.openhands-a2a/conversations``. Beside the common scenarios, a high reasoning effort, which Claude answers with
signed thinking blocks."""

SPEC = {
    "image": "agentenv-openhands",
    "model": "anthropic/claude-haiku-4-5",
    "files": "/home/agent/.openhands-a2a/conversations",
    "version": ["python", "-c", "import importlib.metadata as m; print(m.version('openhands-sdk'))"],
    "own": {
        "reasoning": {"config": {"effort": "high"}, "prompts": [
            "Think it through, then reply with only the answer: how many positive integers below 1000 are divisible "
            "by 3 or 5 but not by 15?"]},
    },
}
