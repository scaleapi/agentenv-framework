"""Antigravity's recording: the image's A2A server runs the pinned SDK, whose harness calls the model through the turn's
relay over Chat Completions; its conversations, the env times and images its turns were given are under
``$ANTIGRAVITY_HOME``, ``/data/.antigravity`` in the image. Beside the common scenarios, a sub-agent the SDK's
``invoke_subagent`` starts, which runs on after the call returns, and a high effort, which the relay sends as
``reasoning_effort``, for the SDK's Thought chunks; Gemini 2.5 Flash streamed none through the relay. The model is Gemini
2.5 Flash since the harness asks for 65,535 output tokens, more than Claude Haiku 4.5 takes."""

SPEC = {
    "image": "agentenv-antigravity",
    "model": "gemini-2.5-flash",
    "files": "/data/.antigravity",
    "version": ["python3", "-c", "import importlib.metadata as m; print(m.version('google-antigravity'))"],
    "own": {
        "subagent": {"prompts": [
            "Use your invoke_subagent tool to have a subagent create notes.txt holding the word forty-two and report "
            "back, then run `cat notes.txt` yourself and reply with only what it prints."]},
        "reasoning": {"config": {"effort": "high"}, "prompts": [
            "Think it through, then reply with only the 20th Fibonacci number, counting F(1) = F(2) = 1."]},
    },
}
