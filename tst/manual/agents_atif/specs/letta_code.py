"""Letta Code's spec for ``record.py``: its default model, gpt-4.1-mini through Chat Completions at the turn's model
relay, and a sub-agent scenario of its own: the Task tool runs a sub-agent in the background, whose model calls only the
relay sees, and which the bridge stops once the turn ends. ``letta --version`` prints ``<version> (Letta Code)``, so the
version is its first word."""

SPEC = {
    "image": "agentenv-letta-code",
    "model": "openai/gpt-4.1-mini",
    "files": "/data/sessions",
    "version": ["sh", "-c", "letta --version | cut -d' ' -f1"],
    "own": {
        "subagent": {"prompts": [
            "Use the Task tool to start a sub-agent that runs `wc -l /etc/passwd` and reports the count. Wait for its "
            "report, then reply with only that number."]},
    },
}
