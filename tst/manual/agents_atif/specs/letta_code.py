"""Letta Code's spec for ``record.py``: its default model, gpt-4.1-mini through Chat Completions at the turn's model
relay, and a sub-agent scenario of its own: the Agent tool runs a sub-agent in the background, a CLI of its own whose
model calls only the relay sees and whose transcript the bridge feeds once the turn's processes are stopped, so the
prompt has the agent wait for its report. ``letta --version`` prints ``<version> (Letta Code)``, so the version is its
first word."""

SPEC = {
    "image": "agentenv-letta-code",
    "model": "openai/gpt-4.1-mini",
    "files": "/data/sessions",
    "version": ["sh", "-c", "letta --version | cut -d' ' -f1"],
    "own": {
        "subagent": {"prompts": [
            "Use the Agent tool, with subagent_type general-purpose, to start a sub-agent that runs "
            "`wc -l /etc/passwd` and reports the count. It runs in the background: wait for its report by reading the "
            "output file its result names with your shell tool, sleeping between reads, until the count is there. "
            "Then reply with only that number."]},
    },
}
