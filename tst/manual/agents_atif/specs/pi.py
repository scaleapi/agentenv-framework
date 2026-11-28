"""pi's spec for ``record.py``: Claude Haiku through Chat Completions at the endpoint, and two scenarios of its own: a
thinking level on a turn with no tool call, since Claude's thinking with tools wants its signed blocks back and pi's
Chat Completions client keeps no signature, and a window small enough, past pi's 16,384-token reserve, that a coding
turn's output sets off a compaction, whose summary call pi reports only in its ``compaction_end``."""

SPEC = {
    "image": "agentenv-pi",
    "model": "anthropic/claude-haiku-4-5",
    "files": "/home/agent/.pi-a2a/sessions",
    "version": ["pi", "--version"],
    "own": {
        "thinking": {"config": {"effort": "low"}, "prompts": [
            "What is 17 times 23? Work it out step by step, then reply with only the number."]},
        "compaction": {"config": {"context_window": 24000, "max_tokens": 4096}, "prompts": [
            "Run `seq 1 3000` with your bash tool, then reply with only the last number it printed.",
            "Now run `seq 1 3000 | head -n 5` with it, and reply with only the first number it printed."]},
    },
}
