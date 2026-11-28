"""pi's spec for ``record.py``: Claude Haiku through Chat Completions at the endpoint, and two scenarios of its own: a
thinking level on a turn with no tool call, since Claude's thinking with tools wants its signed blocks back and pi's
Chat Completions client keeps no signature; and a compaction, whose summary call pi reports only in its
``compaction_end``. pi compacts only past its 16,384-token reserve, and then only a conversation longer than the 20,000
tokens it keeps, by its estimate of four characters a token, so the first turn's two outputs of about 47 KB each pass
both in a 32,000-token window, and the second turn runs on the compacted session."""

SPEC = {
    "image": "agentenv-pi",
    "model": "anthropic/claude-haiku-4-5",
    "files": "/home/agent/.pi-a2a/sessions",
    "version": ["pi", "--version"],
    "own": {
        "thinking": {"config": {"effort": "low"}, "prompts": [
            "What is 17 times 23? Work it out step by step, then reply with only the number."]},
        "compaction": {"config": {"context_window": 32000, "max_tokens": 4096}, "prompts": [
            "Run `seq -f '%g: the quick brown fox jumps over the lazy dog' 1 1000` with your bash tool. After it "
            "finishes, run `seq -f '%g: pack my box with five dozen liquor jugs' 1001 2000` with it, then reply with "
            "only the last line the second command printed.",
            "Now run `seq 1 5` with it, and reply with only the last number it printed."]},
    },
}
