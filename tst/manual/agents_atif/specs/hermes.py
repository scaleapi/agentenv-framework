"""Hermes' spec for ``record.py``: Claude Haiku through Chat Completions at the endpoint, each context's Hermes profile
(its sessions database, memory and SOUL.md) under ``/data/.hermes/profiles``, and three scenarios of its own: a high
reasoning effort, which Hermes sends as ``--reasoning``; a ``delegate_task`` child, whose events come under a session
of its own; and a batch of two children from one call. ``hermes --version`` prints ``Hermes Agent vunknown
(<release date>)`` first, as the image drops the checkout's history Hermes would read its tag from: the date, which
``make_fixtures.py`` keeps as the fixtures' ``cli``, is the release the Dockerfile's ``HERMES_COMMIT`` pins and the
adapter's ``TESTED_WITH`` names. With claude-haiku-4-5 at that endpoint a high effort gives no reasoning back, and the
common ``retried`` scenario's two failures reach no model call: its events hold no failed attempt."""

SPEC = {
    "image": "agentenv-hermes",
    "model": "anthropic/claude-haiku-4-5",
    "files": "/data/.hermes/profiles",
    "version": ["sh", "-c", "hermes --version | head -n 1"],
    "own": {
        "reasoning": {"config": {"effort": "high"}, "prompts": [
            "Think it through, then reply with only the answer: how many positive integers below 1000 are divisible "
            "by 3 or 5 but not by 15?"]},
        "subagent": {"prompts": [
            "Use the delegate_task tool to have a sub-agent run `wc -l /etc/passwd` and report the count back to you. "
            "Then reply with only that number."]},
        "batch": {"prompts": [
            "Make one delegate_task call with two tasks: one sub-agent runs `wc -l /etc/passwd`, the other runs "
            "`uname -s`, each reporting its output. Then reply with both outputs on one line."]},
    },
}
