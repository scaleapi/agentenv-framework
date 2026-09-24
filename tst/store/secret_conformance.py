"""Backend-neutral SecretStore conformance assertions.

Each function takes a ``store`` pre-seeded with ``FIXTURE`` (a present key, a
present-but-empty key) and expected to NOT contain ``ABSENT``. Every backend must
pass every case in ``CASES``.
"""

PRESENT = "conformance_present"
PRESENT_EMPTY = "conformance_present_empty"
ABSENT = "conformance_absent"

# Seed a conforming backend with this mapping before running CASES.
FIXTURE = {PRESENT: "value-1", PRESENT_EMPTY: ""}


def get_returns_value(store):
    assert store.get(PRESENT) == "value-1"


def get_returns_none_for_absent(store):
    assert store.get(ABSENT) is None


def get_returns_empty_string_verbatim(store):
    # An empty-but-present value is distinct from absent — get must not collapse it to None.
    assert store.get(PRESENT_EMPTY) == ""


CASES = [
    get_returns_value,
    get_returns_none_for_absent,
    get_returns_empty_string_verbatim,
]
