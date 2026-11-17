"""The strict JSON schema of one yatf/1 event, for producers in other languages: generated from the models, with every
field and type this version doesn't define forbidden, and checked in as ``yatf-1.schema.json``:

    python -m agentenv_protocol.yatf schema > packages/agentenv-protocol/src/agentenv_protocol/yatf/yatf-1.schema.json
"""
from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from pydantic import TypeAdapter

from .models import FORMAT, SPEC_VERSION, AnyEvent

SCHEMA_FILE = "yatf-1.schema.json"


def json_schema() -> dict[str, Any]:
    """The schema of one event, generated from the models."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"{FORMAT} event, spec {SPEC_VERSION}", **TypeAdapter(AnyEvent).json_schema()}


def schema_text() -> str:
    """The schema as it's checked in."""
    return json.dumps(json_schema(), indent=2, sort_keys=True) + "\n"


def load_schema() -> dict[str, Any]:
    """The checked-in schema."""
    return json.loads(files(__package__).joinpath(SCHEMA_FILE).read_text(encoding="utf-8"))
