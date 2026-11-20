"""JSON written so that equal values give equal bytes, for hashing them."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Optional


def canonical_json(value: Any, default: Optional[Callable[[Any], Any]] = None) -> bytes:
    """``value`` as JSON with its keys sorted and no spaces; ``default`` as for ``json.dumps``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default).encode()
