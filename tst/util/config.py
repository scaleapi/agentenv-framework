"""Test-side helpers for the config surface.

`config_with_document` lives here rather than in `agent_env.config`: nothing in the shipped
package needs a way to choose a document without writing a file, so shipping one would be
production surface with no production caller. A test is allowed to know internals the
package should not expose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from agent_env.config import runtime, snapshot


def config_with_document(document: Mapping[str, Any], *, path: Optional[Path] = None,
                         **kwargs: Any) -> runtime.Config:
    """A `Config` reading `document`, with no file on disk and no discovery.

    Returns the Config rather than installing one process-wide: a Config holds its own
    document, so a caller that built its own would not see a process-wide install.
    """
    config = runtime.Config(**kwargs)
    config._snapshot = snapshot.Snapshot(path=path, _document=document)
    return config
