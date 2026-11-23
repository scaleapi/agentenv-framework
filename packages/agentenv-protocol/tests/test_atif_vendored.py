"""Harbor's ATIF models, vendored: the same code as Harbor 0.26.0's but for import paths, as strict as Harbor's, with its
license beside them, importable with only the protocol's base dependencies."""
import hashlib
import re
import subprocess
import sys
from importlib.resources import files

import pytest

from agentenv_protocol.atif import AtifError, Step, Trajectory, check

MODELS = files("agentenv_protocol.atif.models")
LOCAL_IMPORT = re.compile(r"^(\s*(?:from|import)\s+)agentenv_protocol\.atif\.models\b", re.M)
HEADER = re.compile(r"\A# From Harbor 0\.26\.0, harbor/models/trajectories/(\S+) \(Apache-2\.0\), with its import paths "
                    r"changed: see VENDORED\.md\.\n")


def recorded() -> dict[str, str]:
    table = MODELS.joinpath("VENDORED.md").read_text(encoding="utf-8")
    return dict(re.findall(r"^\| `([^`]+\.py)` \| `([0-9a-f]{64})` \|$", table, re.M))


def test_the_vendored_files_say_they_were_changed_and_differ_from_harbors_only_in_import_paths():
    shipped = {p.name for p in MODELS.iterdir() if p.name.endswith(".py")}
    assert shipped == set(recorded()) and len(shipped) == 11
    for name, digest in recorded().items():
        text = MODELS.joinpath(name).read_text(encoding="utf-8")
        header = HEADER.match(text)
        assert header and header.group(1) == name, name
        assert "harbor" not in "".join(line for line in text.splitlines() if line.lstrip().startswith(("from", "import")))
        upstream = LOCAL_IMPORT.sub(r"\g<1>harbor.models.trajectories", text[header.end():])
        assert hashlib.sha256(upstream.encode("utf-8")).hexdigest() == digest, name


def test_harbors_license_ships_beside_them():
    assert "Apache License" in MODELS.joinpath("LICENSE").read_text(encoding="utf-8")


def test_atif_fields_stay_as_strict_as_harbors():
    with pytest.raises(ValueError):
        Step.model_validate({"step_id": 1, "source": "user", "message": "x", "later_field": 1})
    with pytest.raises(AtifError):
        check({"schema_version": "ATIF-v9.9", "agent": {"name": "a", "version": "1"},
               "steps": [{"step_id": 1, "source": "user", "message": "x"}]})
    assert Trajectory.model_validate({"agent": {"name": "a", "version": "1"},
                                      "steps": [{"step_id": 1, "source": "user", "message": "x"}]})


def test_atif_imports_with_only_the_protocols_base_dependencies():
    probe = ("import sys\nimport agentenv_protocol.atif\n"
             "assert not any(m == 'a2a' or m.startswith('a2a.') for m in sys.modules), 'a2a imported'\n"
             "assert not any(m.startswith('agentenv_protocol.a2a_agent') for m in sys.modules), 'a2a_agent imported'\n")
    subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True)
