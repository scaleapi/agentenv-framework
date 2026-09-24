"""`attribution_of` reads a step's `metadata["attribution"]` and nothing else; persisted
documents written with the flat quartet are folded on read by `metadata_from_legacy_document`.
"""

from agent_env.attribution import attribution_of, metadata_from_legacy_document

_LEGACY = {"product": "p", "customer": "c", "team": "t", "project_id": "id"}


def _step(**attrs):
    return type("Step", (), attrs)()


def test_attribution_of_is_empty_when_metadata_is_unset():
    assert attribution_of(_step()) == {}
    assert attribution_of(_step(metadata=None)) == {}
    assert attribution_of(_step(metadata={"run_label": "nightly"})) == {}


def test_attribution_of_reads_the_reserved_metadata_key_only():
    step = _step(metadata={"attribution": {"cost_center": "research"}, "run_label": "nightly"})
    assert attribution_of(step) == {"cost_center": "research"}


def test_attribution_of_ignores_flat_attributes_on_the_step():
    step = _step(product="p", project_id="id", metadata={"attribution": {"team": "t"}})
    assert attribution_of(step) == {"team": "t"}


def test_attribution_of_returns_a_copy():
    metadata = {"attribution": {"team": "t"}}
    attribution_of(_step(metadata=metadata))["team"] = "mutated"
    assert metadata["attribution"] == {"team": "t"}


def test_a_legacy_document_folds_its_flat_keys_into_metadata():
    assert metadata_from_legacy_document({**_LEGACY, "metadata": {"run_label": "nightly"}}) == {
        "run_label": "nightly",
        "attribution": _LEGACY,
    }


def test_a_legacy_document_without_metadata_still_folds():
    assert metadata_from_legacy_document(_LEGACY) == {"attribution": _LEGACY}
    assert metadata_from_legacy_document({**_LEGACY, "metadata": None}) == {"attribution": _LEGACY}


def test_null_flat_keys_are_not_attribution():
    doc = {"product": None, "customer": None, "team": None, "project_id": None, "metadata": {}}
    assert metadata_from_legacy_document(doc) == {}
    assert metadata_from_legacy_document({"metadata": {"attribution": {"team": "t"}}}) == {
        "attribution": {"team": "t"},
    }


def test_the_metadata_sub_key_wins_over_a_flat_key_on_conflict():
    doc = {"project_id": "flat", "team": "t", "metadata": {"attribution": {"project_id": "authored"}}}
    assert metadata_from_legacy_document(doc) == {"attribution": {"project_id": "authored", "team": "t"}}


def test_folding_does_not_mutate_the_document():
    doc = {"team": "t", "metadata": {"attribution": {"project_id": "authored"}}}
    metadata_from_legacy_document(doc)
    assert doc == {"team": "t", "metadata": {"attribution": {"project_id": "authored"}}}
