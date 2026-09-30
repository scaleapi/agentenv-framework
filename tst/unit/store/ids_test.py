"""The encoding of namespaced ids into object keys, image repositories and filenames.

The golden table is frozen: persisted ``object_url``s and ``image_name``s record these strings,
so a change here orphans every local cache written before it."""

import hashlib
import re

import pytest

from agent_env.store.ids import (
    LOCAL_PREFIX,
    MAX_AUTHORED_LOCAL_ID_BYTES,
    derive_id,
    derived_id,
    fs_safe,
    image_repository,
    is_local_id,
    key_segment,
    parse_namespace,
    validate_local_id,
)

_OCI_REPOSITORY = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*")

GOLDEN = [
    ("@local/~/work/triage/tickets", "local/work-triage-tickets-40da2b3590cf"),
    ("@local/~/work/triage/tickets__image", "local/work-triage-tickets-image-e15ef2bc2446"),
    ("@local/~/My Work/Tickets V2", "local/my-work-tickets-v2-ee679b8e5d3a"),
    ("@local/~/my work/tickets v2", "local/my-work-tickets-v2-7755dd71f0c1"),
    ("@local/agentenv-framework/hello/greeting", "local/agentenv-framework-hello-greeting-cdbe393f04fc"),
    ("@local/Users/bob/Code/Proj/envs/Tickets", "local/users-bob-code-proj-envs-tickets-ac37d78c8636"),
    ("@local/~/" + "x" * 60, "local/" + "x" * 48 + "-afe4bbb1ceec"),
    ("@local/~/" + "a" * 47 + "/b", "local/" + "a" * 47 + "-58f5a6c2bc11"),
    ("@local/~/!!!", "local/bbd9d3a308e9"),
    ("@local/~/ÉtéCafé/日本", "local/etecafe-17fc7e175f85"),
    ("@local/~/Straße/Ångström", "local/strasse-angstrom-2c208f63f689"),
    ("@local/~/ＡＢＣ/ﬁle", "local/abc-file-8652699b786c"),
    ("@local/~/परियोजना/env", "local/env-894f55d4f6c0"),
    ("@local/~/Dropbox (Personal)/tickets", "local/dropbox-personal-tickets-ba7c1450dae3"),
    ("@local/~/Library/CloudStorage/GoogleDrive-a.b@example.com/My Drive/triage/tickets",
     "local/library-cloudstorage-googledrive-a-b-example-com-656d87047b33"),
    ("@local/~/C++/R&D/a,b", "local/c-r-d-a-b-8daef6df1655"),
]

LEGACY = [
    "universe-real",
    "UPPER-Case_Id",
    "morrowline-live-v0/attendee-export.yaml",
    "a.b..c",
    "tickets__image",
    "with space",
    "hash#frag?q",
    "user@example.com",
    "x" * 300,
    "",
    " ",
    "__",
    "@acme/tickets",
    "@LOCAL/upper-namespace",
]


@pytest.mark.parametrize("entity_id, segment", GOLDEN)
def test_local_ids_encode_to_the_frozen_segment(entity_id, segment):
    assert segment.startswith("local/")
    assert segment.endswith(hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:12])
    assert key_segment(entity_id) == segment
    assert image_repository(entity_id) == segment
    assert fs_safe(entity_id) == segment.replace("/", "-", 1)
    assert _OCI_REPOSITORY.fullmatch(image_repository(entity_id))
    assert "/" not in fs_safe(entity_id)


def test_the_hash_covers_the_full_unnormalized_id():
    assert key_segment("@local/~/My Work/Tickets V2") != key_segment("@local/~/my work/tickets v2")
    long_a, long_b = "@local/~/" + "x" * 60 + "a", "@local/~/" + "x" * 60 + "b"
    assert key_segment(long_a) != key_segment(long_b)


@pytest.mark.parametrize("entity_id", LEGACY)
def test_every_other_id_passes_through_every_helper_byte_identical(entity_id):
    assert key_segment(entity_id) == entity_id
    assert image_repository(entity_id) == entity_id
    assert fs_safe(entity_id) == entity_id


@pytest.mark.parametrize("entity_id, namespace, local", [
    ("tickets", None, False),
    ("user@example.com", None, False),
    ("@local/~/a", "local", True),
    ("@local", "local", False),
    ("@acme/tickets", "acme", False),
    ("@", "", False),
])
def test_parse_namespace_and_is_local_id(entity_id, namespace, local):
    assert parse_namespace(entity_id) == namespace
    assert is_local_id(entity_id) is local


def test_a_malformed_local_id_is_not_encoded_as_one():
    assert key_segment("@local") == "@local"


@pytest.mark.parametrize("entity_id", [
    "@local/~/work/triage/tickets",
    "@local/a",
    "@local/~/My Work/Tickets V2",
    "@local/~/ÉtéCafé/日本",
    "@local/~/परियोजना/env",
    "@local/~/cafe\u0301/nai\u0308ve",
    "@local/" + "x" * 4089,
    "@local/" + "é" * 2044,
    "@local/~/Library/CloudStorage/GoogleDrive-a.b@example.com/My Drive/triage",
    "@local/~/Dropbox (Personal)/triage copy (2)",
    "@local/~/C++/R&D/a,b+c",
    "@local/~/node_modules/@types/x",
])
def test_well_formed_local_ids_validate(entity_id):
    validate_local_id(entity_id)


@pytest.mark.parametrize("entity_id, reason", [
    ("@local", "must start with"),
    ("@acme/x", "must start with"),
    ("tickets", "must start with"),
    ("@local/", "empty"),
    ("@local//a", "empty"),
    ("@local/a/", "empty"),
    ("@local/./a", "'.'"),
    ("@local/a/..", "'..'"),
    ("@local/a\x00b", "contains '\\x00'"),
    ("@local/a\nb", "contains '\\n'"),
    ("@local/a\tb", "contains '\\t'"),
    ("@local/a\x7f", "contains '\\x7f'"),
    ("@local/a\ud800", "contains '\\ud800'"),
    ("@local/a#b", "contains '#'"),
    ("@local/$(rm)", "contains '$'"),
    ("@local/a`b`", "contains '`'"),
    ("@local/a\\b", "contains '\\\\'"),
    ("@local/a\"b", "contains '\"'"),
    ("@local/it's", "contains \"'\""),
    ("@local/a;b|c&d", "contains ';|'"),
    ("@local/a:b", "contains ':'"),
    ("@local/a?b%c", "contains '%?'"),
    ("@local/a*b", "only letters, digits, spaces and . _ - ~ / @ ( ) + , &"),
    ("@local/a\u3164b", "contains '\\u3164'"),
    ("@local/a\ufe0fb", "contains '\\ufe0f'"),
    ("@local/a\u034fb", "contains '\\u034f'"),
    ("@local/a\U000e0100b", "contains '\\U000e0100'"),
    ("@local/\u0301x", "combining mark that does not follow"),
    ("@local/a \u0301", "combining mark that does not follow"),
    ("@local/a/\u0301", "combining mark that does not follow"),
    ("@local/~/tickets ", "opens or closes with a space"),
    ("@local/ tickets", "opens or closes with a space"),
    ("@local/~/ /tickets", "opens or closes with a space"),
    ("@local/" + "x" * 4090, "4097 bytes; the limit is 4096"),
    ("@local/" + "é" * 2045, "4097 bytes; the limit is 4096"),
])
def test_malformed_local_ids_are_refused(entity_id, reason):
    with pytest.raises(ValueError, match=re.escape(reason)):
        validate_local_id(entity_id)


def test_a_derived_id_keeps_its_base_namespace():
    assert derive_id("@local/~/work/triage/tickets", "image") == "@local/~/work/triage/tickets__image"
    assert is_local_id(derive_id("@local/~/a", "files"))
    assert derive_id("tickets", "files") == "tickets__files"


def test_a_name_keeps_an_local_bases_namespace_and_a_bare_bases_legacy_spelling():
    assert derived_id("@local/~/work/triage/tickets", "validate-v2", legacy="validate-@local/~/work/triage/tickets-v2") == (
        "@local/~/work/triage/tickets__validate-v2"
    )
    assert derived_id("tickets", "validate-v2", legacy="validate-tickets-v2") == "validate-tickets-v2"
    assert derived_id("", "snapshot-snap", legacy="snapshot-tickets-ab12cd34") == "snapshot-tickets-ab12cd34"


def test_authored_cap_leaves_room_for_the_longest_derived_id():
    at_cap = LOCAL_PREFIX + "a" * (MAX_AUTHORED_LOCAL_ID_BYTES - len(LOCAL_PREFIX))
    validate_local_id(derive_id(derive_id(at_cap, "files"), "0" * 16))
    with pytest.raises(ValueError, match="bytes"):
        validate_local_id(derive_id(derive_id(at_cap + "a", "files"), "0" * 16))
