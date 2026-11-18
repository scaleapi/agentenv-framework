"""The TLS shim: OpenSSL clients on universe time check certificates against real time, through no proxy."""
import pytest

CLIENTS = {"curl", "python", "s_client"}
#: X509_V_ERR_CERT_NOT_YET_VALID and X509_V_ERR_CERT_HAS_EXPIRED, the same in OpenSSL 1.1 and 3.
NOT_YET_VALID, EXPIRED = 9, 10


@pytest.fixture(scope="module")
def tls(gate):
    return gate("tls")


def _all(results: dict, answer: str) -> dict:
    """What isn't ``answer`` among ``results`` ({site: {client: answer}})."""
    return {(site, client): got for site, by_client in results.items() for client, got in by_client.items()
            if got != answer}


def test_with_the_shim_certificates_are_checked_against_real_time_and_bad_ones_still_refused(tls):
    alone, shimmed = tls["without_shim"]["local"], tls["with_shim"]["local"]
    assert all(set(by_client) == CLIENTS for part in (alone, shimmed) for by_client in part.values()), tls
    # The clock library alone checks against the universe's date: a certificate valid only from real now isn't valid
    # yet, and one that expired since the universe's date still is.
    assert (_all({"now": alone["now"]}, "refused"), _all({"past": alone["past"]}, "ok")) == ({}, {}), alone
    # With the shim, against real time: the other way round, the expired one refused.
    assert (_all({"now": shimmed["now"]}, "ok"), _all({"past": shimmed["past"]}, "refused")) == ({}, {}), shimmed
    assert len(tls["with_shim"]["live"]) == 6 and _all(tls["with_shim"]["live"], "ok") == {}, tls["with_shim"]["live"]
    # Valid at real time but issued by no one the system's roots trust: still refused.
    assert _all({"untrusted": tls["with_shim"]["untrusted"]}, "refused") == {}, tls["with_shim"]["untrusted"]
    assert set(tls["with_shim"]["untrusted"]) == CLIENTS, tls["with_shim"]["untrusted"]


def test_two_copies_of_openssl_in_one_process_each_check_with_their_own(tls):
    # The system's OpenSSL 3.0 and a 1.1 beside it, each reached through a library loaded privately, with the system's
    # also in the global scope or not: a 1.1 check handed to 3.0's libcrypto, or the other way, can't give these.
    answers = {"without_shim": {"now": [0, NOT_YET_VALID], "past": [1, 0]},
               "with_shim": {"now": [1, 0], "past": [0, EXPIRED]}}
    for shim, expected in answers.items():
        copies = tls[shim]["copies"]
        got = {mode: {copy: {cert: answer.get(cert) for cert in expected} for copy, answer in by_copy.items()}
               for mode, by_copy in copies.items()}
        assert got == {mode: {"3.0": expected, "1.1": expected} for mode in ("global", "private")}, (shim, copies)
        assert {copy: answer["version"].split()[1][:3] for by_copy in copies.values()
                for copy, answer in by_copy.items()} == {"3.0": "3.0", "1.1": "1.1"}, copies
