"""The TLS shim: OpenSSL clients on universe time check certificates against real time, through no proxy."""

CLIENTS = {"curl", "python", "s_client"}


def _all(results: dict, answer: str) -> dict:
    """What isn't ``answer`` among ``results`` ({site: {client: answer}})."""
    return {(site, client): got for site, by_client in results.items() for client, got in by_client.items()
            if got != answer}


def test_with_the_shim_certificates_are_checked_against_real_time_and_bad_ones_still_refused(gate):
    r = gate("tls")
    alone, shimmed = r["without_shim"]["local"], r["with_shim"]["local"]
    assert all(set(by_client) == CLIENTS for part in (alone, shimmed) for by_client in part.values()), r
    # The clock library alone checks against the universe's date: a certificate valid only from real now isn't valid
    # yet, and one that expired since the universe's date still is.
    assert (_all({"now": alone["now"]}, "refused"), _all({"past": alone["past"]}, "ok")) == ({}, {}), alone
    # With the shim, against real time: the other way round.
    assert (_all({"now": shimmed["now"]}, "ok"), _all({"past": shimmed["past"]}, "refused")) == ({}, {}), shimmed
    assert len(r["with_shim"]["live"]) == 6 and _all(r["with_shim"]["live"], "ok") == {}, r["with_shim"]["live"]
    assert len(r["with_shim"]["bad"]) == 2 and _all(r["with_shim"]["bad"], "refused") == {}, r["with_shim"]["bad"]
