"""The local transfer CA: created once per user, trusted only through its own files, and able to
vouch for local names and private addresses only. Handshakes run over memory BIOs, not sockets."""

import os
import ssl
import stat
from datetime import timedelta
from pathlib import Path

import certifi
import pytest
from cryptography import x509

from agent_env.config.paths import state_root
from agent_env.store.object_store.local_grants import tls
from agent_env.store.object_store.local_grants.tls import check_local_host, local_ca, server_context


def _handshake(server: ssl.SSLContext, *, cafile: Path, hostname: str) -> None:
    client = ssl.create_default_context(cafile=str(cafile))
    client.verify_flags |= ssl.VERIFY_X509_STRICT  # the default from Python 3.13
    to_server, to_client = ssl.MemoryBIO(), ssl.MemoryBIO()
    sides = [
        client.wrap_bio(to_client, to_server, server_hostname=hostname),
        server.wrap_bio(to_server, to_client, server_side=True),
    ]
    done = set()
    for _ in range(10):
        for side in sides:
            if id(side) not in done:
                try:
                    side.do_handshake()
                    done.add(id(side))
                except ssl.SSLWantReadError:
                    pass
        if len(done) == 2:
            return
    raise AssertionError("the handshake did not finish")


@pytest.mark.parametrize("hostname", ["host.docker.internal", "localhost", "127.0.0.1"])
def test_a_server_certificate_verifies_for_the_names_it_was_made_for(hostname):
    ca = local_ca()
    context = server_context(ca, ["localhost", "host.docker.internal", "127.0.0.1"])
    _handshake(context, cafile=ca.cert_path, hostname=hostname)
    _handshake(context, cafile=ca.bundle_path, hostname=hostname)


def test_the_ca_cannot_vouch_for_a_public_name():
    ca = local_ca()
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(server_context(ca, ["example.com"]), cafile=ca.cert_path, hostname="example.com")


def test_the_ca_cannot_vouch_for_a_public_address():
    ca = local_ca()
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(server_context(ca, ["8.8.8.8"]), cafile=ca.cert_path, hostname="8.8.8.8")


def test_the_bundle_is_the_public_roots_plus_the_ca():
    ca = local_ca()
    bundle = ca.bundle_path.read_bytes()
    assert bundle.startswith(Path(certifi.where()).read_bytes().rstrip())
    assert bundle.endswith(ca.cert_path.read_bytes())
    assert x509.load_pem_x509_certificate(ca.cert_path.read_bytes()) == ca.cert


def test_the_ca_key_is_private_to_the_user():
    local_ca()
    tls_dir = state_root() / "tls"
    assert stat.S_IMODE(os.stat(tls_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(tls_dir / "ca.key.pem").st_mode) == 0o600


def test_the_ca_is_created_once_and_reused():
    assert local_ca().cert == local_ca().cert


def test_a_ca_near_expiry_is_replaced(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(tls, "_CA_LIFETIME", timedelta(days=10))
        first = local_ca()
    second = local_ca()
    assert second.cert != first.cert
    assert x509.load_pem_x509_certificate(second.cert_path.read_bytes()) == second.cert
    assert local_ca().cert == second.cert


def test_an_unreadable_ca_is_replaced():
    first = local_ca()
    (state_root() / "tls" / "ca.key.pem").write_text("not a key")
    assert local_ca().cert != first.cert


@pytest.mark.parametrize("host", [
    "localhost", "host.docker.internal", "a.localhost", "127.0.0.1", "172.17.0.1", "192.168.5.2", "10.0.0.8",
    "::1", "fd00::1",
])
def test_local_hosts_are_accepted(host):
    check_local_host(host)


@pytest.mark.parametrize("host", ["example.com", "evil-localhost", "docker.internal", "8.8.8.8", "2001:db8::1"])
def test_other_hosts_are_refused(host):
    with pytest.raises(ValueError):
        check_local_host(host)
