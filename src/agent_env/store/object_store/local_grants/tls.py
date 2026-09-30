"""The local certificate authority behind the grant server's HTTPS.

One CA per user, created on first use under ``<state root>/tls/``. It is name-constrained to
local names and private addresses, so even a leaked key can only vouch for those, and it is
never installed in the host's trust stores: only the containers agent-env starts trust it,
through ``ca-bundle.pem`` (the public roots plus this CA, for ``SSL_CERT_FILE``, which replaces
the defaults) or ``ca.pem`` (the CA alone, for trust stores that add to the defaults).
"""

from __future__ import annotations

import ipaddress
import os
import secrets
import ssl
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import certifi
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from agent_env.config.paths import state_root
from agent_env.store.local_state import ensure_state_dir

# What the CA may vouch for: the names containers reach the host by, and loopback and private addresses.
LOCAL_NAMES = ("localhost", "host.docker.internal")
LOCAL_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "::1/128", "fc00::/7")
)

_CA_LIFETIME = timedelta(days=5 * 365)
_CA_RENEW_BEFORE = timedelta(days=30)  # replaced this long before it expires
_SERVER_LIFETIME = timedelta(days=90)
_CLOCK_SKEW = timedelta(minutes=5)

_CA_FILE = "ca.key.pem"  # the CA's key and certificate, private to the user
_CERT_FILE = "ca.pem"
_BUNDLE_FILE = "ca-bundle.pem"
_PEM_CERT = b"-----BEGIN CERTIFICATE-----"


@dataclass(frozen=True)
class LocalCA:
    """The CA's key and certificate, and the trust files derived from it."""

    key: ec.EllipticCurvePrivateKey
    cert: x509.Certificate
    cert_path: Path
    bundle_path: Path


def check_local_host(host: str) -> None:
    """Refuse a host the CA cannot vouch for: a name other than the local ones, or a public address."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        name = host.lower().rstrip(".")
        if any(name == n or name.endswith("." + n) for n in LOCAL_NAMES):
            return
        raise ValueError(
            f"{host!r} is not a local name; the local transfer CA vouches only for {', '.join(LOCAL_NAMES)} "
            "and loopback or private addresses"
        ) from None
    if not any(address in network for network in LOCAL_NETWORKS):
        raise ValueError(f"{host!r} is not a loopback or private address, which the local transfer CA vouches for")


def local_ca() -> LocalCA:
    """This user's local CA, created on first use and replaced when it nears expiry."""
    tls_dir = state_root() / "tls"
    ensure_state_dir(tls_dir)
    ca_file = tls_dir / _CA_FILE
    loaded = _load(ca_file)
    if loaded is None:
        key, cert = _new_ca()
        pem = _key_pem(key) + cert.public_bytes(serialization.Encoding.PEM)
        # A first CA is claimed with a hard link, so two processes starting at once agree on one.
        replacing = ca_file.exists()
        staged = _write_private(tls_dir, pem)
        try:
            if replacing:
                os.replace(staged, ca_file)
            else:
                try:
                    os.link(staged, ca_file)
                except FileExistsError:
                    pass
        finally:
            staged.unlink(missing_ok=True)
        loaded = _load(ca_file) or (key, cert)
    key, cert = loaded
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    cert_path = _write_if_changed(tls_dir / _CERT_FILE, cert_pem)
    bundle_path = _write_if_changed(tls_dir / _BUNDLE_FILE, Path(certifi.where()).read_bytes().rstrip() + b"\n" + cert_pem)
    return LocalCA(key=key, cert=cert, cert_path=cert_path, bundle_path=bundle_path)


def server_context(ca: LocalCA, hosts: Iterable[str]) -> ssl.SSLContext:
    """A TLS server context presenting a new certificate, signed by ``ca``, for each of ``hosts``."""
    key = ec.generate_private_key(ec.SECP256R1())
    names = list(dict.fromkeys(hosts))
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _CLOCK_SKEW)
        .not_valid_after(min(now + _SERVER_LIFETIME, ca.cert.not_valid_after_utc))
        .add_extension(x509.SubjectAlternativeName([_general_name(n) for n in names]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_key_usage(digital_signature=True), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.key.public_key()), critical=False
        )
        .sign(ca.key, hashes.SHA256())
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    # ssl loads a certificate chain only from a file; this one lives just long enough to be read.
    with tempfile.TemporaryDirectory() as tmp:
        chain = Path(tmp) / "server.pem"
        chain.write_bytes(_key_pem(key) + cert.public_bytes(serialization.Encoding.PEM))
        context.load_cert_chain(chain)
    return context


def _new_ca() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"agent-env local transfer CA {secrets.token_hex(4)}")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _CLOCK_SKEW)
        .not_valid_after(now + _CA_LIFETIME)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_key_usage(key_cert_sign=True, crl_sign=True), critical=True)
        .add_extension(_name_constraints(), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _load(ca_file: Path) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate] | None:
    """The CA in ``ca_file``, or None when it is missing, unreadable, nearly expired, or not one this
    release would create."""
    try:
        data = ca_file.read_bytes()
        key_pem, marker, cert_pem = data.partition(_PEM_CERT)
        key = serialization.load_pem_private_key(key_pem, password=None)
        cert = x509.load_pem_x509_certificate(marker + cert_pem)
        constraints = cert.extensions.get_extension_for_class(x509.NameConstraints).value
    except (OSError, ValueError, TypeError, x509.ExtensionNotFound):
        return None
    if not isinstance(key, ec.EllipticCurvePrivateKey) or key.public_key() != cert.public_key():
        return None
    if constraints != _name_constraints() or cert.not_valid_after_utc - datetime.now(UTC) < _CA_RENEW_BEFORE:
        return None
    return key, cert


def _name_constraints() -> x509.NameConstraints:
    return x509.NameConstraints(
        permitted_subtrees=[x509.DNSName(n) for n in LOCAL_NAMES] + [x509.IPAddress(n) for n in LOCAL_NETWORKS],
        excluded_subtrees=None,
    )


def _key_usage(*, digital_signature: bool = False, key_cert_sign: bool = False, crl_sign: bool = False) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=digital_signature,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=key_cert_sign,
        crl_sign=crl_sign,
        encipher_only=False,
        decipher_only=False,
    )


def _general_name(host: str) -> x509.GeneralName:
    try:
        return x509.IPAddress(ipaddress.ip_address(host))
    except ValueError:
        return x509.DNSName(host)


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


def _write_private(directory: Path, data: bytes) -> Path:
    fd, name = tempfile.mkstemp(dir=directory, prefix=".ca.")  # created 0600
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return Path(name)


def _write_if_changed(path: Path, data: bytes) -> Path:
    try:
        if path.read_bytes() == data:
            return path
    except OSError:
        pass
    staged = _write_private(path.parent, data)
    try:
        staged.chmod(0o644)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return path
