/* A stand-in for a libssl: a library linked against one copy of OpenSSL that verifies certificates through it. Built
   twice, against the system's OpenSSL 3.0 and against a 1.1 built beside it, and loaded by tlscopies. */
#include <openssl/crypto.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509_vfy.h>
#include <stdio.h>

/* The certificate at path verified with itself as the one trust anchor: X509_verify_cert's result, and its error. */
int tls_verify(const char *path, int *error)
{
    FILE *f = fopen(path, "r");
    X509 *cert = f ? PEM_read_X509(f, NULL, NULL, NULL) : NULL;
    X509_STORE *store = X509_STORE_new();
    X509_STORE_CTX *ctx = X509_STORE_CTX_new();
    int ok = -2;
    if (f) fclose(f);
    if (cert && store && ctx && X509_STORE_add_cert(store, cert) && X509_STORE_CTX_init(ctx, store, cert, NULL)) {
        ok = X509_verify_cert(ctx);
        *error = X509_STORE_CTX_get_error(ctx);
    }
    X509_STORE_CTX_free(ctx);
    X509_STORE_free(store);
    X509_free(cert);
    return ok;
}

const char *tls_version(void) { return OpenSSL_version(OPENSSL_VERSION); }
