/* SPDX-License-Identifier: Apache-2.0 */
/*
 * agent-env's TLS shim: OpenSSL checks a peer's certificates against real time, whatever the universe's clock says.
 *
 * Preloaded after the clock library (/etc/ld.so.preload). A program on universe time months from real time would find
 * every live site's certificate not yet valid or expired. Most clients reach live sites through the computer's egress
 * proxy, which hands them certificates valid at the universe date; this is for the ones that bypass it. OpenSSL's
 * libssl verifies a chain with X509_verify_cert, which this defines first in the global scope: it sets the
 * verification's check time to real time, read with a raw system call the clock library doesn't intercept, unless the
 * caller set one, then calls OpenSSL's own, the one in the caller's own libcrypto, so a process holding two copies of
 * OpenSSL (a bundled one beside the system's) verifies each with its own.
 *
 * Not covered: TLS stacks of their own (Chromium's BoringSSL, GnuTLS, NSS, Java's) and static binaries.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

typedef struct x509_store_ctx_st X509_STORE_CTX;
typedef struct X509_VERIFY_PARAM_st X509_VERIFY_PARAM;

/* OpenSSL's X509_V_FLAG_USE_CHECK_TIME (x509_vfy.h), the same in 1.1 and 3 */
#define TLS_USE_CHECK_TIME 0x2
#define TLS_SYSTEM_LIBCRYPTO "libcrypto.so.3"

/* OpenSSL's functions as one copy of it has them */
struct tls_openssl
{
  int (*verify)(X509_STORE_CTX *);
  void (*set_time)(X509_STORE_CTX *, unsigned long, time_t);
  X509_VERIFY_PARAM *(*get0_param)(X509_STORE_CTX *);
  unsigned long (*get_flags)(const X509_VERIFY_PARAM *);
};

int X509_verify_cert(X509_STORE_CTX *ctx);

/* the real time in seconds, read with the system call itself (the clock library intercepts libc's), 0 if it fails */
static long tls_real_time(void)
{
  struct timespec ts = {0, 0};
#if defined(__x86_64__)
  long ret;
  __asm__ volatile("syscall" : "=a"(ret) : "0"(228L), "D"((long) CLOCK_REALTIME), "S"(&ts) : "rcx", "r11", "memory");
  if (ret != 0) return 0;
#elif defined(__aarch64__)
  register long x8 __asm__("x8") = 113;
  register long x0 __asm__("x0") = CLOCK_REALTIME;
  register long x1 __asm__("x1") = (long) &ts;
  __asm__ volatile("svc 0" : "+r"(x0) : "r"(x8), "r"(x1) : "memory");
  if (x0 != 0) return 0;
#else
#error the TLS shim reads real time with a raw system call on x86_64 or aarch64
#endif
  return ts.tv_sec;
}

/* ``name`` in ``scope`` (a handle, or RTLD_NEXT), never this file's own definition */
static void *tls_lookup(void *scope, const char *name)
{
  void *found = dlsym(scope, name);
  return found == (void *) X509_verify_cert ? NULL : found;
}

static void tls_take(void *scope, struct tls_openssl *out)
{
  out->verify = (int (*)(X509_STORE_CTX *)) tls_lookup(scope, "X509_verify_cert");
  out->set_time = (void (*)(X509_STORE_CTX *, unsigned long, time_t)) tls_lookup(scope, "X509_STORE_CTX_set_time");
  out->get0_param = (X509_VERIFY_PARAM * (*)(X509_STORE_CTX *)) tls_lookup(scope, "X509_STORE_CTX_get0_param");
  out->get_flags = (unsigned long (*)(const X509_VERIFY_PARAM *)) tls_lookup(scope, "X509_VERIFY_PARAM_get_flags");
}

/* OpenSSL as the object at ``caller`` sees it: from that object's own dependencies (libssl's libcrypto, whichever copy
   it is); for a caller that isn't a shared library depending on one (a program calling it itself), the next definition
   in the global scope; else the system's libcrypto where it was loaded privately (a Python extension's). Looked up on
   every call, once a handshake: a copy of OpenSSL unloaded and another loaded at its address must not be handed the
   first one's functions. */
static void tls_resolve(void *caller, struct tls_openssl *out)
{
  Dl_info info;
  void *handle;
  memset(out, 0, sizeof *out);
  if (dladdr(caller, &info))
  {
    if (info.dli_fname != NULL && info.dli_fname[0] != 0
        && (handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD)) != NULL)
    {
      tls_take(handle, out);
      dlclose(handle);
      if (out->verify != NULL) return;
    }
  }
  tls_take(RTLD_NEXT, out);
  if (out->verify != NULL) return;
  if ((handle = dlopen(TLS_SYSTEM_LIBCRYPTO, RTLD_NOW | RTLD_NOLOAD)) != NULL)
  {
    tls_take(handle, out);
    dlclose(handle);
  }
}

int X509_verify_cert(X509_STORE_CTX *ctx)
{
  struct tls_openssl ssl;
  tls_resolve(__builtin_return_address(0), &ssl);
  if (ssl.verify == NULL) return -1;
  if (ctx != NULL && ssl.set_time != NULL && ssl.get0_param != NULL && ssl.get_flags != NULL)
  {
    X509_VERIFY_PARAM *param = ssl.get0_param(ctx);
    long now = tls_real_time();
    if (param != NULL && !(ssl.get_flags(param) & TLS_USE_CHECK_TIME) && now > 0) ssl.set_time(ctx, 0, (time_t) now);
  }
  return ssl.verify(ctx);
}
