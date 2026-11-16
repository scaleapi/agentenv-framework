/* Two copies of OpenSSL in one process, as a program with a bundled one beside the system's has them: the system's 3.0
   and a 1.1 built beside it, each reached through tlsverify.c built against it and loaded privately (RTLD_LOCAL), as
   Python loads an extension. With MODE global the system's libcrypto is loaded into the global scope first, as a
   program linked against it has it. Each CERT is verified through both copies in turn, so their calls alternate.
   tlscopies MODE CERT... prints {"<copy>": {"version": ..., "<cert>": [result, error], ...}, ...}. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#define COPIES 2
#define CERTS 8

int main(int argc, char **argv)
{
    const char *names[COPIES] = {"3.0", "1.1"};
    const char *paths[COPIES] = {"/opt/clock/tls/tlsverify3.so", "/opt/clock/tls/tlsverify11.so"};
    int (*verify[COPIES])(const char *, int *);
    const char *(*version[COPIES])(void);
    int ok[COPIES][CERTS], error[COPIES][CERTS], certs = argc - 2;
    if (argc < 3 || certs > CERTS) return 2;
    if (!strcmp(argv[1], "global") && !dlopen("libcrypto.so.3", RTLD_NOW | RTLD_GLOBAL)) return 2;
    for (int i = 0; i < COPIES; i++) {
        void *h = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "%s\n", dlerror()); return 2; }
        verify[i] = (int (*)(const char *, int *)) dlsym(h, "tls_verify");
        version[i] = (const char *(*)(void)) dlsym(h, "tls_version");
    }
    for (int c = 0; c < certs; c++)
        for (int i = 0; i < COPIES; i++) {
            error[i][c] = -1;
            ok[i][c] = verify[i](argv[c + 2], &error[i][c]);
        }
    printf("{");
    for (int i = 0; i < COPIES; i++) {
        printf("%s\"%s\": {\"version\": \"%s\"", i ? ", " : "", names[i], version[i]());
        for (int c = 0; c < certs; c++) printf(", \"%s\": [%d, %d]", argv[c + 2], ok[i][c], error[i][c]);
        printf("}");
    }
    printf("}\n");
    return 0;
}
