/* fork while another thread works the library's tables: one thread duplicates and closes a timerfd in a loop while the
   main thread forks FORKS times, and each child closes its descriptors with close_range, as a child does before exec,
   and exits. A child that hasn't exited within 2 s is stuck (on a lock the forking moment caught held). Prints
   {"forks", "stuck"}. */
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/timerfd.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int tfd;
static void *churn(void *a) { (void)a; for (;;) { int d = dup(tfd); close(d); } return NULL; }

int main(int argc, char **argv) {
    int forks = argc > 1 ? atoi(argv[1]) : 1000, stuck = 0;
    pthread_t t;
    tfd = timerfd_create(CLOCK_REALTIME, 0);
    pthread_create(&t, NULL, churn, NULL);
    for (int i = 0; i < forks; i++) {
        pid_t pid = fork();
        if (pid == 0) { close_range(3, ~0U, 0); _exit(0); }
        int done = 0;
        for (int k = 0; k < 200 && !done; k++) { if (waitpid(pid, NULL, WNOHANG) == pid) done = 1; else usleep(10000); }
        if (!done) { stuck++; kill(pid, SIGKILL); waitpid(pid, NULL, 0); }
    }
    printf("{\"forks\": %d, \"stuck\": %d}\n", forks, stuck);
    return 0;
}
