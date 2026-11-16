# The ubuntu-24.04-browser base computer: Ubuntu 24.04 with no desktop (XDG_SESSION_TYPE=none), for one website
# browser and a page driver whose screen is the browser's page. The browser runs headful on an X server of its own,
# Xvfb on :99, which no agent sees, records or drives, so sites see a browser with a window rather than a headless one.
# It starts as root: root's supervisord (root.conf) runs the clock follower and the egress proxy, each as its own user,
# and the session's supervisord (supervisord.conf) as the session user, which runs every session program. The browser
# goes out only through the site proxy on 127.0.0.1:3128, which writes its network log to
# /var/log/agentenv/network.jsonl. Files a page's file chooser may be given go in ~/Uploads. Composing a driver or an app
# onto it adds its services as programs in /etc/supervisor/conf.d. The trust files agentenv adds are in /etc/agentenv
# from boot: ca.pem, the CAs agentenv adds (the egress proxy's from boot, a deploy's for its sites), and ca-bundle.pem,
# the public roots and those CAs.
#
# The universe clock (clock/, a copy of ubuntu-24.04's, which a unit test keeps identical): agent-env's libfaketime and
# its TLS shim, built in a stage of their own so no compiler is left in the image, are preloaded into every process and
# fake only the session user's, from the clock page the follower writes, which the session user can read but neither
# write nor replace.
#
# The egress proxy (egress/, a copy of ubuntu-24.04's, kept identical the same way), on real time on 127.0.0.1:3130:
# the site proxies hand it every connection to a public address, so a live site loads at any universe date. Built and
# tested in a stage of its own, so no Go toolchain is left in the image.
FROM public.ecr.aws/docker/library/golang:1.26.11-bookworm@sha256:dc40a13b61304647ed7e5d581d246b86413dfc29d63c6d018310feb93cd98fc0 AS egress
COPY egress/ /egress/
RUN cd /egress && go vet ./... && go test -count=1 ./... \
    && CGO_ENABLED=0 GOFLAGS=-mod=readonly go build -trimpath -ldflags=-s -o /out/egress .

FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c AS clock
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential patch curl ca-certificates
COPY clock/ /clock/
RUN sh /clock/build.sh /out

FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor tini python3 python3-venv fonts-dejavu-core tzdata ca-certificates libnss3-tools procps xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && userdel --remove ubuntu \
    && useradd --create-home --uid 1000 --shell /bin/bash user \
    && runuser -u user -- mkdir -p /home/user/Desktop /home/user/Documents /home/user/Downloads /home/user/Uploads \
    && useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-clock \
    && useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-egress \
    && mkdir -p /etc/supervisor/conf.d /var/log/agentenv /run/agentenv /var/log/agentenv-root /run/agentenv-root \
        /usr/local/lib/agentenv-clock /etc/agentenv/clock /usr/local/lib/agentenv-egress \
    && chmod 0700 /var/log/agentenv-root /run/agentenv-root \
    && chown root:agentenv-clock /etc/agentenv/clock && chmod 0750 /etc/agentenv/clock \
    && touch /etc/agentenv/ca.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /etc/agentenv/ca-bundle.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /usr/local/lib/agentenv-egress/roots.pem \
    && chown user:user /var/log/agentenv /run/agentenv
COPY --from=clock /out/libfaketimeMT.so.1 /out/libtls_realtime.so /usr/local/lib/agentenv-clock/
COPY clock/clockpage.py clock/clock-follower /usr/local/lib/agentenv-clock/
COPY clock/clock.conf /etc/agentenv/clock.conf
RUN sed -i -e "s/^AGENTENV_SESSION_UID=.*/AGENTENV_SESSION_UID=$(id -u user)/" \
        -e "s/^AGENTENV_CLOCK_PAGE_UID=.*/AGENTENV_CLOCK_PAGE_UID=$(id -u agentenv-clock)/" /etc/agentenv/clock.conf \
    && chmod 0755 /usr/local/lib/agentenv-clock/clock-follower \
    && printf '%s\n' /usr/local/lib/agentenv-clock/libfaketimeMT.so.1 /usr/local/lib/agentenv-clock/libtls_realtime.so \
        > /etc/ld.so.preload
COPY --from=egress /out/egress /usr/local/lib/agentenv-egress/egress
COPY root.conf /etc/supervisor/root.conf
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY start-computer /usr/local/bin/start-computer
COPY site-proxy.py /usr/local/bin/agentenv-site-proxy
USER root
WORKDIR /home/user
ENV LANG=C.UTF-8 XDG_SESSION_TYPE=none DISPLAY=:99 AGENTENV_SITE_PROXY=http://127.0.0.1:3128 \
    AGENTENV_NETWORK_LOG=/var/log/agentenv/network.jsonl AGENTENV_EGRESS_PROXY=http://127.0.0.1:3130
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start-computer", "init"]
