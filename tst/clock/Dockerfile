# The clock gates' image: Ubuntu 24.04 as the base computer has it, agent-env's libfaketime built by its build.sh and
# stock 0.9.12 from the same tarball beside it, the gate programs in C, Python, Java and Node, Chrome for Testing at
# the browsers bundle's pin, and OpenSSL 1.1.1w's libraries, a second copy of OpenSSL for the TLS gate.
# tst/clock/conftest.py assembles its context, with the downloads, checked.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ARG CHROME_SHA256
ARG CHROME_PACKAGES
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends build-essential patch curl openssl \
        libssl-dev ca-certificates python3 openjdk-21-jdk-headless nodejs util-linux unzip $CHROME_PACKAGES >/dev/null \
    && rm -rf /var/lib/apt/lists/* \
    && userdel -r ubuntu && useradd -m -u 1000 user
COPY downloads/chrome.zip /tmp/chrome.zip
RUN echo "$CHROME_SHA256  /tmp/chrome.zip" | sha256sum -c - \
    && unzip -q /tmp/chrome.zip -d /opt && mv /opt/chrome-linux* /opt/chrome && rm /tmp/chrome.zip
COPY clock /opt/clock-src
COPY downloads/libfaketime.tar.gz /tmp/lft.tgz
RUN /opt/clock-src/build.sh /opt/clock/lib /tmp/lft.tgz \
    && tar xzf /tmp/lft.tgz -C /tmp && make -s -C /tmp/libfaketime-0.9.12/src libfaketimeMT.so.1 \
    && mkdir -p /opt/clock/stock && cp /tmp/libfaketime-0.9.12/src/libfaketimeMT.so.1 /opt/clock/stock/ \
    && rm -rf /tmp/lft.tgz /tmp/libfaketime-0.9.12 \
    && cp /opt/clock-src/clockpage.py /opt/clock/ \
    && mkdir -p /etc/agentenv /run/agentenv-clock && cp /opt/clock-src/clock.conf /etc/agentenv/clock.conf
COPY downloads/openssl-1.1.1w.tar.gz /tmp/openssl.tgz
RUN tar xzf /tmp/openssl.tgz -C /tmp && cd /tmp/openssl-1.1.1w \
    && ./config shared no-tests >/dev/null && make -s -j"$(nproc)" build_libs >/dev/null \
    && mkdir -p /opt/openssl-1.1/lib /opt/openssl-1.1/include \
    && cp -P libcrypto.so libcrypto.so.1.1 /opt/openssl-1.1/lib/ && cp -r include/openssl /opt/openssl-1.1/include/ \
    && cd / && rm -rf /tmp/openssl.tgz /tmp/openssl-1.1.1w
COPY programs /opt/clock/programs
RUN mkdir -p /opt/clock/bin /opt/clock/java && cd /opt/clock/programs \
    && for p in uwait jumpwaits condstress2 semstress envrace rtread torn cancel tfdalias forklock; do \
        gcc -O2 -o /opt/clock/bin/$p $p.c -lpthread || exit 1; done \
    && mkdir -p /opt/clock/tls && gcc -O2 -fPIC -shared -o /opt/clock/tls/tlsverify3.so tlsverify.c -lcrypto \
    && gcc -O2 -fPIC -shared -I/opt/openssl-1.1/include -o /opt/clock/tls/tlsverify11.so tlsverify.c \
        -L/opt/openssl-1.1/lib -lcrypto -Wl,-rpath,/opt/openssl-1.1/lib \
    && gcc -O2 -o /opt/clock/bin/tlscopies tlscopies.c -ldl \
    && javac -d /opt/clock/java UWait.java && chmod 755 marker.sh
COPY harness.py /opt/clock/harness.py
