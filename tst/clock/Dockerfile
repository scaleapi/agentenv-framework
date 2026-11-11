# The clock gates' image: Ubuntu 24.04 as the base computer has it, agent-env's libfaketime built by its build.sh and
# stock 0.9.12 from the same tarball beside it, the gate programs in C, Python, Java and Node, and Chrome for Testing
# at the browsers bundle's pin. tst/clock/conftest.py assembles its context, with both downloads, checked.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ARG CHROME_SHA256
ARG CHROME_PACKAGES
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends build-essential patch curl \
        ca-certificates python3 openjdk-21-jdk-headless nodejs util-linux unzip $CHROME_PACKAGES >/dev/null \
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
COPY programs /opt/clock/programs
RUN mkdir -p /opt/clock/bin /opt/clock/java && cd /opt/clock/programs \
    && for p in uwait jumpwaits condstress2 semstress envrace rtread torn cancel tfdalias; do \
        gcc -O2 -o /opt/clock/bin/$p $p.c -lpthread || exit 1; done \
    && javac -d /opt/clock/java UWait.java && chmod 755 marker.sh
COPY harness.py /opt/clock/harness.py
