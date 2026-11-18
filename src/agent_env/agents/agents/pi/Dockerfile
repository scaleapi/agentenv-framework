# docker build -t pi-a2a src/agent_env/agents/agents/pi
# The env's clock (FD-5916): the computer base's pieces, copied in (clock/, egress/) until the clock library image
# (FD-5700) ships them for each libc. The egress proxy, one static Go binary, and agent-env's libfaketime and its TLS
# shim, built for this image's libc (Ubuntu 24.04's, on the image's own base) in stages of their own so no toolchain
# is left in the image.
FROM public.ecr.aws/docker/library/golang:1.26.11-bookworm@sha256:dc40a13b61304647ed7e5d581d246b86413dfc29d63c6d018310feb93cd98fc0 AS egress
COPY egress/ /egress/
RUN cd /egress && go vet ./... && go test -count=1 ./... \
    && CGO_ENABLED=0 GOFLAGS=-mod=readonly go build -trimpath -ldflags=-s -o /out/egress .

FROM mirror.gcr.io/library/ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55 AS clock
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential patch curl ca-certificates
COPY clock/ /clock/
RUN sh /clock/build.sh /out

FROM mirror.gcr.io/library/ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
WORKDIR /app
# Named sources: agent-env packs the build context install/v1 copies from the COPY sources.
COPY a2a_server.py changelog.py install.sh peers.py pi_agent.py requirements.txt start.sh ./
RUN sh install.sh \
    && useradd --create-home agent \
    && mkdir /workspace \
    && chown agent /app /workspace
# The env's clock: the follower and the egress proxy, each as its own user, under root's supervisord (root.conf), and
# agent-env's libfaketime and its TLS shim preloaded into every process, faking only the agent's (clock.conf). The
# follower's config is the agent's user's to write (agentenv_protocol.a2a_agent.clock) and the follower's to read.
# supervisord goes into the agent's venv, whose bin is on the PATH start-agent runs with.
RUN /opt/pi-a2a-venv/bin/pip install --no-cache-dir supervisor==4.2.5 \
    && useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-clock \
    && useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-egress \
    && mkdir -p /usr/local/lib/agentenv-clock /usr/local/lib/agentenv-egress /etc/agentenv \
    && install -d -o agent -g agentenv-clock -m 2750 /etc/agentenv/clock \
    && touch /etc/agentenv/ca.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /etc/agentenv/ca-bundle.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /usr/local/lib/agentenv-egress/roots.pem
COPY --from=clock /out/libfaketimeMT.so.1 /out/libtls_realtime.so /usr/local/lib/agentenv-clock/
COPY --from=egress /out/egress /usr/local/lib/agentenv-egress/egress
COPY clock/clockpage.py clock/clock-follower /usr/local/lib/agentenv-clock/
COPY clock/clock.conf /etc/agentenv/clock.conf
COPY root.conf /etc/agentenv/root.conf
COPY start-agent /usr/local/bin/start-agent
RUN sed -i -e "s/^AGENTENV_SESSION_UID=.*/AGENTENV_SESSION_UID=$(id -u agent)/" \
        -e "s/^AGENTENV_CLOCK_PAGE_UID=.*/AGENTENV_CLOCK_PAGE_UID=$(id -u agentenv-clock)/" /etc/agentenv/clock.conf \
    && chmod 0755 /usr/local/lib/agentenv-clock/clock-follower /usr/local/bin/start-agent \
    && printf '%s\n' /usr/local/lib/agentenv-clock/libfaketimeMT.so.1 /usr/local/lib/agentenv-clock/libtls_realtime.so \
        > /etc/ld.so.preload
ENV PATH="/opt/pi-a2a-venv/bin:/opt/pi/bin:/opt/node/bin:${PATH}"
EXPOSE 8000
# Starts as root, which start-agent needs for the clock; the A2A server and pi run as the agent's user. CMD rather
# than ENTRYPOINT: Modal passes a sandbox's env to the image's CMD only.
CMD ["/usr/local/bin/start-agent"]
