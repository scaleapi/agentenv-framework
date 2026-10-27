# The scaffold: the reference driver on a hand-written Ubuntu 24.04 desktop, until agent-env composes computers. The
# test stages driver.py, fixture.py and the protocol package (protocol/) beside this file.
FROM public.ecr.aws/docker/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-tk xvfb x11-utils fonts-dejavu-core procps tini \
        libxcb1 libxcb-render0 libxcb-shm0 libxcb-randr0 libxcb-xfixes0 libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*
COPY protocol /opt/protocol
RUN python3 -m venv /opt/driver \
    && /opt/driver/bin/pip install --no-cache-dir "/opt/protocol[computer]" "mss>=9,<11" "python-xlib>=0.33,<0.34"
RUN useradd --create-home --uid 10001 user
COPY start.sh driver.py fixture.py /opt/desktop/
USER user
WORKDIR /home/user
ENV DISPLAY=:99 XDG_SESSION_TYPE=x11
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/sh", "/opt/desktop/start.sh"]
