# Built from packages/agentenv-protocol, so the driver runs on this checkout's SDK:
#   docker build -f examples/computer_x11/Dockerfile -t x11-desktop .
FROM public.ecr.aws/docker/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-tk xvfb x11-utils fonts-dejavu-core procps \
        libxcb1 libxcb-render0 libxcb-shm0 libxcb-randr0 libxcb-xfixes0 libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/protocol/
COPY src /opt/protocol/src
RUN python3 -m venv /opt/driver \
    && /opt/driver/bin/pip install --no-cache-dir "/opt/protocol[computer]" "mss>=9,<11" "python-xlib>=0.33,<0.34"
RUN useradd --create-home --uid 10001 user
COPY examples/computer_x11/start.sh examples/computer_x11/driver.py examples/computer_x11/fixture.py /opt/desktop/
USER user
WORKDIR /home/user
ENV DISPLAY=:99 XDG_SESSION_TYPE=x11
EXPOSE 18765
CMD ["/bin/sh", "/opt/desktop/start.sh"]
