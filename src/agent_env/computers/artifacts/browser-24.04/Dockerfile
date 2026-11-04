# The browser-24.04 base computer: Ubuntu 24.04 with no display server or desktop (XDG_SESSION_TYPE=none), for one
# website browser, which runs headless there, and a page driver whose screen is the browser's page. supervisord runs
# every program as the session user, and the browser goes out only through the site proxy on 127.0.0.1:3128. Composing
# a driver or an app onto it adds its services as programs in /etc/supervisor/conf.d.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor tini python3 python3-venv fonts-dejavu-core tzdata ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && userdel --remove ubuntu \
    && useradd --create-home --uid 1000 --shell /bin/bash user \
    && runuser -u user -- mkdir -p /home/user/Desktop /home/user/Documents /home/user/Downloads \
    && mkdir -p /etc/supervisor/conf.d /var/log/agentenv /run/agentenv \
    && chown user:user /var/log/agentenv /run/agentenv
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY start-computer /usr/local/bin/start-computer
COPY site-proxy.py /usr/local/bin/agentenv-site-proxy
USER user
WORKDIR /home/user
ENV LANG=C.UTF-8 XDG_SESSION_TYPE=none AGENTENV_SITE_PROXY=http://127.0.0.1:3128
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start-computer"]
