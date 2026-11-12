# The ubuntu-24.04-browser base computer: Ubuntu 24.04 with no desktop (XDG_SESSION_TYPE=none), for one website
# browser and a page driver whose screen is the browser's page. The browser runs headful on an X server of its own,
# Xvfb on :99, which no agent sees, records or drives, so sites see a browser with a window rather than a headless one.
# supervisord runs every program as the session user, and the browser goes out only through the site proxy on
# 127.0.0.1:3128.
# Files a page's file chooser may be given go in ~/Uploads. Composing a driver or an app onto it adds its services as
# programs in /etc/supervisor/conf.d. The trust files agentenv adds are in /etc/agentenv from boot: ca.pem, empty until
# a deploy adds a CA, and ca-bundle.pem, the public roots and those CAs.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor tini python3 python3-venv fonts-dejavu-core tzdata ca-certificates procps xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && userdel --remove ubuntu \
    && useradd --create-home --uid 1000 --shell /bin/bash user \
    && runuser -u user -- mkdir -p /home/user/Desktop /home/user/Documents /home/user/Downloads /home/user/Uploads \
    && mkdir -p /etc/supervisor/conf.d /var/log/agentenv /run/agentenv \
    && mkdir -p /etc/agentenv && touch /etc/agentenv/ca.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /etc/agentenv/ca-bundle.pem \
    && chown user:user /var/log/agentenv /run/agentenv
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY start-computer /usr/local/bin/start-computer
COPY site-proxy.py /usr/local/bin/agentenv-site-proxy
USER user
WORKDIR /home/user
ENV LANG=C.UTF-8 XDG_SESSION_TYPE=none DISPLAY=:99 AGENTENV_SITE_PROXY=http://127.0.0.1:3128
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start-computer"]
