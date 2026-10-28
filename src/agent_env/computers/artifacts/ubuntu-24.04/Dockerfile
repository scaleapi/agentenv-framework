# The ubuntu-24.04 base computer: Ubuntu 24.04 with XFCE on Xvfb, D-Bus and AT-SPI, every program run by supervisord
# as the session user. Composing a driver or an app onto it adds its services as programs in /etc/supervisor/conf.d.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11-utils x11-xserver-utils libxcb1 libxcb-xinerama0 \
        xfce4-session xfwm4 xfce4-panel xfdesktop4 xfce4-settings xfconf thunar \
        dbus dbus-x11 at-spi2-core \
        supervisor tini \
        python3 python3-venv python3-tk \
        fonts-dejavu-core locales tzdata ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen \
    && userdel --remove ubuntu \
    && useradd --create-home --uid 1000 --shell /bin/bash user \
    && mkdir -p /etc/supervisor/conf.d /var/log/agentenv /run/agentenv \
    && chown user:user /var/log/agentenv /run/agentenv
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY xfwm4.xml /etc/xdg/xfce4/xfconf/xfce-perchannel-xml/xfwm4.xml
COPY start-desktop /usr/local/bin/start-desktop
USER user
WORKDIR /home/user
ENV DISPLAY=:99 XDG_SESSION_TYPE=x11 LANG=en_US.UTF-8
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start-desktop", "init"]
