# The ubuntu-24.04 base computer: Ubuntu 24.04 with XFCE on Xvfb, D-Bus and AT-SPI, whose session user's home has a
# desktop's standard folders (Documents, Downloads, ...). It starts as root: root's supervisord (root.conf) runs the clock
# follower as its own user and the session's supervisord (supervisord.conf) as the session user, which runs every
# session program. Composing a driver or an app onto it adds its services as programs in /etc/supervisor/conf.d. The
# trust files agentenv adds are in /etc/agentenv from boot: ca.pem, empty until a deploy adds a CA, and ca-bundle.pem,
# the public roots and those CAs.
#
# The universe clock (clock/): agent-env's libfaketime and its TLS shim, built in a stage of their own so no compiler is
# left in the image, are preloaded into every process and fake only the session user's, from the clock page the
# follower writes, which the session user can read but neither write nor replace.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c AS clock
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential patch curl ca-certificates
COPY clock/ /clock/
RUN sh /clock/build.sh /out

FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11-utils x11-xserver-utils libxcb1 libxcb-xinerama0 \
        xfce4-session xfwm4 xfce4-panel xfdesktop4 xfce4-settings xfconf thunar \
        dbus dbus-x11 at-spi2-core xdg-user-dirs \
        supervisor tini \
        python3 python3-venv python3-tk \
        fonts-dejavu-core locales tzdata ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen \
    && userdel --remove ubuntu \
    && useradd --create-home --uid 1000 --shell /bin/bash user \
    && runuser -u user -- xdg-user-dirs-update \
    && useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin agentenv-clock \
    && mkdir -p /etc/supervisor/conf.d /var/log/agentenv /run/agentenv /var/log/agentenv-root /run/agentenv-root \
        /usr/local/lib/agentenv-clock /etc/agentenv/clock \
    && chmod 0700 /var/log/agentenv-root /run/agentenv-root \
    && chown root:agentenv-clock /etc/agentenv/clock && chmod 0750 /etc/agentenv/clock \
    && touch /etc/agentenv/ca.pem \
    && cp /etc/ssl/certs/ca-certificates.crt /etc/agentenv/ca-bundle.pem \
    && chown user:user /var/log/agentenv /run/agentenv
COPY --from=clock /out/libfaketimeMT.so.1 /out/libtls_realtime.so /usr/local/lib/agentenv-clock/
COPY clock/clockpage.py clock/clock-follower /usr/local/lib/agentenv-clock/
COPY clock/clock.conf /etc/agentenv/clock.conf
RUN sed -i -e "s/^AGENTENV_SESSION_UID=.*/AGENTENV_SESSION_UID=$(id -u user)/" \
        -e "s/^AGENTENV_CLOCK_PAGE_UID=.*/AGENTENV_CLOCK_PAGE_UID=$(id -u agentenv-clock)/" /etc/agentenv/clock.conf \
    && chmod 0755 /usr/local/lib/agentenv-clock/clock-follower \
    && printf '%s\n' /usr/local/lib/agentenv-clock/libfaketimeMT.so.1 /usr/local/lib/agentenv-clock/libtls_realtime.so \
        > /etc/ld.so.preload
COPY root.conf /etc/supervisor/root.conf
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY xfwm4.xml /etc/xdg/xfce4/xfconf/xfce-perchannel-xml/xfwm4.xml
COPY start-desktop /usr/local/bin/start-desktop
USER root
WORKDIR /home/user
ENV DISPLAY=:99 XDG_SESSION_TYPE=x11 LANG=en_US.UTF-8
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start-desktop", "init"]
