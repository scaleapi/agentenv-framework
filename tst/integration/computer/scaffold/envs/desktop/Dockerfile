# The scaffold: the reference driver and the Fixture Desk on the ubuntu-24.04 base computer, until agent-env composes
# computers. The test builds the base and names it in BASE, and stages driver.py, fixture.py and the protocol package
# (protocol/) beside this file.
ARG BASE
FROM ${BASE}
USER root
COPY protocol /opt/protocol
RUN python3 -m venv /opt/driver \
    && /opt/driver/bin/pip install --no-cache-dir "/opt/protocol[computer]" "mss>=9,<11" "python-xlib>=0.33,<0.34"
COPY driver.py fixture.py /opt/desktop/
COPY desktop.conf /etc/supervisor/conf.d/desktop.conf
USER user
