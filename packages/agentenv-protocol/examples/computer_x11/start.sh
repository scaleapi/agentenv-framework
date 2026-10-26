#!/bin/sh
# The desktop: Xvfb and the fixture each restart if they die, then the driver serves on MCP_PORT.
(while true; do Xvfb :99 -screen 0 1280x800x24 -nolisten tcp -ac >/tmp/xvfb.log 2>&1; sleep 0.5; done) &
for i in $(seq 100); do xdpyinfo -display :99 >/dev/null 2>&1 && break; sleep 0.1; done
(while true; do
   for i in $(seq 100); do xdpyinfo -display :99 >/dev/null 2>&1 && break; sleep 0.1; done
   python3 /opt/desktop/fixture.py >/tmp/fixture.log 2>&1; sleep 0.5
 done) &
exec /opt/driver/bin/python /opt/desktop/driver.py
