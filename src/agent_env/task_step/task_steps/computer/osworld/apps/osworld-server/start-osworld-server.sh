#!/usr/bin/env bash
# Runs the OSWorld guest server on 127.0.0.1:5000 with the app's venv; the computer's supervisord restarts it when it
# exits (its app services are autorestart). The OS's dist-packages go after the venv's, for AT-SPI's bindings, which
# pip can't install.
cd "$(dirname "$0")/payload"
export DISPLAY="${DISPLAY:-:0}"
export PYTHONUNBUFFERED=1
# python-xlib refuses to connect without an Xauthority file; the computer's X server takes connections without one.
[ -e "${XAUTHORITY:-$HOME/.Xauthority}" ] || touch "${XAUTHORITY:-$HOME/.Xauthority}"
exec ../venv/bin/python -c 'import runpy, sys; sys.path.append("/usr/lib/python3/dist-packages"); sys.argv = ["main.py"]; runpy.run_path("main.py", run_name="__main__")'
