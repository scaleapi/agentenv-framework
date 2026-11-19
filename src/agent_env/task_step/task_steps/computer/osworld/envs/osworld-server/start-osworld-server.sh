#!/usr/bin/env bash
# Runs upstream OSWorld's guest server (payload/main.py, downloaded unmodified) on 127.0.0.1:5000 with the app's venv;
# the computer's supervisord restarts it when it exits (its app services are autorestart). main.py is loaded under a
# name of its own rather than run as __main__, whose `app.run(debug=True, host="0.0.0.0")` would serve Werkzeug's
# debugger on every interface; the app's own server is what forwards to it. The OS's dist-packages go after the
# venv's, for AT-SPI's bindings, which pip can't install.
cd "$(dirname "$0")/payload"
export DISPLAY="${DISPLAY:-:0}"
export PYTHONUNBUFFERED=1
# python-xlib refuses to connect without an Xauthority file; the computer's X server takes connections without one.
[ -e "${XAUTHORITY:-$HOME/.Xauthority}" ] || touch "${XAUTHORITY:-$HOME/.Xauthority}"
exec ../venv/bin/python -c 'import runpy, sys; sys.path[:0] = ["."]; sys.path.append("/usr/lib/python3/dist-packages"); sys.argv = ["main.py"]; runpy.run_path("main.py", run_name="osworld_guest")["app"].run(host="127.0.0.1", port=5000, debug=False, threaded=True)'
