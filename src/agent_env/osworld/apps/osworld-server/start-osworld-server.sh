#!/usr/bin/env bash
# Runs the OSWorld guest server on 127.0.0.1:5000 with the app's venv, restarting it when it exits. The OS's
# dist-packages go after the venv's, for AT-SPI's bindings, which pip can't install.
cd "$(dirname "$0")/payload"
export DISPLAY="${DISPLAY:-:0}"
export PYTHONUNBUFFERED=1
while true; do
    ../venv/bin/python -c 'import runpy, sys; sys.path.append("/usr/lib/python3/dist-packages"); sys.argv = ["main.py"]; runpy.run_path("main.py", run_name="__main__")'
    echo "osworld-server exited ($?); restarting in 5s" >&2
    sleep 5
done
