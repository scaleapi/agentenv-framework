"""An app for the probe test whose window, launched at boot, stays above every other window over the middle of the
screen, where the conformance fixture's grid is, as a dialog an app keeps on top does."""
from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Ready, computer_app, linux


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(launch=Launch(at="boot", command=["venv/bin/python", "window.py"], window={"class": "Covered"},
                     ready=Ready(window=True)))
class Covered(ComputerAppEnvironment):
    pass
