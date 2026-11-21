"""A GTK app started at boot, before HCU enables accessibility on its first AX call."""
from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Packages, Ready, computer_app, linux


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64"]}])
@linux(packages=Packages(apt=["python3-gi", "gir1.2-gtk-3.0"]),
       launch=Launch(at="boot", command=["/usr/bin/python3", "window.py"],
                     window={"class": "HcuAccessibility"}, ready=Ready(window=True)),
       paths=[".hcu-accessibility"])
class AccessibilityProbe(ComputerAppEnvironment):
    pass
