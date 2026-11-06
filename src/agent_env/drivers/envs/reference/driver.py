"""The reference computer driver, for an X11 desktop: computer/v1 as ``X11Driver`` sends it, mss capturing the screen and
XTEST, through python-xlib, sending the input. It needs ``agentenv-framework-protocol[computer,x11]`` where it runs."""
from __future__ import annotations

from agentenv_protocol.computer import X11Driver, computer_driver


@computer_driver(name="reference",
                 supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class ReferenceDriver(X11Driver):
    pass


if __name__ == "__main__":
    ReferenceDriver().serve()
