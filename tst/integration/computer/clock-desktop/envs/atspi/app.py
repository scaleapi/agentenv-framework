"""AT-SPI's Python bindings, which the base doesn't have, for the clock desktop test to read the panel's clock as an
assistive technology reads it."""
from agentenv_protocol.computer import ComputerAppEnvironment, Packages, computer_app, linux


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["python3-gi", "gir1.2-atspi-2.0"]))
class Atspi(ComputerAppEnvironment):
    pass
