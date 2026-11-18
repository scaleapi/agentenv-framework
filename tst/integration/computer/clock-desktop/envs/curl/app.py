"""curl, which the base doesn't have, for the clock desktop test to reach a live site with on universe time."""
from agentenv_protocol.computer import ComputerAppEnvironment, Packages, computer_app, linux


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["curl"]))
class Curl(ComputerAppEnvironment):
    pass
