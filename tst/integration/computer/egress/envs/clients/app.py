"""TLS clients for the egress proxy tests: a JDK from Ubuntu's archive, an official Node build, which carries its own
OpenSSL so the TLS shim doesn't reach it, and curl. Each goes out through the egress proxy by the session's settings
alone: Java's HttpClient (Fetch.java) and Node's fetch (fetch.mjs) print the status a URL answers."""
from agentenv_protocol.computer import ComputerAppEnvironment, Download, Packages, RunScript, computer_app, linux

NODE = "24.22.2"
_NODE = {"amd64": ("x64", "48cc0e21e048790e54c0cf5749c2725005313d652651c40b7a21feb9697aaf75"),
         "arm64": ("arm64", "eb187837a2230d4047161ac0be008584e16b46f726e9c9490ee35dc6b17d9ba8")}


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["openjdk-21-jdk-headless", "curl", "xz-utils"]),
       downloads=[Download(url=f"https://nodejs.org/dist/v{NODE}/node-v{NODE}-linux-{name}.tar.xz", sha256=sha256,
                           to="node.tar.xz", arch=arch) for arch, (name, sha256) in _NODE.items()],
       run=[RunScript(script="setup.sh")])
class Clients(ComputerAppEnvironment):
    pass
