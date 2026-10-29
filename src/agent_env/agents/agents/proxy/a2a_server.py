"""Serves the proxy agent. ``AGENT_CARD`` is what agent-env's install/v1 reads. ``install_agent`` formats each command
with ``str.format``."""

from agentenv_protocol.a2a_agent import INSTALL_V1, a2a_agent, enable
from agentenv_protocol.a2a_agent.proxy import EXTENSIONS, IDENTITY, ProxyAgent, ProxyConfig, stopping_actions

_UNPACK = "mkdir -p {work_dir}/agent && tar -xzf {agent_ctx_tar} -C {work_dir}/agent"
# In the place of the agent it replaces: on a VM's host, its {host_workdir} as its {host_user}; in a container, as root in the
# container's working directory, as an agent installed there works.
_INSTALL_COMMANDS_HOST = [_UNPACK, "bash {work_dir}/agent/install.sh host {agent_name} {a2a_port} {host_workdir} {host_user}"]
_INSTALL_COMMANDS = [_UNPACK, "bash {work_dir}/agent/install.sh container {container} {a2a_port}"]


@a2a_agent(
    identity=IDENTITY,
    config=ProxyConfig,
    extensions=(
        *EXTENSIONS,
        enable(INSTALL_V1, install_commands=_INSTALL_COMMANDS,
               required_params=["container", "work_dir", "agent_ctx_tar", "a2a_port"],
               install_commands_host=_INSTALL_COMMANDS_HOST,
               required_params_host=["work_dir", "agent_ctx_tar", "a2a_port", "agent_name", "host_workdir", "host_user"],
               a2a_port=8000),
    ),
    lifespan=stopping_actions,
)
class InstalledProxy(ProxyAgent):
    pass


agent = InstalledProxy()
app = agent.create_app()
AGENT_CARD = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
