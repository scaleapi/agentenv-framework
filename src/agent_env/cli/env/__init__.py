import click

from .deploy import deploy
from .gateway import gateway
from .get_instance import get_instance
from .mcp_server import mcp_server
from .multi import multi
from .service_db import service_db
from .snapshot import snapshot
from .state import init_env_state, teardown_env_state
from .website import website
from .website_browser import website_browser


@click.group()
def env():
    """Environment commands."""
    pass


env.add_command(deploy)
env.add_command(gateway)
env.add_command(get_instance)
env.add_command(init_env_state)
env.add_command(mcp_server)
env.add_command(multi)
env.add_command(service_db)
env.add_command(snapshot)
env.add_command(teardown_env_state)
env.add_command(website)
env.add_command(website_browser)
