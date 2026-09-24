import click

from .add_skill import add_skill
from .deploy import deploy
from .get import get
from .get_instance import get_instance
from .put import put
from .validate import validate


@click.group(name="a2a-agent")
def a2a_agent():
    """A2A agent commands."""
    pass


a2a_agent.add_command(put)
a2a_agent.add_command(get)
a2a_agent.add_command(deploy)
a2a_agent.add_command(get_instance)
a2a_agent.add_command(validate)
a2a_agent.add_command(add_skill)
