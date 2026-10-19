"""The ``agent_env.entity_refs`` import path still gives the declarations that moved to ``agent_env.entity.refs``."""

import agent_env.entity.refs as refs
import agent_env.entity_refs as old


def test_the_old_import_path_gives_the_same_declarations():
    assert {name: getattr(old, name) for name in old.__all__} == {name: getattr(refs, name) for name in old.__all__}
