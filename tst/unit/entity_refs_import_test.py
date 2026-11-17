"""The ``agent_env.entity_refs`` import path still gives the declarations that moved to ``agent_env.entity.refs``."""

import agent_env.entity.refs as refs
import agent_env.entity_refs as old
import agent_env.task_step.registry as registry


def test_the_old_import_path_gives_the_same_declarations():
    moved = [name for name in old.__all__ if name != "step_ref_sites"]
    assert {name: getattr(old, name) for name in moved} == {name: getattr(refs, name) for name in moved}
    assert old.step_ref_sites is registry.step_ref_sites
