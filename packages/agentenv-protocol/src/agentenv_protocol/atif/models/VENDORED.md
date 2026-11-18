# Vendored ATIF models

These files are Harbor's ATIF models, from `harbor/models/trajectories/` in the Harbor 0.26.0 release on PyPI (`harbor-0.26.0-py3-none-any.whl`, sha256
`b7c752c5b51ae2ee1fc2ea30621493c7dc0b5a3d0157295bed4552115027baee`). Harbor is Apache-2.0; `LICENSE` beside this file is the license that wheel ships.

The only change is import paths: each `from harbor.models.trajectories...` import reads `from agentenv_protocol.atif.models...`. Nothing else differs. A unit
test turns the imports back and checks each file's sha256 against the table below.

Refresh with `python scripts/vendor_atif.py <version>` from the repository root, then review the diff and the fixtures.

| File | Upstream sha256 |
| --- | --- |
| `__init__.py` | `8eac32584166ff8f3a30d42027bc53793c4804fed7b07dcc4a94795557e11846` |
| `agent.py` | `ae26f91b1aa9cba2f9f308b3b1b6a8e92c0851bb191e4bd8df3042b49a0425c6` |
| `content.py` | `568f8ac708141b63b5facb363274b93d8bb76ae36250bcee656781f189ad3426` |
| `final_metrics.py` | `c4897c3b7f6d6be0d891d5fc9a72d13407d01c897c099e3ca195c50424734caf` |
| `metrics.py` | `c6d724d80d250dd27fefc4462a3f70aff2333fe42775382cb8073b889becf165` |
| `observation.py` | `2bb40e781053428e172af45110b0119bda4143594d592e414e9af0a60f057d28` |
| `observation_result.py` | `edbe26ca85aa8ef5f13a08a62d4d777d20832a7a5b67dcfa9162d6dbdaaf4c0a` |
| `step.py` | `6d8cc176449e0d5be46b40fdc9802bcb417f6b11d295c111e670055552abaa4a` |
| `subagent_trajectory_ref.py` | `207388b30eec20aaba3d21effb66f984caecfa0cb23dbaaa8fc055abd6347aea` |
| `tool_call.py` | `b28fc0526c45a5a253edaaaa305a8f53ac436e5fd44918fe4dadbe573214030b` |
| `trajectory.py` | `74c3aa74f3de38cdcdf3727b771c7ad75d66c2f5a6b43ec7982ed5b1e56b6fbf` |
