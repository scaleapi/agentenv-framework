"""SkyRL's trainers on a bundle served as an OpenEnv env (``agent_env.openenv.build_app``, or the package ``openenv
import`` writes): ``python -m agent_env.training.skyrl URL OUT`` writes the server's tasks as SkyRL's parquet rows, each
with its tools in its system prompt, ``agent_env.training.skyrl.env.AgentEnvTextEnv`` plays each row's task as one
episode on the server, ``agent_env.training.skyrl.generator.AgentEnvGenerator`` plays again a rollout the env fails,
and ``python -m agent_env.training.skyrl.train`` runs SkyRL's trainer with both."""
