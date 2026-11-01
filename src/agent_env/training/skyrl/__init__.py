"""SkyRL on a bundle served as an OpenEnv env: ``python -m agent_env.training.skyrl`` writes SkyRL's rows,
``env.AgentEnvTextEnv`` plays each row's task as one episode, ``generator.AgentEnvGenerator`` plays a failed rollout
again, and ``python -m agent_env.training.skyrl.train`` runs SkyRL's trainer with both."""
