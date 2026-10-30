"""NeMo Gym, and the trainers that take its environments (NeMo RL among them), on a bundle's tasks served as an OpenEnv
env: ``python -m agent_env.training.nemo_gym`` writes the bundle's resources server into a Gym root, with its config and
a JSONL file of rows per split, and ``server.AgentEnvResourcesServer`` is that server, which runs in the venv NeMo Gym
builds for it."""
