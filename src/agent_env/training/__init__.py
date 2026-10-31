"""Training on AgentEnv tasks with other projects' trainers, one module per trainer, each behind an extra of its own:
``agent_env.training.trl`` for TRL, ``agent_env.training.nemo_gym`` for NeMo Gym and the trainers that take its
environments, NeMo RL among them, and ``agent_env.training.verl`` for verl."""
