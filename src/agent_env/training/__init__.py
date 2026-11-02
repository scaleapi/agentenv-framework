"""Training on AgentEnv tasks with other projects' trainers, one module per trainer, each behind an extra of its own:
``agent_env.training.trl`` for TRL, ``agent_env.training.nemo_gym`` for NeMo Gym and the trainers that take its
environments, NeMo RL among them, ``agent_env.training.verl`` for verl, and ``agent_env.training.skyrl`` for
SkyRL. ``agent_env.training.harness`` and ``agent_env.training.areal`` train a bundle's own agents instead, with TRL's
harness rollouts and AReaL's agent workflows; the second needs no extra."""
