"""Miles' trainers (https://github.com/radixark/miles) on a bundle served as an OpenEnv env (``agent_env.openenv.build_app``,
or the package ``openenv import`` writes): ``python -m agent_env.training.miles URL OUT`` writes the server's tasks as
Miles' prompt data, and ``agent_env.training.miles.agent`` holds the agent function Miles calls for each sample, which
plays the row's task as one episode on the server through Miles' session server, with the reward hook and the
dynamic-sampling filter beside it."""
