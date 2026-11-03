"""slime's trainers (https://github.com/THUDM/slime) on a bundle served as an OpenEnv env (``agent_env.openenv.build_app``,
or the package ``openenv import`` writes): ``python -m agent_env.training.slime URL OUT`` writes the server's tasks as
slime's prompt data, and ``agent_env.training.slime.generate`` holds the generate function slime calls for each sample,
which plays the row's task as one episode on the server through a session on slime's OpenAI adapter, with the
dynamic-sampling filter beside it."""
