"""verl's trainers on a bundle served as an OpenEnv env (``agent_env.openenv.build_app``, or the package ``openenv
import`` writes): ``python -m agent_env.training.verl URL OUT`` writes the server's tasks as verl's parquet rows with the
config of the agent loop that plays them, and ``agent_env.training.verl.loop.AgentEnvAgentLoop``, which runs in verl's
agent loop workers, plays each row's task as one episode on the server and takes its reward."""
