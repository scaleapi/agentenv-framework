# AgentEnv on Vercel Sandbox

This candidate adds the `vercel` sandbox provider to AgentEnv. AgentEnv runs the task, connects the agent to its environment, collects the trajectory and scores the result. Vercel provides the separate sandboxes running the environment and agent.

The included demo exercises one complete evaluation: deploy an items environment behind a gateway with its database services, load `starting-item`, deploy a deterministic agent, ask it to add `vercel-review-complete` through MCP, and grade its response. The controller independently reads the final state, verifies the stored task and trajectory, reconnects, and tears down the successful run. The agent performs real MCP calls; it does not call a language model. This validates the integration, not model quality.

## Setup

Prerequisites: Git, Python 3.11.4 or newer, uv, Node.js and Vercel CLI 62.5.0 or newer (the demo was tested with 62.5.0). Use a Vercel project where you can create 4-vCPU/8-GB sandboxes for the build and environment, plus a 2-vCPU/4-GB agent. The demo builds Linux images in a fresh Vercel sandbox, so local Docker is not needed. Network access to the package and image registries referenced by the repository Dockerfiles is required.

Clone this branch to use the unreleased provider. The published PyPI package does not contain this candidate.

```bash
git clone --branch feat/vercel-sandbox https://github.com/vercel-labs/agentenv-framework.git
cd agentenv-framework
uv sync --frozen --extra vercel
export AGENT_ENV_CONFIG="$PWD/examples/vercel-sandbox/config.toml"
```

The dedicated config selects Vercel for environments and agents. The demo overrides the document and object stores with local stores in its output directory.

## Run with an existing Vercel login

If needed, run `vercel login`. Choose an existing project and team that you have access to. The wrapper uses the public `vercel project token` command and passes its short-lived OIDC token only to the child process. It does not print the token or write it to disk.

Use a fresh output directory for each attempt; the demo refuses to overwrite an existing one. Keep the dedicated configuration selected above so unrelated settings cannot affect the demo. Do not add explicit `token`, `team_id` or `project_id` fields for this OIDC route.

```bash
.venv/bin/python examples/vercel-sandbox/with_vercel_login.py \
  --scope YOUR_TEAM --project YOUR_PROJECT -- \
  .venv/bin/python examples/vercel-sandbox/run_e2e.py \
  --repo . --output /tmp/agentenv-vercel-e2e-run-1
```

The demo creates three temporary sandboxes: an image builder, an environment and an agent. Successful runs remove all three and verify subsequent lookups return 404. Failed runs stop and retain tracked sandboxes and record their identifiers for diagnosis. The runner also records handles reaching its termination guard during setup rollback. Failures before a handle reaches the runner can still follow the provider’s own creation-cleanup behavior. Retained resources are intentionally left for diagnosis; this demo does not automatically reap them.

The image-download fixture serves only disposable test images. Its fixed signing key is public test data; this fixture is not a production artifact store. The demo uses local SQLite and filesystem stores for run records and artifacts, and applies a restricted policy allowing the test package host and Vercel routes to the workload sandboxes. Scale's actual stores and network requirements need their own acceptance run.

## What passing means

Run with normal Python execution; the runner refuses `python -O` because its assertions are acceptance checks.

Expected final output:

```text
END-TO-END EVALUATION PASSED: score 1
```

The output folder retains:

- `success.json`: final pass, score, run identifier and cleanup assertions.
- `task.json`, `task-instance.json` and `context.json`: the five task steps and recorded result.
- `environment-state.json`: exactly `starting-item` and `vercel-review-complete`.
- `objects/`: collected task artifacts, including the agent trajectory.
- `resources.json`, `images.json` and `source-hashes.json`: resource, image and source provenance.
- `build.stdout`, `build.stderr` and `build-command.json`: image-build evidence.

The stdout/stderr stream can also be redirected to a local log. Output folders contain deployment coordinates and private local paths; review them before sharing. Keep these generated outputs outside the checkout.

## Review scope

The integration remains in-tree as requested, with optional Python SDK dependencies. CPU and memory are minimum requests rounded to supported shapes; fixed disk sizing is logged. OIDC and explicit configuration credentials are supported. This POC covers ephemeral runs and active reconnect. Persistent resume, credential brokering and model-quality benchmarking are separate work.

Recorded follow-ups include cleanup of commands still running during teardown, upload retry diagnostics, resource log severity and readiness behavior. Production workload acceptance, actual stores, CPU/memory semantics, optional OIDC and maintenance ownership remain for Scale to confirm.

See [RESULTS.md](RESULTS.md) for the validation scope and remaining acceptance work.
