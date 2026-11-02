.PHONY: help install lint unit-test int-test int-test-fast int-test-slow installer-test trl-test nemo-gym-test verl-test skyrl-test clean-install-test test clean

VENV := .venv
VENV_BIN := $(abspath $(VENV))/bin
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
SYSTEM_PYTHON ?= python3.11

# Put the venv on PATH (like activation) so subprocesses spawned by tests —
# e.g. generated CLI scripts using `#!/usr/bin/env python3` — resolve the venv
# interpreter and its dependencies.
export PATH := $(VENV_BIN):$(PATH)


help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install the package with dev dependencies
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ./packages/agentenv-protocol
	$(PIP) install -e ".[dev]"

lint: ## Lint with ruff, the check the unit job runs
	$(VENV_BIN)/ruff check

unit-test: ## Run the unit test suite in parallel (no external services)
	$(PYTHON) -m pytest tst/unit/ packages/agentenv-protocol/tests/ -n auto

# Integration tests. Requires Docker and a local OCI registry on :5000
# (docker run -d -p 5000:5000 public.ecr.aws/docker/library/registry:2);
# they run on the local default backends. Slow tests are explicitly marked with
# ``@pytest.mark.int_test_slow`` (or module-scope ``pytestmark``) in each
# test file — see tst/integration/conftest.py for the convention.
#
# Distribution mode differs by tier:
# - Fast: loadscope keeps class/module fixtures together; -n auto is safe.
# - Slow: serial. Several slow modules (gateway_test, task_steps_test,
#   env_test) build the *same* docker image tags from their
#   module-scoped fixtures. Under xdist parallel workers they race on the
#   shared docker daemon (and on hardcoded artifact IDs like
#   "task-step-test-deploy-multi"), producing ERROR-at-setup cascades that
#   look like real bugs but are pure concurrency. Running serial costs wall
#   clock but is the only way to keep slow runs reliable without rewriting
#   every module's fixtures with worker-unique tags. CI splits the slow tier
#   across runners instead (tst/util/shards.py). To run one runner's share:
#     AGENTENV_TEST_SHARD=2/8 make int-test-slow PYTEST_ARGS=--ignore=tst/integration/env/gateway/gateway_test.py
#     AGENTENV_TEST_SHARD=1/3 .venv/bin/python -m pytest -m int_test_slow tst/integration/env/gateway/gateway_test.py
INT_PYTEST_ARGS_FAST ?= --dist=loadscope --tb=short --durations=20 -p no:cacheprovider
INT_PYTEST_ARGS_SLOW ?= --tb=short --durations=20 -p no:cacheprovider
INT_FAST_WORKERS ?= auto

int-test: int-test-fast int-test-slow ## Run all integration tests (fast tier first, then slow tier)

int-test-fast: ## Run only the fast integration tests (skip @int_test_slow)
	$(PYTHON) -m pytest tst/integration/ -m 'not int_test_slow' -n $(INT_FAST_WORKERS) $(INT_PYTEST_ARGS_FAST) $(PYTEST_ARGS)

int-test-slow: ## Run only the @int_test_slow integration tests (serial to avoid docker / sandbox contention)
	$(PYTHON) -m pytest tst/integration/ -m 'int_test_slow' $(INT_PYTEST_ARGS_SLOW) $(PYTEST_ARGS)

# The installer tier runs the real pip, uv and pipx. It builds agentenv-framework from the checkout
# and installs offline; only the first run for a uv.lock downloads the dependencies, into .cache/.
installer-test: ## Run plugin add/remove through the real pip, uv and pipx, and the container journey (needs uv, pipx, Docker)
	$(PYTHON) -m pytest tst/installer -n auto --timeout=900 $(PYTEST_ARGS)

# The TRL contract tier trains one GRPOTrainer step on CPU through agent_env.training.trl, with a tiny model from the
# Hub. trl and torch aren't in the dev extra: install them first, as .github/workflows/trl.yml does.
trl-test: ## Run one GRPOTrainer step on CPU through the TRL helper (needs trl and CPU torch, and the Hub)
	$(PYTHON) -m pytest tst/trl --timeout=900 $(PYTEST_ARGS)

# The NeMo Gym tier plays the bundle's resources server in-process. nemo-gym needs Python 3.13.14 or later and isn't in
# the dev extra: sync it on that Python first (uv sync --extra dev --extra nemo-gym), as .github/workflows/nemo-gym.yml does.
nemo-gym-test: ## Play rollouts through NeMo Gym's resources server for a bundle (needs nemo-gym, on Python 3.13.14+)
	$(PYTHON) -m pytest tst/nemo_gym --timeout=900 $(PYTEST_ARGS)

# The verl tier plays a row through AgentEnvAgentLoop on verl itself, with Qwen3's tokenizer from the Hub. verl and
# torch aren't in the dev extra: install them first, as .github/workflows/verl.yml does.
verl-test: ## Play rows through the verl agent loop on verl itself (needs verl and CPU torch, and the Hub)
	$(PYTHON) -m pytest tst/verl --timeout=900 $(PYTEST_ARGS)

# The SkyRL tier plays rows through AgentEnvGenerator and AgentEnvTextEnv on SkyRL itself, with Qwen3's and Qwen3.5's
# tokenizers from the Hub. SkyRL, skyrl-gym and torch aren't in the dev extra, and SkyRL needs Python 3.12 or later:
# install them first, as .github/workflows/skyrl.yml does.
skyrl-test: ## Play rows through the SkyRL env and generator on SkyRL itself (needs SkyRL and CPU torch, and the Hub)
	$(PYTHON) -m pytest tst/skyrl --timeout=900 $(PYTEST_ARGS)

# The clean-install gate CI runs, on the same Python. The script gives every step an allowlisted
# environment, so nothing in this shell reaches the install it checks. A failed run keeps its folder.
clean-install-test: ## Build both distributions, install them clean from public PyPI and run hello twice (needs python3.11, uv, git)
	@work=$$(mktemp -d); \
	if $(SYSTEM_PYTHON) .github/scripts/clean_install.py --work "$$work"; then rm -rf "$$work"; \
	else status=$$?; echo "kept $$work to inspect"; exit $$status; fi

test: ## Run every tier: unit, integration and installer (needs Docker and a local OCI registry; tests needing uv, pipx, a model endpoint or a remote sandbox skip without them)
	$(PYTHON) -m pytest tst/ -v

clean: ## Remove the virtualenv and Python caches
	rm -rf $(VENV)
	find tst src -type d -name __pycache__ -prune -exec rm -rf {} +
