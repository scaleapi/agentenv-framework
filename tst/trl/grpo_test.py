"""TRL's GRPOTrainer trains a step through ``agent_env.training.trl`` on CPU, against the OpenEnv server ``build_app``
makes over the Episode API's fake steps. TRL marks ``environment_factory`` experimental, so this pins what the helper
relies on: TRL resets an environment with its row, calls its tools, reads ``get_reward``, and trains on the rewards.
As in TRL's own environment tests, a tiny model's ``generate`` is stood in for. CI runs it against TRL 1.16, the oldest
that trains on CPU, and the latest; AsyncGRPO needs a vLLM server, so it isn't covered here."""

from unittest.mock import patch

import pytest

from tst.unit.training.trl_test import PLAYED
from tst.util.capabilities import TRL, missing_capability_reason

torch = pytest.importorskip("torch", reason=missing_capability_reason(TRL))
trl = pytest.importorskip("trl", reason=missing_capability_reason(TRL))

MODEL = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
ADD = '<tool_call>\n{"name": "add_item", "arguments": {"item": "crate"}}\n</tool_call>'
END = '<tool_call>\n{"name": "end_turn", "arguments": {"response": "Added a crate."}}\n</tool_call>'


def test_grpotrainer_trains_a_step_on_a_served_bundle_and_scores_every_rollout(served, teardowns, tmp_path,
                                                                                 monkeypatch):
    monkeypatch.setenv("TRL_EXPERIMENTAL_SILENCE", "1")
    env = served({"a": PLAYED, "b": PLAYED})
    rewards = []

    class Recorded(env.factory):
        def get_reward(self):
            rewards.append(super().get_reward())
            return rewards[-1]

    trainer = trl.GRPOTrainer(model=MODEL, train_dataset=env.dataset, environment_factory=Recorded, args=trl.GRPOConfig(
        output_dir=str(tmp_path), learning_rate=0.1, per_device_train_batch_size=4, num_generations=2, max_steps=1,
        max_tool_calling_iterations=4, shuffle_dataset=False, logging_steps=1, report_to="none", bf16=False,
        use_cpu=True))
    before = {name: param.clone() for name, param in trainer.model.named_parameters()}

    with patch.object(trainer.model, "generate", side_effect=_Policy(trainer.processing_class)):
        trainer.train()

    [logged] = [row for row in trainer.state.log_history if "tools/call_frequency" in row]
    assert sorted(rewards) == [0.0, 0.0, 1.0, 1.0]
    assert (logged["tools/call_frequency"], logged["tools/failure_frequency"], logged["rewards/Recorded/mean"]) == (
        pytest.approx(1.0), 0.0, pytest.approx(0.5))
    assert any(not torch.equal(param, trainer.model.get_parameter(name)) for name, param in before.items())
    env.close()
    assert len(teardowns.finished) == len(teardowns.started)


class _Policy:
    """Stands in for ``generate``: of each group, the first rollout adds a crate, then ends its turn, then stops, and the
    second answers without a tool, so its turn is ended empty and scores 0."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0

    def __call__(self, input_ids, **_):
        self.calls += 1
        if self.calls == 1:
            texts = [ADD if row % 2 == 0 else "I won't." for row in range(len(input_ids))]
        else:
            texts = [END if self.calls == 2 else "Done."] * len(input_ids)
        completions = [self.tokenizer(text + "<|im_end|>", add_special_tokens=False).input_ids for text in texts]
        width = max(map(len, completions))
        padded = [ids + [self.tokenizer.pad_token_id] * (width - len(ids)) for ids in completions]
        return torch.cat([input_ids, torch.tensor(padded, device=input_ids.device)], dim=-1)
