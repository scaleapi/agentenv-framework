"""What slime calls for each sample of the rows ``python -m agent_env.training.slime`` writes: the generate function
(``--custom-generate-function-path agent_env.training.slime.generate.generate``) and the dynamic-sampling filter
(``--dynamic-sampling-filter-path agent_env.training.slime.generate.keep_scored_groups``). It runs in slime's own
environment, on its rollout's event loop."""

from __future__ import annotations

import uuid
from typing import Any

from slime.agent.adapters import OpenAIAdapter
from slime.agent.aiohttp_threaded import run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from agent_env.training._chat import keep_scored_groups, play

__all__ = ["generate", "keep_scored_groups"]


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> list[Sample]:
    """Plays the sample's row as one episode, the model chatting through a session of its own on slime's OpenAI
    adapter, which records its turns. Returns the session's turns as training samples carrying the episode's reward and
    ``play``'s metadata: one sample, or one per turn the adapter couldn't join to the turn before. An episode with no
    recorded turn comes back as a single aborted sample, which ``keep_scored_groups`` leaves out with its group."""
    adapter = _Adapter(args)
    sid = uuid.uuid4().hex
    adapter.adapter.open_session(
        sid, sampling_defaults=sampling_params, max_context_tokens=args.rollout_max_context_len or 0,
        apply_chat_template_kwargs={**args.apply_chat_template_kwargs, **(sample.apply_chat_template_kwargs or {})})
    played = await play(adapter.url, sample.prompt, {"user": sid}, sample.metadata)
    samples = await adapter.adapter.finish_session(sid, base_sample=sample, reward=played["reward"],
                                                   extra_metadata=played)
    if samples:
        return samples
    sample.tokens, sample.response, sample.response_length = [0, 0], "", 1
    sample.loss_mask, sample.rollout_log_probs = [0], [0.0]
    sample.reward, sample.remove_sample, sample.status = played["reward"], True, Sample.Status.ABORTED
    sample.metadata = {**sample.metadata, **played}
    return [sample]


class _Adapter(metaclass=SingletonMeta):
    """The rollout process's OpenAI adapter, in front of its SGLang router, reading the model's tool calls with
    ``--sglang-tool-call-parser``."""

    def __init__(self, args: Any):
        if not args.sglang_tool_call_parser:
            raise ValueError("slime's OpenAI adapter reads the model's tool calls with --sglang-tool-call-parser; pass "
                             "the parser for the model's chat template, qwen25 for Qwen2.5 and Qwen3")
        self.adapter = OpenAIAdapter(
            tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=args.sglang_tool_call_parser, reasoning_parser=args.sglang_reasoning_parser,
            rollout_args=args)
        served = run_app_in_thread(self.adapter.app, host="127.0.0.1", thread_name="agentenv-adapter",
                                   runner_kwargs={"handler_cancellation": True})
        self.url = f"http://127.0.0.1:{served.port}"
