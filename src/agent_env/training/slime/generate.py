"""What slime calls for each sample of the rows ``python -m agent_env.training.slime`` writes: the generate function
(``--custom-generate-function-path agent_env.training.slime.generate.generate``) and the dynamic-sampling filter
(``--dynamic-sampling-filter-path agent_env.training.slime.generate.keep_scored_groups``). It runs in slime's own
environment, on its rollout's event loop."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, ClassVar

from slime.agent.adapters import OpenAIAdapter
from slime.agent.aiohttp_threaded import run_app_in_thread
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from agent_env.training._chat import keep_scored_groups, play

__all__ = ["generate", "keep_scored_groups"]


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> list[Sample]:
    """Plays the sample's row as one episode, the model chatting through a session of its own on slime's OpenAI
    adapter, which records its turns. Returns the session's turns as training samples carrying the episode's reward and
    ``play``'s metadata: one sample, or one per turn the adapter couldn't join to the turn before. In an eval it returns
    one sample per episode, carrying only the reward, so slime's eval reward is a mean over episodes, not turns. An
    episode with no recorded turn comes back as a single aborted sample, which ``keep_scored_groups`` leaves out with its
    group. slime cancels a rollout it no longer needs (``abort_mode``): its session is dropped and its episode closed."""
    adapter = _Adapter.of(args)
    sid = uuid.uuid4().hex
    adapter.adapter.open_session(
        sid, sampling_defaults=sampling_params, max_context_tokens=args.rollout_max_context_len or 0,
        apply_chat_template_kwargs={**args.apply_chat_template_kwargs, **(sample.apply_chat_template_kwargs or {})})
    try:
        played = await play(adapter.url, sample.prompt, {"user": sid}, sample.metadata)
        if evaluation:
            await adapter.adapter.drop_session(sid)
            return _bare(sample, played, Sample.Status.COMPLETED)
        samples = await adapter.adapter.finish_session(sid, base_sample=sample, reward=played["reward"],
                                                       extra_metadata=played)
    except asyncio.CancelledError:
        await adapter.adapter.drop_session(sid)
        raise
    return samples or _bare(sample, played, Sample.Status.ABORTED)


generate.abort_mode = "request"


def _bare(sample: Sample, played: dict[str, Any], status: Sample.Status) -> list[Sample]:
    """``sample`` with no turns to train on, as slime's own agent examples return one: a masked placeholder token, the
    episode's reward and metadata, and ``remove_sample``."""
    sample.tokens, sample.response, sample.response_length = [0, 0], "", 1
    sample.loss_mask, sample.rollout_log_probs = [0], [0.0]
    sample.reward, sample.remove_sample, sample.status = played["reward"], True, status
    sample.metadata = {**sample.metadata, **played}
    return [sample]


class _Adapter:
    """The rollout process's OpenAI adapter, in front of its SGLang router, reading the model's tool calls with
    ``--sglang-tool-call-parser``; made for the process's first sample."""

    served: ClassVar[_Adapter | None] = None

    @classmethod
    def of(cls, args: Any) -> _Adapter:
        if cls.served is None:
            cls.served = cls(args)
        return cls.served

    def __init__(self, args: Any):
        if not args.sglang_tool_call_parser:
            raise ValueError("slime's OpenAI adapter reads the model's tool calls with --sglang-tool-call-parser; pass "
                             "the parser for the model's chat template, qwen25 for Qwen2.5 and Qwen3")
        self.adapter = OpenAIAdapter(
            tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=args.sglang_tool_call_parser, reasoning_parser=args.sglang_reasoning_parser,
            rollout_args=args)
        app = run_app_in_thread(self.adapter.app, host="127.0.0.1", thread_name="agentenv-adapter",
                                runner_kwargs={"handler_cancellation": True})
        self.url = f"http://127.0.0.1:{app.port}"
