"""``AgentEnvAgent``: a request it posts to its model server shows each tool output that is an image's data URL in a
user message after the output, which NeMo Gym's own converter turns into the chat image vLLM reads, while its other
requests pass as they are. CI runs it on the nemo-gym ``uv.lock`` pins and on the latest nemo-gym."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tst.util.capabilities import NEMO_GYM, missing_capability_reason

agent = pytest.importorskip("agent_env.training.nemo_gym.agent", reason=missing_capability_reason(NEMO_GYM),
                            exc_type=ImportError)
converter = pytest.importorskip("nemo_gym.responses_converter", reason=missing_capability_reason(NEMO_GYM))
openai_utils = pytest.importorskip("nemo_gym.openai_utils", reason=missing_capability_reason(NEMO_GYM))
server_utils = pytest.importorskip("nemo_gym.server_utils", reason=missing_capability_reason(NEMO_GYM))

IMAGE = "data:image/png;base64,iVBORw0KGgo="
ASKED = [{"role": "user", "content": "What's on the screen?"},
         {"type": "function_call", "call_id": "c1", "name": "computer_screenshot", "arguments": "{}"},
         {"type": "function_call_output", "call_id": "c1", "output": IMAGE}]


def test_a_tool_output_thats_an_image_reaches_the_model_in_a_user_message_after_it():
    played, post = _agent()

    asyncio.run(played.server_client.post(server_name="policy_model", url_path="/v1/responses",
                                          json=openai_utils.NeMoGymResponseCreateParamsNonStreaming(input=ASKED)))

    posted = post.await_args.kwargs["json"]
    chat = converter.VLLMConverter(return_token_id_information=False).responses_to_chat_completion_create_params(
        openai_utils.NeMoGymResponseCreateParamsNonStreaming(**posted))
    assert [(message["role"], message.get("content")) for message in chat.messages][2:] == [
        ("tool", agent.SHOWN), ("user", [{"type": "image_url", "image_url": {"url": IMAGE, "detail": "auto"}}])]
    assert agent.shown([*ASKED, {"role": "user", "content": "And now?"}])[:4] == posted["input"]


def test_requests_to_other_servers_and_without_images_pass_as_they_are():
    played, post = _agent()
    asked = openai_utils.NeMoGymResponseCreateParamsNonStreaming(input=ASKED[:1])

    asyncio.run(played.server_client.post(server_name="dock", url_path="/computer_screenshot", json={"x": 1}))
    asyncio.run(played.server_client.post(server_name="policy_model", url_path="/v1/responses", json=asked))

    assert [call.kwargs["json"] for call in post.await_args_list] == [{"x": 1}, asked]


def _agent():
    client = MagicMock(spec=server_utils.ServerClient)
    client.post = post = AsyncMock()
    played = agent.AgentEnvAgent(
        config=agent.SimpleAgent.model_fields["config"].annotation(
            host="127.0.0.1", port=0, entrypoint="app.py", name="dock_agentenv_agent",
            resources_server={"type": "resources_servers", "name": "dock"},
            model_server={"type": "responses_api_models", "name": "policy_model"}),
        server_client=client)
    return played, post
