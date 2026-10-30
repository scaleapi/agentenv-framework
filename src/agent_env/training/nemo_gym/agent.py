"""NeMo Gym's ``simple_agent`` for a resources server that answers a tool's image as a data URL (``images`` in its
config). A chat model's tool message carries text only, so each request to the model shows each such image in a user
message after its tool output, while the rollout keeps the tool output as it was: the data URL NeMo RL counts as the
turn's image. It runs in the venv NeMo Gym builds for it."""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

try:
    from responses_api_agents.simple_agent.app import SimpleAgent
except ModuleNotFoundError as exc:
    if exc.name not in ("responses_api_agents", "nemo_gym"):
        raise
    raise ImportError("agent_env.training.nemo_gym.agent requires pip install 'agentenv-framework[nemo-gym]', on "
                      "Python 3.13.14 or later as NeMo Gym is") from exc

#: What the model reads in place of a tool output that is an image, which the user message after it holds.
SHOWN = "The screenshot is in the next message."


class AgentEnvAgent(SimpleAgent):
    def model_post_init(self, context: Any) -> None:
        """Shows the model the images its requests' tool outputs hold, through the client's ``post``, which every
        version of ``simple_agent`` calls its model server with; the client is this server's own."""
        super().model_post_init(context)
        object.__setattr__(self.server_client, "post", _showing(self.server_client.post, self.config.model_server.name))


def shown(items: list[Any]) -> list[Any]:
    """``items`` with each tool output that is an image's data URL replaced by ``SHOWN`` and followed by a user message
    holding the image; the same items always give the same list, so each turn's prompt extends the last."""
    out = []
    for item in items:
        if _image_output(item):
            out += [{**item, "output": SHOWN},
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_image", "image_url": item["output"], "detail": "auto"}]}]
        else:
            out.append(item)
    return out


def _showing(post: Callable[..., Awaitable[Any]], model_server: str) -> Callable[..., Awaitable[Any]]:
    async def posting(server_name: str, url_path: str, **kwargs: Any) -> Any:
        if server_name == model_server and kwargs.get("json") is not None:
            body = kwargs["json"]
            body = body.model_dump(exclude_unset=True) if isinstance(body, BaseModel) else dict(body)
            if isinstance(items := body.get("input"), list) and any(map(_image_output, items)):
                kwargs["json"] = {**body, "input": shown(items)}
        return await post(server_name, url_path, **kwargs)

    return posting


def _image_output(item: Any) -> bool:
    return (isinstance(item, dict) and item.get("type") == "function_call_output"
            and isinstance(item.get("output"), str) and item["output"].startswith("data:image/"))
