from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from agentenv_protocol.a2a_agent import (
    AGENT_CONFIG_V1,
    TRAJECTORY_V1,
    TRIGGERS_V1,
)
from starlette.testclient import TestClient

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(_EXAMPLES_DIR))


class FakeModelClient:
    def __init__(
        self,
        text: str = "Model-generated response",
        chunks: tuple[str, ...] = ("Model-generated ", "response"),
    ) -> None:
        self.text = text
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text, input_tokens=5, output_tokens=3)

    async def stream(self, **kwargs: Any):
        self.calls.append(kwargs)
        for chunk in self.chunks:
            yield chunk


def _load_example(name: str) -> ModuleType:
    path = _EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agentenv_example_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load example: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_advanced = _load_example("advanced_agent")
_basic = _load_example("basic_agent")
_custom = _load_example("custom_extensions_agent")
_multimodal = _load_example("multimodal_agent")
_reference = sys.modules["reference_model"]
_streaming = _load_example("streaming_agent")

AdvancedAgent = _advanced.AdvancedAgent
BasicAgent = _basic.BasicAgent
CustomExtensionsAgent = _custom.CustomExtensionsAgent
MultimodalAgent = _multimodal.MultimodalAgent
NOTES_V1 = _custom.NOTES_V1
StreamingAgent = _streaming.StreamingAgent


@pytest.mark.asyncio
async def test_reference_model_forwards_generic_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(self, url: str, **kwargs: Any) -> Response:
            captured["request"] = {"url": url, **kwargs}
            return Response()

    monkeypatch.setattr(_reference.httpx, "AsyncClient", Client)
    model = _reference.OpenAICompatibleModel(
        api_key="test-key",
        base_url="https://models.example/v1",
    )

    response = await model.complete(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"request_label": "docs"},
        api_key="request-scoped-key",
    )

    assert response.text == "ok"
    assert captured == {
        "client": {"timeout": 120},
        "request": {
            "url": "https://models.example/v1/chat/completions",
            "headers": {"Authorization": "Bearer request-scoped-key"},
            "json": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"request_label": "docs"},
            },
        },
    }


def _extension(card: dict[str, Any], uri: str) -> dict[str, Any]:
    return next(
        item for item in card["capabilities"]["extensions"] if item["uri"] == uri
    )


def _message(
    prompt: str,
    *,
    request_id: str = "request-1",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "kind": "message",
        "messageId": f"message-{request_id}",
        "role": "user",
        "parts": [{"kind": "text", "text": prompt}],
    }
    if metadata is not None:
        message["metadata"] = metadata
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/send",
        "params": {
            "message": message,
            "configuration": {"blocking": True},
        },
    }


def test_basic_example_uses_typed_config_and_framework_trajectory() -> None:
    model_client = FakeModelClient(text="The model says hello.")
    with TestClient(BasicAgent(model_client).create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        config = _extension(card, AGENT_CONFIG_V1.uri)
        supported = config["params"]["methods"]["set"]["request"]["supported"]
        assert {"model", "system_prompt", "max_input_chars", "provider_token"} <= set(
            supported
        )
        schema = config["params"]["methods"]["set"]["request"]["schema"]
        assert schema["properties"]["provider_token"]["writeOnly"] is True
        assert "task_id" not in supported

        configured = client.post(
            config["params"]["endpoint"],
            json={
                "model": "test-model",
                "system_prompt": "Answer clearly.",
                "provider_token": "test-provider-token",
            },
        )
        assert configured.status_code == 200
        assert client.get(config["params"]["endpoint"]).json() == {
            "config": {
                "model": "test-model",
                "system_prompt": "Answer clearly.",
                "provider_token": "***",
            }
        }

        response = client.post(
            "/a2a",
            json=_message("hello sdk", metadata={"request_label": "docs"}),
        )
        task = response.json()["result"]
        assert task["status"]["state"] == "completed"
        assert task["status"]["message"]["parts"][0]["text"] == (
            "The model says hello."
        )
        assert model_client.calls == [
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Answer clearly."},
                    {"role": "user", "content": "hello sdk"},
                ],
                "metadata": {"request_label": "docs"},
                "api_key": "test-provider-token",
            }
        ]

        trajectory = _extension(card, TRAJECTORY_V1.uri)
        trajectory_response = client.post(
            trajectory["params"]["endpoint"], json={"task_id": task["id"]}
        )
        assert trajectory_response.json()["trajectory"][0]["type"] == (
            "model_completion"
        )


def test_custom_extensions_example_advertises_and_runs_both_forms() -> None:
    model_client = FakeModelClient(text="Custom extension agent response")
    with TestClient(CustomExtensionsAgent(model_client).create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        custom = _extension(card, "urn:example:word-count/v1")
        assert set(custom["params"]["methods"]) == {"count"}
        assert client.post(
            "/ext/word-count/v1", json={"text": "one two three"}
        ).json() == {"words": 3}

        notes = _extension(card, NOTES_V1.uri)
        assert set(notes["params"]["methods"]) == {"create", "list"}
        assert client.post("/ext/notes/v1", json={"text": "read the card"}).json() == {
            "id": 1,
            "text": "read the card",
        }
        assert client.get("/ext/notes/v1").json() == {
            "notes": [{"id": 1, "text": "read the card"}]
        }
        task = client.post("/a2a", json=_message("Use the custom agent")).json()[
            "result"
        ]
        assert task["status"]["message"]["parts"][0]["text"] == (
            "Custom extension agent response"
        )


def test_advanced_example_uses_lifespan_and_reports_the_sdk_override() -> None:
    model_client = FakeModelClient(text="advanced model response")
    app = AdvancedAgent(model_client).create_app()
    assert app.state.agentenv_a2a.registry.conformance() == {
        "standard_operation_overrides": [
            {"uri": AGENT_CONFIG_V1.uri, "operation": "set"},
            {"uri": AGENT_CONFIG_V1.uri, "operation": "get"},
            {"uri": TRIGGERS_V1.uri, "operation": "register"},
            {"uri": TRIGGERS_V1.uri, "operation": "decide"},
            {"uri": TRIGGERS_V1.uri, "operation": "state"},
        ]
    }

    with TestClient(app) as client:
        assert app.state.reference_runtime == {"ready": True}
        card = client.get("/.well-known/agent.json").json()
        assert card["capabilities"]["streaming"] is False

        config = _extension(card, AGENT_CONFIG_V1.uri)
        assert (
            client.post(
                config["params"]["endpoint"], json={"response_prefix": "Configured"}
            ).status_code
            == 200
        )
        assert client.get(config["params"]["endpoint"]).json() == {
            "config": {"response_prefix": "Configured"}
        }

        registration = client.post(
            "/ext/triggers",
            json={
                "triggers": [
                    {
                        "id": "reference-step-one",
                        "when": {"type": "step", "turn": 1},
                        "actions": [{"type": "say", "text": "registered"}],
                    }
                ]
            },
        )
        assert registration.status_code == 200

        delegated = client.post("/ext/triggers/decide", json={"turn": 1})
        assert delegated.json() == {
            "parts": [{"kind": "text", "text": "registered"}],
            "done": False,
            "fired": ["reference-step-one"],
        }
        state = client.get("/ext/triggers").json()
        assert any(entry["kind"] == "fired" for entry in state["firing_log"])

        decision = client.post("/ext/triggers/decide", json={"turn": 3})
        assert decision.json() == {
            "parts": [],
            "done": True,
            "fired": ["reference-turn-limit"],
        }

        response = client.post("/a2a", json=_message("echo these words"))
        task = response.json()["result"]
        assert task["status"]["state"] == "completed"
        assert task["status"]["message"]["parts"][0]["text"] == (
            "Configured: advanced model response"
        )

    assert app.state.reference_runtime == {"ready": False}


def test_multimodal_example_accepts_an_image_and_returns_structured_data() -> None:
    model_client = FakeModelClient(text="The image contains one red pixel.")
    with TestClient(MultimodalAgent(model_client).create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        assert card["defaultInputModes"] == ["text", "image/png"]
        assert card["defaultOutputModes"] == ["text", "application/json"]

        request = _message("Describe the attachment", request_id="multimodal")
        request["params"]["message"]["parts"].append(
            {
                "kind": "file",
                "file": {
                    "name": "pixel.png",
                    "mimeType": "image/png",
                    "bytes": "iVBORw0KGgo=",
                },
            }
        )
        response = client.post("/a2a", json=request)

    task = response.json()["result"]
    assert task["status"]["state"] == "completed"
    assert task["status"]["message"]["parts"] == [
        {
            "kind": "text",
            "text": "The image contains one red pixel.",
        },
        {
            "kind": "data",
            "data": {
                "structured_output": {
                    "prompt": "Describe the attachment",
                    "images": [
                        {
                            "name": "pixel.png",
                            "mime_type": "image/png",
                            "transport": "inline",
                        }
                    ],
                }
            },
        },
    ]
    assert model_client.calls[0]["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the attachment"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ],
    }


def test_streaming_example_advertises_and_emits_streaming_updates() -> None:
    model_client = FakeModelClient(chunks=("Streamed: ", "hello stream"))
    with TestClient(StreamingAgent(model_client).create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        assert card["capabilities"]["streaming"] is True

        request = _message("hello stream", request_id="stream-request")
        request["method"] = "message/stream"
        with client.stream("POST", "/a2a", json=request) as response:
            body = "\n".join(response.iter_lines())

    assert '"kind":"status-update"' in body
    assert '"text":"Streamed: hello stream"' in body
    assert '"state":"completed"' in body
