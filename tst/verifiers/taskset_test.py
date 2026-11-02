"""verifiers' NeMo Gym taskset, prime-rl's, plays a written server's rows against ``serve`` with the null harness:
``vf-eval`` seeds each row, calls its tools natively, and takes the reward from verify. A rollout the environment
fails is masked, and with ``--masked-as-error`` it comes back failed, which prime-rl leaves out of training, rather
than scored 0. The model is a streamed stand-in. CI runs it on verifiers 0.4.0 and on verifiers' main branch."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_env.training.nemo_gym.prepare import write_server
from tst.nemo_gym.server_test import _free_port, _up
from tst.unit.training.nemo_gym_test import PLAYED
from tst.util.capabilities import NEMO_GYM, VERIFIERS, missing_capability_reason

pytest.importorskip("agent_env.training.nemo_gym.server", reason=missing_capability_reason(NEMO_GYM),
                    exc_type=ImportError)
VF_EVAL = shutil.which("vf-eval")
pytestmark = pytest.mark.skipif(VF_EVAL is None, reason=missing_capability_reason(VERIFIERS))


def test_vf_eval_scores_a_played_row_and_fails_a_masked_one(serve, tmp_path, teardowns):
    bundle_server = serve({"a": PLAYED})
    folder = write_server(bundle_server.url, tmp_path / "gym")
    row = json.loads((folder / "data" / "all.jsonl").read_text().splitlines()[0])
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({**row, "task_id": "b"}) + "\n")
    port, model = _free_port(), _model()

    served = subprocess.Popen([sys.executable, "-m", "agent_env.training.nemo_gym", "serve", str(folder), "--port",
                               str(port), "--masked-as-error"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url = _up(served, f"http://127.0.0.1:{port}")
        (tmp_path / "eval.toml").write_text(_config(tmp_path, url, model.server_address[1]))
        evaluated = subprocess.run([VF_EVAL, "@", str(tmp_path / "eval.toml"), "-n", "2", "-r", "1", "--no-push"],
                                   env={**os.environ, "STAND_IN_API_KEY": "unused"}, capture_output=True, text=True,
                                   timeout=600)
    finally:
        served.terminate()
        served.wait(timeout=30)
        model.shutdown()
    bundle_server.stop()

    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    traces = {trace["task"]["data"]["name"]: trace for trace in _traces(tmp_path / "out")}
    played, masked = traces["rows:0"], traces["rows:1"]
    assert (played["ok"], played.get("dropped"), played["rewards"]["nemo_gym"]["score"]) == (True, None, 1.0)
    assert not masked["ok"]
    assert "503 Service Unavailable" in masked["errors"][-1]["message"]
    assert len(teardowns.finished) == 3


def _config(tmp_path, url, model_port):
    return f"""model = "stand-in"
output_dir = "{tmp_path / 'out'}"

[client]
base_url = "http://127.0.0.1:{model_port}/v1"
api_key_var = "STAND_IN_API_KEY"

[env.taskset]
id = "nemo-gym"
dataset = "{tmp_path / 'rows.jsonl'}"

[env.taskset.task]
resources_url = "{url}"

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "subprocess"
"""


def _traces(out):
    episodes = [json.loads(line) for path in out.rglob("traces.jsonl") for line in path.read_text().splitlines()]
    return [trace for episode in episodes for trace in episode["traces"]]


class _StandIn(BaseHTTPRequestHandler):
    """An OpenAI-compatible model, streamed as verifiers asks: it adds a crate, then replies once the tool answered."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answered = any(message.get("role") == "tool" for message in body["messages"])
        delta = ({"role": "assistant", "content": "Added a crate."} if answered else
                 {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_add", "type": "function", "function": {
                     "name": "add_item", "arguments": json.dumps({"item": "crate"})}}]})
        head = {"id": "stand-in", "object": "chat.completion.chunk", "created": int(time.time()), "model": "stand-in"}
        events = [head | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                  head | {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if answered else "tool_calls"}]},
                  head | {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
        data = ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _model():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StandIn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

