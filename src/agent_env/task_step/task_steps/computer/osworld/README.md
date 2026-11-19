OSWorld tasks on agent-env computers, from upstream OSWorld ([xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld),
Apache-2.0) at commit `5123bfdbbc34b947bcffcff564abc62d0ab7072d` (`OSWORLD_COMMIT`). The `osworld_initialize` and
`osworld_evaluate` steps set a task up and grade it in the worker, never where the agent can reach it; this bundle ships
the computer app they reach the task's computer through. It has no tasks.

`envs/osworld-server/` is the `osworld-server` computer app, for an X11 desktop on Linux (`amd64` and `arm64`), such as
`ubuntu-24.04`: upstream's guest server (`desktop_env/server/main.py`, with `pyxcursor.py` and `screen_recording.py`),
downloaded when the computer installs the app and checked by sha256, run unmodified as the session user on
127.0.0.1:5000, without Werkzeug's debugger. The app's own server forwards `/osworld/<route>` to it, and its card
advertises that prefix as the endpoint of the `urn:agentenv:osworld-guest/v1` extension, which the computer serves
behind the env's control token. Name it in a computer env's `apps` as `osworld-server`, its alias, or by its id,
`@local/agentenv-framework/osworld/osworld-server`.

## Grading in the worker

The steps run upstream's own `SetupController`, getters and metrics at the same commit, unmodified, in the worker
(`runtime.py`). Upstream builds every URL as `http://{vm_ip}:{port}`, so each setup or check runs against loopback
ports of its own, where a forwarder sends every call on to the computer's `/osworld` endpoint with the env's control
token (`forwarder.py`). The worker holds that token and any judge's model key; the agent holds neither.

- **Upstream's source** is fetched on first use from GitHub's archive of the commit, checked by sha256, and kept under
  the cache root (`~/.cache/agentenv/osworld/<commit>`). An image can bake it in instead: run
  `python -m agent_env.task_step.task_steps.computer.osworld.upstream <folder>` when building it, and set
  `AGENT_ENV_OSWORLD_SOURCE=<folder>` (`upstream.py`).
- **What upstream imports** is this folder's `requirements.txt`, which a worker running the steps installs:
  `pip install -r <it>`. It isn't an extra, since agent-env's own code imports none of it. It leaves out easyocr,
  which pulls torch in for `compare_image_text` alone, so that metric fails.
- **LLM-judge funcs** (`agent_judge` and its `agent_judge_<format>` variants): agent-env ships no judge. The worker's
  deployment registers one with `evaluate.register_judge(func, judge)`, and it runs in the worker with the run's judge
  model (`user_overrides.judge_model`, else the func's `model` option, else the step's `judge_model`). A judge func with
  no judge registered scores 0, saying so.
- **Not supported:**
  - OSWorld-V2 tasks (`osworld_v2_task_path`), whose setup and evaluate are the task's own code; the steps fail saying so.
  - A setup or check that needs the computer's Chrome DevTools port (9222) or VLC's HTTP port (8080), which the app
    doesn't expose; it fails saying which.
  - Google Drive setups and upstream's proxy.
