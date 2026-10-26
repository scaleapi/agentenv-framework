"""Attribution wire format, and each run's environment, for the gemini proxy.

The gemini CLI has no custom-header hook, so a2a_server.py proxies its model
calls and injects attribution itself. `build_attribution_headers`' `projectId:` /
`taskId:` tokens are parsed back out by agent-env's runtime attribution validator,
so a rename silently orphans gemini's spend.
"""

from tst.unit.agents.gemini.agent import gemini_bridge

# --- attribution wire format --------------------------------------------------


def test_both_ids_emit_customer_id_and_both_tags():
    headers = gemini_bridge.build_attribution_headers("proj-1", "task-1")
    assert headers["x-litellm-customer-id"] == "proj-1"
    assert headers["x-litellm-tags"] == "projectId:proj-1,taskId:task-1"


def test_project_only_omits_task_tag():
    headers = gemini_bridge.build_attribution_headers("proj-1", None)
    assert headers["x-litellm-tags"] == "projectId:proj-1"


def test_task_only_emits_no_customer_id():
    headers = gemini_bridge.build_attribution_headers(None, "task-1")
    assert headers == {"x-litellm-tags": "taskId:task-1"}


def test_nothing_to_attribute_is_empty():
    assert gemini_bridge.build_attribution_headers(None, None) == {}


def test_a_run_reaches_the_endpoint_through_its_own_proxy_route_and_settings(tmp_path):
    settings = tmp_path / "run.json"
    env = gemini_bridge.cli_env({"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": "https://gw"},
                                system_settings=settings, base_url="http://127.0.0.1:8000/litellm-gemini/t1")

    assert (env["GEMINI_API_KEY"], env["GOOGLE_GEMINI_BASE_URL"], env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"]) == (
        "k", "http://127.0.0.1:8000/litellm-gemini/t1", str(settings))


def test_an_endpoint_the_environment_names_is_left_alone(tmp_path):
    env = gemini_bridge.cli_env({"GEMINI_API_KEY": "own", "GOOGLE_GEMINI_BASE_URL": "https://own", "LITELLM_API_KEY": "k"},
                                system_settings=tmp_path / "run.json", base_url="http://127.0.0.1:8000/litellm-gemini/t1")

    assert (env["GEMINI_API_KEY"], env["GOOGLE_GEMINI_BASE_URL"]) == ("own", "https://own")
