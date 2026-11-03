"""The website state verifier: each check reads one value of one session of one site, as the site's /go?sid= reports it
through the gateway, and compares it; the step is refused at put for an op it doesn't know, and for a site the env
version it pins doesn't serve."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.verifiers import website_state_verifier
from agent_env.task_step.task_steps.verifiers.website_state_verifier import StateCheck, WebsiteStateVerifierTaskStep

APPS = {"mail": {"s1": {
    "initial_state": {"inbox": [{"id": 1, "read": False}], "user": {"name": "Ada"}},
    "current_state": {"inbox": [{"id": 1, "read": True}, {"id": 2, "read": False}], "user": {"name": "Ada"},
                      "drafts": {"d1": "hi"}, "contacts": {"ada@example.com": {"vip": True}}},
    "state_diff": {"inbox": {"initial": [{"id": 1, "read": False}],
                             "current": [{"id": 1, "read": True}, {"id": 2, "read": False}]},
                   "drafts.d1": {"current": "hi"}}}}}


@pytest.mark.parametrize(("check", "passed"), [
    (StateCheck(site="mail", sid="s1", path="user.name", op="eq", value="Ada"), True),
    (StateCheck(site="mail", sid="s1", path="inbox.0.read", op="eq", value=True), True),
    (StateCheck(site="mail", sid="s1", path="inbox.-1.id", op="eq", value=2), True),
    (StateCheck(site="mail", sid="s1", path="inbox", op="ge", value=2, reduce="len"), True),
    (StateCheck(site="mail", sid="s1", path="inbox", op="eq", value=1, reduce="len", source="initial_state"), True),
    (StateCheck(site="mail", sid="s1", path="drafts", op="eq", value=["d1"], reduce="keys"), True),
    (StateCheck(site="mail", sid="s1", path="drafts.d1", op="contains", value="h", source="state_diff"), False),
    (StateCheck(site="mail", sid="s1", path="drafts.d1", op="eq", value={"current": "hi"}, source="state_diff"), True),
    (StateCheck(site="mail", sid="s1", path="", op="eq", value=2, reduce="len", source="state_diff"), True),
    (StateCheck(site="mail", sid="s1", path="user.tz", op="eq", value=None), True),
    (StateCheck(site="mail", sid="s1", path="user.tz", op="lt", value=3), False),
    (StateCheck(site="mail", sid="s1", path="user.name", op="in", value=["Ada", "Grace"]), True),
    (StateCheck(site="mail", sid="s1", path="user.name", op="gt", value=1), False),
    (StateCheck(site="mail", sid="s1", path="inbox.5", op="ne", value=None), False),
    (StateCheck(site="mail", sid="s1", path="inbox.--1", op="eq", value=None), True),
    (StateCheck(site="mail", sid="s2", path="", op="eq", value=False, reduce="bool"), True),
    (StateCheck(site="mail", sid="s1", path=r"contacts.ada@example\.com.vip", op="eq", value=True), True),
    (StateCheck(site="mail", sid="s1", path="contacts.ada@example.com.vip", op="eq", value=None), True),
])
def test_a_check_reads_its_value_and_compares_it(check, passed):
    assert check.result(APPS)["result"] is passed


def test_a_check_of_a_session_with_no_state_says_so():
    row = StateCheck(site="mail", sid="s9", path="user.name", op="eq", value="Ada").result(APPS)
    assert (row["result"], row["actual"], row["detail"]) == (False, None, "'mail' has no session 's9'")
    assert row["criterion"] == "mail[s9].current_state.user.name eq 'Ada'"


def test_every_check_that_names_an_unknown_op_source_or_reduce_is_refused_at_once():
    with pytest.raises(ValueError) as refused:
        WebsiteStateVerifierTaskStep(id="v", version=None, env_id="office", checks=[
            {"site": "mail", "path": "a", "op": "matches"},
            {"site": "mail", "path": "b", "op": "eq", "source": "final_state", "reduce": "sum"}])
    assert str(refused.value).count("check of 'mail'") == 3


def test_the_step_round_trips_and_is_registered():
    step = WebsiteStateVerifierTaskStep(id="v", version=None, env_id="office", score_aggregator="weighted_average",
                                       checks=[{"site": "mail", "sid": "s1", "path": "inbox", "op": "ge", "value": 2,
                                                "reduce": "len", "weight": 2.0}])
    assert WebsiteStateVerifierTaskStep.from_dict(step.to_dict()).to_dict() == step.to_dict()
    assert get_task_step_registry()["website_state_verifier"] is WebsiteStateVerifierTaskStep


def _site(name):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id=f"{name}-server", description="", image_name=f"registry.example/{name}@sha256:" + "d" * 64))
    return WebsiteEnv.put(id=f"{name}-site", server_docker_image_artifact=image, environment_name=name)


def test_put_refuses_a_check_of_a_site_the_pinned_env_version_doesnt_serve(local_stores):
    mail, calendar = _site("mail"), _site("calendar")
    MultiEnv.put(id="office", mcp_server_envs=[], website_envs=[mail, calendar])
    MultiEnv.put(id="office", mcp_server_envs=[], website_envs=[calendar])
    with pytest.raises(ValueError, match=r"'notes' is not a site of office v1, whose sites are \['calendar', 'mail'\]"):
        WebsiteStateVerifierTaskStep.put(id="v", env_id="office", env_version=1, checks=[
            {"site": "notes", "path": "", "op": "eq"}, {"site": "mail", "path": "", "op": "eq"}])
    stored = WebsiteStateVerifierTaskStep.put(id="v", env_id="office", env_version=1, retry_config={
        "retry_from_step_id": "deploy"}, checks=[{"site": "mail", "path": "", "op": "eq"}])
    assert stored.version == 1 and stored.retry_config.retry_from_step_id == "deploy"
    assert WebsiteStateVerifierTaskStep.from_dict(stored.to_dict()).env_version == 1
    pinned = WebsiteStateVerifierTaskStep(id="w", version=None, env_id="office", env_version=2,
                                          checks=[{"site": "mail", "path": "", "op": "eq"}])
    assert pinned.preflight() == ["website_state_verifier 'w': 'mail' is not a site of office v2, whose sites are "
                                  "['calendar']"]
    assert WebsiteStateVerifierTaskStep(id="u", version=None, env_id="office",
                                        checks=[{"site": "mail", "path": "", "op": "eq"}]).preflight() == []
    assert WebsiteStateVerifierTaskStep(id="s", version=None, env_id="mail-site", env_version=1,
                                        checks=[{"site": "mail", "path": "", "op": "eq"}]).preflight() == []
    plain = MultiEnv.put(id="plain", mcp_server_envs=[])
    assert "with no website" in WebsiteStateVerifierTaskStep(id="y", version=None, env_id=plain.id, checks=[]).preflight()[0]
    assert WebsiteStateVerifierTaskStep(id="x", version=None, env_id="missing", checks=[]).preflight() == []


def test_execute_reads_each_sites_go_through_the_gateway_and_scores_the_checks(monkeypatch):
    by_site = {"mail": APPS["mail"], "calendar": {"s1": {"current_state": {"events": [1, 2]}}}}
    asked = []

    def gateway(request):
        site = request.url.path.split("/")[2]
        sid = request.url.params["sid"]
        asked.append((request.url.path, sid))
        return httpx.Response(200, json=by_site[site][sid])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(website_state_verifier.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(gateway), **kw))
    monkeypatch.setattr(website_state_verifier, "gateway_url_of", lambda deployed: "http://gateway")
    monkeypatch.setattr(website_state_verifier, "_sites", lambda env: {"mail", "calendar"})
    monkeypatch.setattr(website_state_verifier.Env, "get", lambda id, version=None: None)
    step = WebsiteStateVerifierTaskStep(id="v", version=None, env_id="office", verifier_id="office-check", checks=[
        {"site": "mail", "sid": "s1", "path": "inbox.0.read", "op": "eq", "value": True},
        {"site": "mail", "sid": "s1", "path": "drafts", "op": "eq", "value": 0, "reduce": "len"},
        {"site": "calendar", "sid": "s1", "path": "events", "op": "eq", "value": 2, "reduce": "len"}])
    context = TaskStepContext()
    context.deployed_envs.append(SimpleNamespace(env_id="office", env_version=1))

    asyncio.run(step.execute(context))

    assert asked == [("/svc/calendar/go", "s1"), ("/svc/mail/go", "s1")]
    verification = context.metadata["verifications"]["office-check"]
    assert [row["result"] for row in verification["results"]] == [True, False, True]
