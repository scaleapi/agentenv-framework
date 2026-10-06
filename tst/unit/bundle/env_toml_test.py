"""Envs written from a bundle's env folders: an MCP server, a website and a multi, each over the images built from
its folder or named in the store, named by its env.toml or its source's environment card, and refused before any
write when its env.toml can't be written."""

import json

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.plan import check_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.bundle.materialize import materialize
from agent_env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.store.routing import namespace_routing
from tst.unit.bundle._support import RefusingStore, layout, local_store, plan_of

ROOT = "@local/~/triage"
CARD = 'from agentenv_protocol import environment_card\n\n\n@environment_card(name="{}")\nclass Server:\n    pass\n'
DOCKERFILE = "FROM scratch\nCOPY . /app\n"


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


def _bundle(root, files, *deployed):
    layout(root, {**files, "tasks/t.json": json.dumps(
        [{"id": f"deploy-{name}", "type": "deploy_env", "env_id": name} for name in deployed])})
    return root


def _mcp(name, card=None):
    return {f"envs/{name}/Dockerfile": DOCKERFILE, f"envs/{name}/server.py": CARD.format(card or name)}


def _run(root, dry_run=False):
    return materialize(plan_of(root), dry_run=dry_run)


def _summary(materialization):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialization.writes}


def _store_image(id, version=None):
    with namespace_routing():
        return get_artifact_store().put_document(DockerImageArtifact(
            id=id, description=id, image_name=f"registry.example/{id}:v1", tar_gz_s3_url=f"s3://bucket/{id}.tar.gz"))


def _problems(root):
    with pytest.raises(BundleError) as caught:
        plan_of(root)
    return caught.value.problems


# MCP servers


def test_an_mcp_server_folder_is_written_over_the_image_built_from_it_named_by_its_card(bundle_dir, builds):
    _bundle(bundle_dir, _mcp("crm"), "crm")

    first = _summary(_run(bundle_dir))
    env = Env.get(f"{ROOT}/crm")

    assert first[f"{ROOT}/crm__env_image"] == (1, False, ("new",)) and first[f"{ROOT}/crm"] == (1, False, ("new",))
    assert isinstance(env, MCPServerEnv)
    assert (env.environment_name, env.env_provider_type, env.metadata) == ("crm", "gateway", {})
    assert (env.docker_image_artifact.id, env.docker_image_artifact.version) == (f"{ROOT}/crm__env_image", 1)
    assert [call["build"][:2] for call in builds] == [("Dockerfile", ["Dockerfile", "server.py"])]

    assert all(reused for _, reused, _ in _summary(_run(bundle_dir)).values()) and len(builds) == 1
    (bundle_dir / "envs/crm/server.py").write_text(CARD.format("crm") + "# edited\n")
    rebuilt = _summary(_run(bundle_dir))
    assert rebuilt[f"{ROOT}/crm__env_image"] == (2, False, ("files changed: server.py",))
    assert rebuilt[f"{ROOT}/crm"] == (2, False, (f"artifact {ROOT}/crm__env_image is written anew (v1 → v2)",))
    assert Env.get(f"{ROOT}/crm").docker_image_artifact.version == 2


def test_an_environment_name_in_env_toml_wins_over_the_card_and_a_dockerfile_elsewhere_names_where_to_look(
    bundle_dir, builds,
):
    _bundle(bundle_dir, {
        "envs/crm/docker/Dockerfile": DOCKERFILE, "envs/crm/docker/server.py": CARD.format("crm"),
        "envs/crm/other.py": CARD.format("not-this"), "envs/crm/env.toml": 'image = { dockerfile = "docker/Dockerfile" }\n',
        "envs/named/Dockerfile": DOCKERFILE, "envs/named/server.py": CARD.format("crm"),
        "envs/named/env.toml": 'environment_name = "tickets"\nenv_provider_type = "server"\n',
    }, "crm", "named")

    _run(bundle_dir)

    assert Env.get(f"{ROOT}/crm").environment_name == "crm"
    named = Env.get(f"{ROOT}/named")
    assert (named.environment_name, named.env_provider_type) == ("tickets", "server")


def test_an_env_over_a_store_image_pins_the_version_the_plan_read_and_is_rewritten_when_the_store_gains_one(
    bundle_dir, builds,
):
    _store_image("base")
    _bundle(bundle_dir, {"envs/crm/env.toml": 'image = "base"\nenvironment_name = "crm"\n'}, "crm")

    _run(bundle_dir)
    assert Env.get(f"{ROOT}/crm").docker_image_artifact.version == 1
    _store_image("base")
    rewritten = _summary(_run(bundle_dir))

    assert rewritten[f"{ROOT}/crm"] == (2, False, ("artifact base has a new version in the store (v1 → v2)",))
    assert Env.get(f"{ROOT}/crm").docker_image_artifact.version == 2
    assert builds == []


# Websites and multis


def test_a_website_builds_dockerfile_backend_and_dockerfile_frontend_and_takes_its_name_from_the_backend(
    bundle_dir, builds,
):
    _bundle(bundle_dir, {"envs/shop/Dockerfile.backend": DOCKERFILE, "envs/shop/Dockerfile.frontend": DOCKERFILE,
                         "envs/shop/server.py": CARD.format("shop"), "envs/shop/env.toml": 'type = "website"\n'}, "shop")

    _run(bundle_dir)
    env = Env.get(f"{ROOT}/shop")

    assert isinstance(env, WebsiteEnv) and env.environment_name == "shop"
    assert (env.backend_docker_image_artifact.id, env.frontend_docker_image_artifact.id) == (
        f"{ROOT}/shop__backend_image", f"{ROOT}/shop__frontend_image")
    assert [call["build"][0] for call in builds] == ["Dockerfile.backend", "Dockerfile.frontend"]


def test_a_multi_names_bundle_and_store_envs_and_is_rewritten_when_a_child_is(bundle_dir, builds):
    with namespace_routing():
        MCPServerEnv.put(id="mail", docker_image_artifact=_store_image("mail-image"), environment_name="mail")
    _bundle(bundle_dir, {
        **_mcp("crm"), "envs/shop/Dockerfile.backend": DOCKERFILE, "envs/shop/Dockerfile.frontend": DOCKERFILE,
        "envs/shop/env.toml": 'type = "website"\nenvironment_name = "shop"\n',
        "envs/suite/env.toml": 'type = "multi"\nmcp_server_envs = ["crm", "mail"]\nwebsite_envs = ["shop"]\nname = "suite"\n',
    }, "suite")

    _run(bundle_dir)
    suite = Env.get(f"{ROOT}/suite")
    assert isinstance(suite, MultiEnv) and suite.name == "suite"
    assert [(env.id, env.version) for env in [*suite.mcp_server_envs, *suite.website_envs]] == [
        (f"{ROOT}/crm", 1), ("mail", 1), (f"{ROOT}/shop", 1)]

    (bundle_dir / "envs/crm/server.py").write_text(CARD.format("crm") + "# edited\n")
    rewritten = _summary(_run(bundle_dir))
    assert rewritten[f"{ROOT}/suite"] == (2, False, (f"env {ROOT}/crm is written anew (v1 → v2)",))
    assert rewritten[f"{ROOT}/shop"][1] and rewritten[f"{ROOT}/shop__backend_image"][1]


# Refused before any write


def _website(name, toml=""):
    return {f"envs/{name}/Dockerfile.backend": DOCKERFILE, f"envs/{name}/Dockerfile.frontend": DOCKERFILE,
            f"envs/{name}/env.toml": f'type = "website"\nenvironment_name = "{name}"\n{toml}'}


@pytest.mark.parametrize("files, problem", [
    ({"envs/x/Dockerfile": DOCKERFILE},
     "envs/x: env.toml: environment_name isn't set, and the source its image is built from declares no "
     "@environment_card(name=...); set it"),
    ({"envs/x/Dockerfile": DOCKERFILE, "envs/x/a.py": CARD.format("a"), "envs/x/b.py": CARD.format("b")},
     "envs/x: env.toml: environment_name isn't set, and the source its image is built from declares several "
     "environment cards ('a', 'b'); set it"),
    ({"envs/x/env.toml": 'image = "base"\n'},
     "envs/x: env.toml: environment_name isn't set, and its image isn't built from this folder, so there's no source "
     "to read it from; set it"),
    ({**_mcp("x"), "envs/x/env.toml": '[metadata]\nowner = "me"\n'},
     "envs/x: env.toml: unknown key 'metadata'; a mcp_server env takes image, environment_name, env_provider_type, "
     "type and id"),
    ({**_mcp("x"), "envs/x/env.toml": 'environment_name = ""\n'}, "envs/x: env.toml: environment_name can't be empty"),
    ({**_mcp("x"), "envs/x/env.toml": 'environment_name = "gateway"\n'},
     "envs/x: env.toml: environment_name 'gateway' is one a gateway deploy names its own containers (db-mcp, gateway, "
     "pgweb, servicedb, website-browser); choose another"),
    (_website("x", 'env_provider_type = "server"\n'),
     "envs/x: env.toml: env_provider_type 'server' deploys one MCP server, not a website env"),
    ({**_mcp("x"), "envs/x/env.toml": 'image = { dockerfile = "Dockerfile", build_args = { A = "1" } }\n'},
     "envs/x: image: an image built from this folder takes only dockerfile, not build_args"),
    ({"envs/x/env.toml": 'type = "multi"\nname = "suite"\n'},
     "envs/x: env.toml: a multi env needs at least one env in mcp_server_envs or website_envs"),
    ({**_mcp("crm"), "envs/x/env.toml": 'type = "multi"\nmcp_server_envs = ["crm"]\nname = "my suite"\n'},
     "envs/x: env.toml: name must be non-empty with no whitespace, not 'my suite'"),
    ({**_website("shop"), "envs/x/env.toml": 'type = "multi"\nmcp_server_envs = ["shop"]\n'},
     "envs/x: mcp_server_envs[0]: 'shop' is this bundle's website, but this field takes mcp_server"),
    ({**_mcp("a", "crm"), **_mcp("b", "crm"), "envs/x/env.toml": 'type = "multi"\nmcp_server_envs = ["a", "b"]\n'},
     "envs/x: mcp_server_envs[1]: environment_name 'crm' is also mcp_server_envs[0]'s; each of a multi's "
     "mcp_server_envs needs its own"),
    ({"envs/x/env.toml": 'type = "gateway_server"\n'},
     "envs/x: a gateway_server env isn't written from a bundle: config names the one every deploy uses "
     "(default_gateway_env_id), and a run builds it when it's missing"),
    ({"envs/x/env.toml": 'type = "service_db"\n'},
     "envs/x: a service_db env isn't written from a bundle: config names the one every deploy uses "
     "(default_service_db_env_id), and a run builds it when it's missing"),
], ids=["no-card", "several-cards", "store-image-unnamed", "metadata", "empty-name", "reserved-name",
        "website-on-server", "build-args", "multi-without-envs", "multi-name", "multi-wrong-child", "multi-same-names",
        "gateway", "service-db"])
def test_an_env_toml_that_cant_be_written_is_refused_before_any_write(bundle_dir, files, problem):
    _store_image("base")
    _bundle(bundle_dir, files, "x")

    assert problem in _problems(bundle_dir)
    assert not local_store().path.exists()


def test_an_unknown_env_provider_type_is_refused_naming_the_known_ones(bundle_dir):
    _bundle(bundle_dir, {**_mcp("x"), "envs/x/env.toml": 'env_provider_type = "nope"\n'}, "x")

    (problem,) = _problems(bundle_dir)
    assert problem.startswith("envs/x: env.toml: env_provider_type: Unknown env_provider_type: 'nope' (expected one of")


def test_a_multi_naming_a_store_env_of_another_type_is_refused(bundle_dir):
    with namespace_routing():
        WebsiteEnv.put(id="web", backend_docker_image_artifact=_store_image("b"), frontend_docker_image_artifact=_store_image("f"),
                       environment_name="web")
    _bundle(bundle_dir, {"envs/x/env.toml": 'type = "multi"\nmcp_server_envs = ["web"]\n'}, "x")

    assert _problems(bundle_dir) == ("envs/x: mcp_server_envs[0]: 'web' is a website in the store, but this field takes "
                                     "mcp_server",)


def test_an_env_folder_whose_id_is_another_type_of_env_in_the_store_is_refused(bundle_dir):
    with namespace_routing():
        WebsiteEnv.put(id=f"{ROOT}/x", backend_docker_image_artifact=_store_image("b"),
                       frontend_docker_image_artifact=_store_image("f"), environment_name="x")
    _bundle(bundle_dir, _mcp("x"), "x")

    assert _problems(bundle_dir) == (f"envs/x: '{ROOT}/x' is a website env in the store, and an env keeps its type "
                                     "across versions; rename the folder",)


def test_a_check_reports_an_env_toml_problem_without_reading_a_store(bundle_dir, monkeypatch):
    _bundle(bundle_dir, {"envs/x/Dockerfile": DOCKERFILE}, "x")
    monkeypatch.setattr("agent_env.config.runtime.Config.get_document_store", lambda self: RefusingStore())

    with pytest.raises(BundleError) as caught:
        check_bundle(resolve_bundle(parse_bundle(bundle_dir)))

    assert caught.value.problems == ("envs/x: env.toml: environment_name isn't set, and the source its image is built "
                                     "from declares no @environment_card(name=...); set it",)
