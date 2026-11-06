"""The registry image the integration tests' throwaway registries run: the mirror the local-backends workflow pulls before
the tests, so a test starts one without pulling from Docker Hub."""

REGISTRY_IMAGE = "mirror.gcr.io/library/registry:2"
